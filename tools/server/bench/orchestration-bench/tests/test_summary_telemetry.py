import json
import math
from pathlib import Path

from runner import RequestRecord
from summary import load_jsonl, quantile, summarize_local_samples, summarize_log, summarize_requests, summarize_thermal
from telemetry import LocalSample, PhoneThermalSample, parse_hal_temperatures, parse_rss_mib, parse_thermal_status
from workload import Stage

# Excerpt of `dumpsys thermalservice` from a Pixel Fold (Android 17). The cached section holds an old BIG value.
DUMPSYS_ = """Thermal Status: 2
Cached temperatures:
\tTemperature{mValue=68.0, mType=0, mName=BIG, mStatus=0}
HAL Ready: true
Current temperatures from HAL:
\tTemperature{mValue=33.0, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=31.88008, mType=3, mName=VIRTUAL-SKIN, mStatus=0}
Current cooling devices from HAL:
\tCoolingDevice{mValue=0, mType=2, mName=BIG}
"""


def rec(turn: int, stage: Stage, cache_n: int, prompt_n: int, e2e_ms: float = 100.0, warmup: bool = False, error: str = "") -> RequestRecord:
    """Builds a request record whose ideal reuse is 10 tokens below its prompt size."""
    return RequestRecord(turn=turn, stage=stage, warmup=warmup, id_slot=0, t_start=float(turn), ttft_ms=e2e_ms / 2, e2e_ms=e2e_ms, prompt_tokens=cache_n + prompt_n,
                         cache_n=cache_n, prompt_n=prompt_n, ideal_cached=cache_n + prompt_n - 10, client_prompt_tokens=cache_n + prompt_n, error=error)


def test_quantile_interpolates_between_sorted_values() -> None:
    """
    Given: the values 1, 2, 3, 4 in random order.
    When: the median and the 95th percentile are computed.
    Then: the median is 2.5 and p95 is 3.85 (linear interpolation), and an empty list gives NaN.
    """
    assert quantile([4, 1, 3, 2], 0.5) == 2.5
    assert math.isclose(quantile([4, 1, 3, 2], 0.95), 3.85)
    assert math.isnan(quantile([], 0.5))


def test_hit_rate_is_token_weighted_and_skips_warmup_and_errors() -> None:
    """
    Given: stage 1 records of a warm-up turn, a failed request, and two measured turns with 90/100 and 0/100 cached tokens.
    When: the requests are summarized.
    Then: the hit rate is 90/200 (warm-up and failed requests are ignored, and big prompts weigh more), and one error is counted.
    """
    rs = [rec(0, Stage.ROUTER, 0, 100, warmup=True), rec(1, Stage.ROUTER, 90, 10), rec(2, Stage.ROUTER, 0, 100), rec(3, Stage.ROUTER, 100, 0, error="boom")]
    m = summarize_requests(rs)
    assert math.isclose(m["stage1.hit"], 90 / 200)
    assert math.isclose(m["stage1.ideal"], 180 / 200)
    assert m["n_errors"] == 1


def test_turn_latency_counts_only_turns_with_all_three_stages() -> None:
    """
    Given: turn 1 with all three stages (100 + 200 + 300 ms) and turn 2 where stage 3 failed.
    When: the requests are summarized.
    Then: only turn 1 counts, so the median turn latency is 600 ms.
    """
    rs = [rec(1, Stage.ROUTER, 0, 1, 100), rec(1, Stage.EXTRACTOR, 0, 1, 200), rec(1, Stage.SYNTHESIS, 0, 1, 300),
          rec(2, Stage.ROUTER, 0, 1, 100), rec(2, Stage.EXTRACTOR, 0, 1, 100), rec(2, Stage.SYNTHESIS, 0, 1, 100, error="x")]
    m = summarize_requests(rs)
    assert m["n_turns"] == 1
    assert m["turn_p50"] == 600


def test_gpu_energy_integrates_power_only_when_every_sample_has_power() -> None:
    """
    Given: laptop samples over 10 s with GPU power rising linearly from 10 W to 30 W, then the same samples plus one without a power reading.
    When: the samples are summarized for 2 turns.
    Then: the first gives 200 J in total (100 J per turn), the second reports no energy, and both report peak RSS and VRAM.
    """
    full = [LocalSample(t=0.0, rss_mib=100.0, vram_mib=900.0, gpu_power_w=10.0), LocalSample(t=5.0, gpu_power_w=20.0), LocalSample(t=10.0, rss_mib=300.0, gpu_power_w=30.0)]
    m = summarize_local_samples(full, 2, 0.0, 10.0)
    assert m["energy_j_per_turn"] == 100.0 and m["peak_rss_mib"] == 300.0 and m["peak_vram_mib"] == 900.0

    partial = full + [LocalSample(t=7.0, rss_mib=200.0)]
    m2 = summarize_local_samples(partial, 2, 0.0, 10.0)
    assert "energy_j_per_turn" not in m2 and m2["peak_rss_mib"] == 300.0


def test_thermal_summary_keeps_window_and_skips_failed_samples() -> None:
    """
    Given: thermal samples before, inside, and after the window, plus a failed sample inside it.
    When: the window is summarized.
    Then: only the samples inside count: peak big temperature 45 C and peak status 1.
    """
    samples = [PhoneThermalSample(t=0.0, thermal_status=3, temp_big=60.0), PhoneThermalSample(t=5.0, thermal_status=1, temp_big=45.0),
               PhoneThermalSample(t=6.0, error="adb timeout"), PhoneThermalSample(t=20.0, temp_big=70.0)]
    assert summarize_thermal(samples, 1.0, 10.0) == {"peak_temp_big": 45.0, "peak_thermal_status": 1.0}


def test_load_jsonl_reads_older_sample_lines_without_new_fields(tmp_path: Path) -> None:
    """
    Given: a samples file written before the GPU fields existed (only t, rss_mib, vram_mib).
    When: it is loaded as LocalSample lines.
    Then: the known fields load and the new fields default to None, so old laptop results still summarize.
    """
    path = tmp_path / "rep0.samples.jsonl"
    path.write_text(json.dumps({"t": 1.0, "rss_mib": 10.0, "vram_mib": 20.0}) + "\n", encoding="utf-8")
    assert load_jsonl(path, LocalSample) == [LocalSample(t=1.0, rss_mib=10.0, vram_mib=20.0)]


def test_log_summary_counts_cache_events_and_sums_update_time() -> None:
    """
    Given: a server log with two RAM restores, one eviction, one empty-slot skip, one purge, and two prompt cache updates.
    When: the log is summarized.
    Then: the counts match, and the update time is the sum (3.5 ms).
    """
    log = (" - found better prompt with f_keep = 0.9, f_sim = 1.0\n - found better prompt with f_keep = 0.5, f_sim = 1.0\n"
           " - cache size limit reached, removing oldest entry (size = 1.0 MiB)\nprompt cache update took 1.25 ms\nprompt cache update took 2.25 ms\n"
           "slot get_availabl: id  1 | task -1 |  - skipping, slot is empty\nsrv purging slot 2 with 100 tokens\n")
    m = summarize_log(log)
    assert m["log.ram_restore"] == 2 and m["log.ram_evict"] == 1 and m["log.empty_skip"] == 1 and m["log.purge"] == 1 and m["log.full_reprocess"] == 0
    assert m["log.cache_update_ms"] == 3.5


def test_hal_temperatures_ignore_cached_section_and_cooling_devices() -> None:
    """
    Given: dumpsys output where the cached section says BIG = 68 C and the HAL section says BIG = 33 C.
    When: the temperatures and the thermal status are parsed.
    Then: BIG is 33 C, VIRTUAL-SKIN is read, cooling devices are dropped, and the status is 2.
    """
    assert parse_hal_temperatures(DUMPSYS_) == {"BIG": 33.0, "VIRTUAL-SKIN": 31.88008}
    assert parse_thermal_status(DUMPSYS_) == 2


def test_rss_is_parsed_in_mib_from_proc_status() -> None:
    """
    Given: a /proc status text with VmRSS of 2048 kB, and one without VmRSS.
    When: the RSS is parsed.
    Then: the first gives 2 MiB and the second gives None.
    """
    assert parse_rss_mib("Name:\tllama-server\nVmRSS:\t    2048 kB\n") == 2.0
    assert parse_rss_mib("Name:\tx\n") is None
