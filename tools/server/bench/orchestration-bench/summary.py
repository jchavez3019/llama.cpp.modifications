"""Summaries of orchestration benchmark runs.

Files of one repetition ``rep<k>`` in ``<root>/<device>/<model>/<config>/``:

- ``rep<k>.requests.jsonl``: one ``RequestRecord`` per request;
- ``rep<k>.meta.json``: run settings, and ``phone_clock_offset_s`` for phone runs;
- ``rep<k>.server.log``: server log (``-lv 4``);
- laptop: ``rep<k>.samples.jsonl`` (``LocalSample`` lines);
- phone: ``rep<k>.pftrace`` (Perfetto), ``rep<k>.thermal.jsonl`` (``PhoneThermalSample`` lines), and ``rep<k>.trace_metrics.json`` (cache written by this module).

Usage:
    python summary.py bench-results/phase0
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

from phone_trace import cached_trace_metrics
from runner import RequestRecord, output_path
from telemetry import LocalSample, PhoneThermalSample
from workload import FOREGROUND_STAGES, Stage

Metrics = dict[str, float]
D = TypeVar("D")

_P50 = 0.5
_P95 = 0.95
_PCT = 100.0
_MS_PER_S = 1000.0
_ALL_STAGES: tuple[Stage, ...] = (*FOREGROUND_STAGES, Stage.BACKGROUND)

# Server log lines (``-lv 4``) that mark cache events. See server-context.cpp and server-task.cpp.
LOG_EVENTS_: dict[str, str] = {
    "ram_restore": "found better prompt with f_keep",
    "ram_save": "saving prompt with length",
    "ram_evict": "removing oldest entry",
    "ckpt_restore": "restored context checkpoint",
    "ckpt_erase": "erasing old context checkpoint",
    "full_reprocess": "forcing full prompt re-processing",
    "idle_save": "saving idle slot to prompt cache",
    "empty_skip": "skipping, slot is empty",
    "purge": "purging slot",
}
_CACHE_UPDATE_RE = re.compile(r"prompt cache update took ([\d.]+) ms")

# Fork telemetry lines (info level): one per RAM cache save, and one per prompt start.
_CACHE_SAVE_RE = re.compile(r"prompt cache save: (\d+) tokens, ([\d.]+) MiB, ([\d.]+) ms")
_CACHE_USE_RE = re.compile(r"prompt cache use: src = (\w+), reused = (\d+), prefill = (\d+), load = ([\d.]+) ms, ([\d.]+) MiB")
CACHE_SOURCES_: tuple[str, ...] = ("ram", "tree", "slot", "none")


def quantile(values: list[float], q: float) -> float:
    """Returns the ``q`` quantile with linear interpolation, or NaN for an empty list."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _ratio(num: float, den: float) -> float:
    """Returns ``num / den``, or NaN when the denominator is not positive."""
    return num / den if den > 0 else float("nan")


def load_jsonl(path: Path, cls: type[D]) -> list[D]:
    """Reads a JSONL file into dataclass instances. Unknown keys are ignored, so older files with fewer fields still load."""
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    out: list[D] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(cls(**{k: v for k, v in json.loads(line).items() if k in names}))
    return out


def summarize_requests(records: list[RequestRecord]) -> Metrics:
    """Reduces the measured (non warm-up, no error) requests of one repetition to metrics.

    Per stage ``s``: ``s.hit`` (server cache hit rate), ``s.ideal`` (hit rate of a perfect prefix cache), ``s.prefill`` (mean prefilled tokens), ``s.ttft_p50`` and ``s.ttft_p95`` (ms). For the turn: ``turn_p50`` and ``turn_p95`` (sum of the foreground stage latencies, ms), ``n_turns``, ``n_errors``.

    :param records: All records of the repetition.
    :return: Flat metric dict.
    """
    measured = [r for r in records if not r.warmup and not r.error]
    out: Metrics = {"n_errors": float(sum(1 for r in records if r.error))}

    # Per-stage cache reuse and latency. Hit rates are token-weighted, so long prompts count more.
    for stage in _ALL_STAGES:
        rs = [r for r in measured if r.stage == stage]
        if not rs:
            continue
        out[f"{stage}.hit"] = _ratio(sum(r.cache_n for r in rs), sum(r.prompt_tokens for r in rs))
        out[f"{stage}.ideal"] = _ratio(sum(r.ideal_cached for r in rs), sum(r.client_prompt_tokens for r in rs))
        out[f"{stage}.prefill"] = statistics.fmean(r.prompt_n for r in rs)
        out[f"{stage}.ttft_p50"] = quantile([r.ttft_ms for r in rs], _P50)
        out[f"{stage}.ttft_p95"] = quantile([r.ttft_ms for r in rs], _P95)

    # Turn latency: only turns where every foreground stage succeeded.
    by_turn: dict[int, list[RequestRecord]] = {}
    for r in measured:
        if r.stage in FOREGROUND_STAGES:
            by_turn.setdefault(r.turn, []).append(r)
    turn_ms = [sum(r.e2e_ms for r in rs) for rs in by_turn.values() if len(rs) == len(FOREGROUND_STAGES)]
    out["n_turns"] = float(len(turn_ms))
    out["turn_p50"] = quantile(turn_ms, _P50)
    out["turn_p95"] = quantile(turn_ms, _P95)
    return out


def summarize_local_samples(samples: list[LocalSample], n_turns: int, t_begin: float, t_end: float) -> Metrics:
    """Reduces laptop samples in ``[t_begin, t_end]`` to peak memory and GPU energy per turn.

    GPU energy integrates the board power over time (trapezoid rule). It is only reported when every sample in the window has a power reading.

    :param samples: Laptop samples.
    :param n_turns: Measured turns in the window.
    :param t_begin: Window start (epoch seconds).
    :param t_end: Window end (epoch seconds).
    :return: Flat metric dict.
    """
    win = sorted((s for s in samples if not s.error and t_begin <= s.t <= t_end), key=lambda s: s.t)
    out: Metrics = {}
    rss = [s.rss_mib for s in win if s.rss_mib is not None]
    vram = [s.vram_mib for s in win if s.vram_mib is not None]
    if rss:
        out["peak_rss_mib"] = max(rss)
    if vram:
        out["peak_vram_mib"] = max(vram)

    power = [(s.t, s.gpu_power_w) for s in win if s.gpu_power_w is not None]
    if len(power) >= 2 and len(power) == len(win) and n_turns > 0:
        joules = sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(power, power[1:]))
        out["energy_j_per_turn"] = joules / n_turns
    return out


def summarize_thermal(samples: list[PhoneThermalSample], t_begin: float, t_end: float) -> Metrics:
    """Returns the peak big-core temperature and the peak thermal status inside the window."""
    win = [s for s in samples if not s.error and t_begin <= s.t <= t_end]
    out: Metrics = {}
    big = [s.temp_big for s in win if s.temp_big is not None]
    status = [s.thermal_status for s in win if s.thermal_status is not None]
    if big:
        out["peak_temp_big"] = max(big)
    if status:
        out["peak_thermal_status"] = float(max(status))
    return out


def summarize_log(log_text: str) -> Metrics:
    """Counts cache events in a server log and sums the prompt cache update time.

    :param log_text: Server log written with ``-lv 4``.
    :return: ``log.<event>`` counts and ``log.cache_update_ms``. When the log has fork telemetry lines, also ``log.save_ms``, ``log.save_mib``, ``log.load_ms``, ``log.load_mib``, and ``log.src_<source>`` prompt counts.
    """
    out: Metrics = {f"log.{name}": float(log_text.count(pattern)) for name, pattern in LOG_EVENTS_.items()}
    out["log.cache_update_ms"] = sum(float(m.group(1)) for m in _CACHE_UPDATE_RE.finditer(log_text))

    # Telemetry metrics exist only for logs that have the lines, so older runs show "-" instead of 0.
    uses = _CACHE_USE_RE.findall(log_text)
    if uses:
        saves = _CACHE_SAVE_RE.findall(log_text)
        out["log.save_ms"] = sum(float(ms) for _, _, ms in saves)
        out["log.save_mib"] = sum(float(mib) for _, mib, _ in saves)
        out["log.load_ms"] = sum(float(u[3]) for u in uses)
        out["log.load_mib"] = sum(float(u[4]) for u in uses)
        for src in CACHE_SOURCES_:
            out[f"log.src_{src}"] = float(sum(1 for u in uses if u[0] == src))
    return out


def summarize_rep(run_dir: Path, rep: str) -> Metrics:
    """Summarizes one repetition (files ``<rep>.*`` in ``run_dir``)."""
    prefix = run_dir / rep
    records = load_jsonl(output_path(prefix, ".requests.jsonl"), RequestRecord)
    out = summarize_requests(records)

    # Telemetry covers the measured turns only: from the first measured request start to the last end.
    measured = [r for r in records if not r.warmup]
    if measured:
        t_begin = min(r.t_start for r in measured)
        t_end = max(r.t_start + r.e2e_ms / _MS_PER_S for r in measured)
        n_turns = int(out["n_turns"])

        samples = output_path(prefix, ".samples.jsonl")
        if samples.exists():
            out.update(summarize_local_samples(load_jsonl(samples, LocalSample), n_turns, t_begin, t_end))
        thermal = output_path(prefix, ".thermal.jsonl")
        if thermal.exists():
            out.update(summarize_thermal(load_jsonl(thermal, PhoneThermalSample), t_begin, t_end))

        # Phone trace: energy from the power rails, server memory, and big-core frequency.
        trace = output_path(prefix, ".pftrace")
        if trace.exists() and n_turns > 0:
            meta = json.loads(output_path(prefix, ".meta.json").read_text(encoding="utf-8"))
            tm = cached_trace_metrics(trace, output_path(prefix, ".trace_metrics.json"), t_begin, t_end, float(meta.get("phone_clock_offset_s", 0.0)))
            if tm.energy_rails_j is not None:
                out["energy_j_per_turn"] = tm.energy_rails_j / n_turns
            if tm.energy_cpu_j is not None:
                out["cpu_energy_j_per_turn"] = tm.energy_cpu_j / n_turns
            if tm.peak_rss_mib is not None:
                out["peak_rss_mib"] = tm.peak_rss_mib
            if tm.mean_big_freq_mhz is not None:
                out["mean_big_freq_mhz"] = tm.mean_big_freq_mhz

    # Log counters include the warm-up turns, so the swap time is divided by all turns.
    log_path = output_path(prefix, ".server.log")
    if log_path.exists():
        out.update(summarize_log(log_path.read_text(encoding="utf-8", errors="replace")))
        n_all_turns = len({r.turn for r in records})
        out["log.cache_update_ms_per_turn"] = _ratio(out["log.cache_update_ms"], n_all_turns)
        for key in ("save_ms", "load_ms"):
            if f"log.{key}" in out:
                out[f"log.{key}_per_turn"] = _ratio(out[f"log.{key}"], n_all_turns)
    return out


def aggregate(reps: list[Metrics]) -> Metrics:
    """Returns the median of every metric over the repetitions that report it."""
    keys = sorted({k for m in reps for k in m})
    return {k: statistics.median([m[k] for m in reps if k in m]) for k in keys}


def _fmt(m: Metrics, key: str, scale: float = 1.0, digits: int = 0) -> str:
    """Formats one metric for a table cell, or ``-`` when it is missing or NaN."""
    v = m.get(key)
    if v is None or v != v:
        return "-"
    return f"{v * scale:.{digits}f}"


def markdown(root: Path) -> str:
    """Returns a stage table and a turn table for every ``<device>/<model>/<config>`` directory under ``root``."""
    rows: list[tuple[str, str, str, int, Metrics]] = []
    for run_dir in sorted(p for p in root.glob("*/*/*") if p.is_dir()):
        reps = sorted(p.name.removesuffix(".requests.jsonl") for p in run_dir.glob("*.requests.jsonl"))
        if reps:
            device, model, config = run_dir.relative_to(root).parts
            rows.append((device, model, config, len(reps), aggregate([summarize_rep(run_dir, r) for r in reps])))

    stage_cols = ["Device", "Model", "Config", "Stage", "Hit %", "Ideal %", "Prefill tok", "TTFT p50 ms", "TTFT p95 ms"]
    turn_cols = ["Device", "Model", "Config", "Reps", "Turn p50 ms", "Turn p95 ms", "Peak RSS MiB", "Peak VRAM MiB", "Peak BIG C", "Big MHz", "J/turn", "CPU J/turn",
                 "Swap ms/turn", "Save ms/turn", "Load ms/turn", "Src ram/tree/slot/none", "RAM restores", "RAM evictions", "Empty-slot skips", "Purges", "Full reprocess", "Errors"]
    stage_tbl = [_row(stage_cols), _row(["---"] * len(stage_cols))]
    turn_tbl = [_row(turn_cols), _row(["---"] * len(turn_cols))]
    for device, model, config, n_reps, m in rows:
        for stage in _ALL_STAGES:
            if f"{stage}.hit" in m:
                stage_tbl.append(_row([device, model, config, stage, _fmt(m, f"{stage}.hit", _PCT, 1), _fmt(m, f"{stage}.ideal", _PCT, 1),
                                       _fmt(m, f"{stage}.prefill"), _fmt(m, f"{stage}.ttft_p50"), _fmt(m, f"{stage}.ttft_p95")]))
        turn_tbl.append(_row([
            device, model, config, str(n_reps), _fmt(m, "turn_p50"), _fmt(m, "turn_p95"), _fmt(m, "peak_rss_mib"), _fmt(m, "peak_vram_mib"),
            _fmt(m, "peak_temp_big", digits=1), _fmt(m, "mean_big_freq_mhz"), _fmt(m, "energy_j_per_turn", digits=1), _fmt(m, "cpu_energy_j_per_turn", digits=1),
            _fmt(m, "log.cache_update_ms_per_turn"), _fmt(m, "log.save_ms_per_turn", digits=1), _fmt(m, "log.load_ms_per_turn", digits=1),
            "/".join(_fmt(m, f"log.src_{src}") for src in CACHE_SOURCES_), _fmt(m, "log.ram_restore"), _fmt(m, "log.ram_evict"), _fmt(m, "log.empty_skip"), _fmt(m, "log.purge"),
            _fmt(m, "log.full_reprocess"), _fmt(m, "n_errors"),
        ]))
    return "\n".join(stage_tbl) + "\n\n" + "\n".join(turn_tbl) + "\n"


def _row(cells: list[str]) -> str:
    """Joins cells into one Markdown table row."""
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize orchestration benchmark results as Markdown tables")
    parser.add_argument("root", type=Path, help="results root, laid out as <device>/<model>/<config>/rep<k>.*")
    print(markdown(parser.parse_args().root))


if __name__ == "__main__":
    main()
