"""Reads a phone Perfetto trace (see ``telemetry.PERFETTO_CONFIG``) for one measured window.

Clocks: trace timestamps are the phone's BOOTTIME. The request records use the laptop wall clock. A laptop time ``t`` maps to trace time as ``(t + phone_clock_offset_s) * 1e9 - (REALTIME - BOOTTIME)``, where the last term comes from the trace's own clock snapshot.

Power rails (Pixel on-device power monitor) are cumulative energy counters in microwatt-seconds, one per rail. The energy of a window is the counter difference between its ends, which stays correct while the phone charges.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from perfetto.trace_processor import TraceProcessor

_NS_PER_S = 1_000_000_000
_UWS_PER_J = 1_000_000.0
_BYTES_PER_MIB = 1024 * 1024
_KHZ_PER_MHZ = 1000.0

# Cortex-X1 and Cortex-A78 cores of the Tensor G2, where the server runs (taskset f0).
BIG_CORES_: tuple[int, ...] = (4, 5, 6, 7)
_RAIL_PREFIX = "power.rails."
_CPU_RAIL_PREFIX = "power.rails.cpu."
_SERVER_PROCESS_PATTERN = "%llama-server%"

Series = list[tuple[int, float]]


@dataclass(frozen=True)
class TraceWindowMetrics:
    """Phone metrics over one measured window. Fields are None when the trace has no data for them.

    :param energy_rails_j: Energy of all named power rails (SoC, memory, display, radios).
    :param energy_cpu_j: Energy of the three CPU rails.
    :param peak_rss_mib: Peak resident memory of the server process.
    :param mean_big_freq_mhz: Mean frequency of the cores in ``BIG_CORES_``. Lower values mean thermal throttling.
    :param window_s: Window length.
    """

    energy_rails_j: float | None
    energy_cpu_j: float | None
    peak_rss_mib: float | None
    mean_big_freq_mhz: float | None
    window_s: float


def counter_delta(series: Series, t0: int, t1: int) -> float | None:
    """Returns the increase of a cumulative counter between ``t0`` and ``t1``, with linear interpolation between samples.

    :param series: ``(timestamp, value)`` samples sorted by time.
    :param t0: Window start (same unit as the timestamps).
    :param t1: Window end.
    :return: The increase, or None if the samples do not cover the window.
    """
    if len(series) < 2 or t0 < series[0][0] or t1 > series[-1][0]:
        return None
    return _interpolate(series, t1) - _interpolate(series, t0)


def _interpolate(series: Series, t: int) -> float:
    """Returns the value of a sorted series at ``t`` by linear interpolation (``t`` must be inside the series)."""
    i = bisect.bisect_left([ts for ts, _ in series], t)
    if series[i][0] == t:
        return series[i][1]
    (ta, va), (tb, vb) = series[i - 1], series[i]
    return va + (vb - va) * (t - ta) / (tb - ta)


def to_trace_ns(t_laptop: float, phone_clock_offset_s: float, realtime_minus_boottime_ns: int) -> int:
    """Converts a laptop wall-clock time to a trace (phone BOOTTIME) timestamp.

    :param t_laptop: Laptop epoch seconds.
    :param phone_clock_offset_s: Phone wall clock minus laptop wall clock.
    :param realtime_minus_boottime_ns: Phone REALTIME minus BOOTTIME, from the trace clock snapshot.
    :return: Trace timestamp in nanoseconds.
    """
    return int((t_laptop + phone_clock_offset_s) * _NS_PER_S) - realtime_minus_boottime_ns


def _field(row: object, column: str) -> int | float | str:
    """Returns one column of a TraceProcessor row. Rows carry their columns as dynamic attributes."""
    value: int | float | str = getattr(row, column)
    return value


def _series(tp: TraceProcessor, query: str) -> dict[str, Series]:
    """Runs a query returning ``name, ts, value`` rows and groups them into sorted series by name."""
    out: dict[str, Series] = {}
    for row in tp.query(query):
        out.setdefault(str(_field(row, "name")), []).append((int(_field(row, "ts")), float(_field(row, "value"))))
    for s in out.values():
        s.sort()
    return out


def _sum_deltas(series: dict[str, Series], prefix: str, t0: int, t1: int) -> float | None:
    """Sums the window increase of every series whose name starts with ``prefix``; None if none covers the window."""
    deltas = [d for name, s in series.items() if name.startswith(prefix) and (d := counter_delta(s, t0, t1)) is not None]
    return sum(deltas) if deltas else None


def analyze_trace(path: Path, t_begin: float, t_end: float, phone_clock_offset_s: float) -> TraceWindowMetrics:
    """Reads the metrics of one measured window from a trace.

    :param path: Trace file.
    :param t_begin: Window start, laptop epoch seconds.
    :param t_end: Window end, laptop epoch seconds.
    :param phone_clock_offset_s: Phone wall clock minus laptop wall clock, measured when the run started.
    :return: The window metrics.
    """
    tp = TraceProcessor(trace=str(path))
    try:
        # Map the window onto the trace clock with the first clock snapshot.
        snap = {str(_field(r, "clock_name")): int(_field(r, "clock_value")) for r in tp.query("select clock_name, clock_value from clock_snapshot where snapshot_id = (select min(snapshot_id) from clock_snapshot)")}
        offset_ns = snap["REALTIME"] - snap["BOOTTIME"]
        t0 = to_trace_ns(t_begin, phone_clock_offset_s, offset_ns)
        t1 = to_trace_ns(t_end, phone_clock_offset_s, offset_ns)

        # Energy: rails are cumulative, so only the window ends matter.
        rails = _series(tp, f"select t.name as name, c.ts as ts, c.value as value from counter c join track t on c.track_id = t.id where t.name like '{_RAIL_PREFIX}%'")
        energy_rails = _sum_deltas(rails, _RAIL_PREFIX, t0, t1)
        energy_cpu = _sum_deltas(rails, _CPU_RAIL_PREFIX, t0, t1)

        # Memory and frequency: instantaneous values, reduced over the samples inside the window.
        rss = [float(_field(r, "value")) for r in tp.query(
            "select c.value as value from counter c join process_counter_track t on c.track_id = t.id join process p using (upid) "
            f"where t.name = 'mem.rss' and p.name like '{_SERVER_PROCESS_PATTERN}' and c.ts between {t0} and {t1}")]
        cores = ", ".join(str(c) for c in BIG_CORES_)
        freq = [float(_field(r, "value")) for r in tp.query(
            "select c.value as value from counter c join cpu_counter_track t on c.track_id = t.id "
            f"where t.name = 'cpufreq' and t.cpu in ({cores}) and c.ts between {t0} and {t1}")]
    finally:
        tp.close()

    return TraceWindowMetrics(
        energy_rails_j=energy_rails / _UWS_PER_J if energy_rails is not None else None,
        energy_cpu_j=energy_cpu / _UWS_PER_J if energy_cpu is not None else None,
        peak_rss_mib=max(rss) / _BYTES_PER_MIB if rss else None,
        mean_big_freq_mhz=sum(freq) / len(freq) / _KHZ_PER_MHZ if freq else None,
        window_s=t_end - t_begin,
    )


def cached_trace_metrics(trace: Path, cache: Path, t_begin: float, t_end: float, phone_clock_offset_s: float) -> TraceWindowMetrics:
    """Returns the window metrics of a trace, reading them from ``cache`` when it exists, else computing and writing them.

    Starting the trace processor takes a few seconds, so summaries over many runs reuse the cache.
    """
    if cache.exists():
        return TraceWindowMetrics(**json.loads(cache.read_text(encoding="utf-8")))
    metrics = analyze_trace(trace, t_begin, t_end, phone_clock_offset_s)
    cache.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
    return metrics
