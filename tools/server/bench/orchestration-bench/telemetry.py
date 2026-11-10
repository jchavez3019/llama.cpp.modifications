"""Telemetry recorded while a run executes.

Laptop: ``LocalSampler`` polls the server process RSS from ``/proc`` and the GPU through NVML (``pynvml``), every 100 ms.

Phone: ``PhoneTelemetry`` records a Perfetto system trace (power rails, battery, process memory, CPU frequency, all on one clock) and polls the SoC temperatures with ``dumpsys`` every few seconds, because thermal zones are not readable from the trace on a user build. ``phone_trace.py`` reads the trace afterwards.

Samples are written as one JSON line each, with the client wall-clock time ``t`` so they can be matched to the request records.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Callable, Generic, TypeAlias, TypeVar

import pynvml

# NVML device handle (a ctypes pointer). pynvml.c_nvmlDevice_t is a variable, so it cannot be used as a type.
_NvmlDevice: TypeAlias = "ctypes._Pointer[pynvml.struct_c_nvmlDevice_t]"

_KIB_PER_MIB = 1024
_BYTES_PER_MIB = 1024 * 1024
_MW_PER_W = 1000.0
_ADB_TIMEOUT_S = 20
_LOCAL_INTERVAL_S = 0.1
_THERMAL_INTERVAL_S = 5.0
_PERFETTO_STOP_POLL_S = 0.5
_PERFETTO_STOP_TIMEOUT_S = 30
# Keep tracing a little after the work ends, so the slowest polled counter (250 ms) has a sample after the window end.
_PERFETTO_TAIL_S = 1.0

_HAL_SECTION = "Current temperatures from HAL:"
_TEMP_RE = re.compile(r"Temperature\{mValue=([-\d.]+), mType=-?\d+, mName=([^,]+),")
_THERMAL_STATUS_RE = re.compile(r"^Thermal Status: (\d+)", re.MULTILINE)
_RSS_RE = re.compile(r"^VmRSS:\s+(\d+) kB", re.MULTILINE)

# The adb shell user may only read Perfetto configs from stdin and write traces here (SELinux).
REMOTE_TRACE_DIR = "/data/misc/perfetto-traces"

# Perfetto trace config for the phone. Power rails and battery counters are cumulative or instantaneous counters polled every 250 ms; they stay valid while the phone charges.
PERFETTO_CONFIG = """
buffers { size_kb: 65536 fill_policy: RING_BUFFER }
data_sources { config { name: "android.power" android_power_config {
  battery_poll_ms: 250
  battery_counters: BATTERY_COUNTER_CURRENT battery_counters: BATTERY_COUNTER_VOLTAGE battery_counters: BATTERY_COUNTER_CHARGE
  collect_power_rails: true } } }
data_sources { config { name: "linux.process_stats" process_stats_config { proc_stats_poll_ms: 500 } } }
data_sources { config { name: "linux.sys_stats" sys_stats_config {
  meminfo_period_ms: 1000 meminfo_counters: MEMINFO_MEM_AVAILABLE
  cpufreq_period_ms: 250 } } }
"""

T = TypeVar("T")


def parse_rss_mib(proc_status: str) -> float | None:
    """Returns the resident set size in MiB from the text of ``/proc/<pid>/status``, or None if absent."""
    m = _RSS_RE.search(proc_status)
    return int(m.group(1)) / _KIB_PER_MIB if m else None


def parse_hal_temperatures(dumpsys: str) -> dict[str, float]:
    """Returns the current temperatures (Celsius) by sensor name from ``dumpsys thermalservice``.

    Only the "Current temperatures from HAL" section is read. The "Cached temperatures" section can hold old values.

    :param dumpsys: Output of ``dumpsys thermalservice``.
    :return: Sensor name to temperature.
    """
    start = dumpsys.find(_HAL_SECTION)
    if start < 0:
        return {}

    # The section ends at the next header line (a line that does not start with whitespace).
    lines: list[str] = []
    for line in dumpsys[start + len(_HAL_SECTION):].splitlines()[1:]:
        if line and not line.startswith(("\t", " ")):
            break
        lines.append(line)
    return {m.group(2): float(m.group(1)) for m in _TEMP_RE.finditer("\n".join(lines))}


def parse_thermal_status(dumpsys: str) -> int | None:
    """Returns the Android thermal status (0 = none, higher = more throttling), or None if absent."""
    m = _THERMAL_STATUS_RE.search(dumpsys)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class LocalSample:
    """One laptop sample. GPU fields are None when NVML cannot report them (for example, power on some laptop GPUs).

    :param t: Client wall-clock time (epoch seconds).
    :param rss_mib: Server resident set size (physical RAM in use).
    :param vram_mib: GPU memory used by the server process.
    :param gpu_power_w: GPU board power.
    :param gpu_util_pct: GPU utilization over the last driver sample period.
    :param gpu_temp_c: GPU temperature.
    :param error: Error text when the sample failed, else empty.
    """

    t: float
    rss_mib: float | None = None
    vram_mib: float | None = None
    gpu_power_w: float | None = None
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    error: str = ""


@dataclass(frozen=True)
class PhoneThermalSample:
    """One phone temperature sample from ``dumpsys thermalservice``.

    :param t: Client wall-clock time (epoch seconds).
    :param thermal_status: Android thermal status (0 = none).
    :param temp_big: Cortex-X1 cluster temperature.
    :param temp_mid: Cortex-A78 cluster temperature.
    :param temp_little: Cortex-A55 cluster temperature.
    :param temp_skin: Virtual skin temperature.
    :param error: Error text when the sample failed, else empty.
    """

    t: float
    thermal_status: int | None = None
    temp_big: float | None = None
    temp_mid: float | None = None
    temp_little: float | None = None
    temp_skin: float | None = None
    error: str = ""


class _Sampler(Generic[T]):
    """Base class: calls ``sample()`` on a background thread every ``interval_s`` and writes each sample as a JSON line.

    Sampling is not synchronized with the requests. Each sample carries its time, and the summary keeps the samples inside the measured window.

    :param out_path: JSONL output file.
    :param interval_s: Target time between sample starts. A slow sample starts the next one at once.
    """

    def __init__(self, out_path: Path, interval_s: float) -> None:
        self.out_path = out_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._out: IO[str] | None = None

    def sample(self) -> T:
        """Returns one sample. Subclasses implement it and report failures in the sample's ``error`` field."""
        raise NotImplementedError

    def _loop(self) -> None:
        assert self._out is not None
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._out.write(json.dumps(asdict(self.sample())) + "\n")  # type: ignore[call-overload]
            self._out.flush()
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - t0)))

    def __enter__(self) -> "_Sampler[T]":
        self._out = self.out_path.open("w", encoding="utf-8")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._out is not None:
            self._out.close()


class LocalSampler(_Sampler[LocalSample]):
    """Samples RSS and GPU state of a local server process.

    :param pid: Server process id.
    :param out_path: JSONL output file.
    :param interval_s: Time between samples.
    :param gpu_index: NVML device index.
    """

    def __init__(self, pid: int, out_path: Path, interval_s: float = _LOCAL_INTERVAL_S, gpu_index: int = 0) -> None:
        super().__init__(out_path, interval_s)
        self.pid = pid
        self.gpu_index = gpu_index
        self._gpu: _NvmlDevice | None = None
        self._nvml_ready = False

    def __enter__(self) -> "LocalSampler":
        # NVML is optional: without a driver the sampler still records RSS.
        try:
            pynvml.nvmlInit()
            self._nvml_ready = True
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except pynvml.NVMLError:
            self._gpu = None
        super().__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        super().__exit__(exc_type, exc, tb)
        if self._nvml_ready:
            pynvml.nvmlShutdown()
            self._nvml_ready = False

    def sample(self) -> LocalSample:
        t = time.time()
        try:
            rss = parse_rss_mib(Path(f"/proc/{self.pid}/status").read_text(encoding="utf-8"))
        except OSError as exc:
            return LocalSample(t=t, error=f"rss: {exc}")
        gpu = self._gpu
        if gpu is None:
            return LocalSample(t=t, rss_mib=rss)
        return LocalSample(
            t=t, rss_mib=rss,
            vram_mib=_nvml_or_none(lambda: _process_vram_mib(gpu, self.pid)),
            gpu_power_w=_nvml_or_none(lambda: pynvml.nvmlDeviceGetPowerUsage(gpu) / _MW_PER_W),
            gpu_util_pct=_nvml_or_none(lambda: float(pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu)),
            gpu_temp_c=_nvml_or_none(lambda: float(pynvml.nvmlDeviceGetTemperature(gpu, pynvml.NVML_TEMPERATURE_GPU))),
        )


def _nvml_or_none(read: Callable[[], float | None]) -> float | None:
    """Returns ``read()``, or None if NVML does not support the query on this GPU."""
    try:
        return read()
    except pynvml.NVMLError:
        return None


def _process_vram_mib(gpu: _NvmlDevice, pid: int) -> float | None:
    """Returns the GPU memory used by one process, or None if the process has no GPU context."""
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu):
        if proc.pid == pid and proc.usedGpuMemory is not None:
            return proc.usedGpuMemory / _BYTES_PER_MIB
    return None


def _adb(adb: str, cmd: str) -> str:
    """Runs one ``adb shell`` command and returns its stdout."""
    return subprocess.run([adb, "shell", cmd], capture_output=True, text=True, timeout=_ADB_TIMEOUT_S, check=False).stdout


class PhoneThermalSampler(_Sampler[PhoneThermalSample]):
    """Polls the phone's SoC temperatures. ``dumpsys`` costs some phone CPU time, so the interval is long.

    :param adb: Path to ``adb``.
    :param out_path: JSONL output file.
    :param interval_s: Time between samples.
    """

    def __init__(self, adb: str, out_path: Path, interval_s: float = _THERMAL_INTERVAL_S) -> None:
        super().__init__(out_path, interval_s)
        self.adb = adb

    def sample(self) -> PhoneThermalSample:
        t = time.time()
        try:
            out = _adb(self.adb, "dumpsys thermalservice")
        except subprocess.SubprocessError as exc:
            return PhoneThermalSample(t=t, error=str(exc))
        temps = parse_hal_temperatures(out)
        return PhoneThermalSample(
            t=t, thermal_status=parse_thermal_status(out),
            temp_big=temps.get("BIG"), temp_mid=temps.get("MID"), temp_little=temps.get("LITTLE"), temp_skin=temps.get("VIRTUAL-SKIN"),
        )


class PerfettoSession:
    """Records a Perfetto trace on the phone while the context is open, then pulls it to ``out_path``.

    :param adb: Path to ``adb``.
    :param out_path: Local trace file (``.pftrace``).
    :param config: Perfetto text config.
    """

    def __init__(self, adb: str, out_path: Path, config: str = PERFETTO_CONFIG) -> None:
        self.adb = adb
        self.out_path = out_path
        self.config = config
        self.remote = f"{REMOTE_TRACE_DIR}/{out_path.name}"
        self._pid = ""

    def __enter__(self) -> "PerfettoSession":
        # "--background" detaches and prints the pid of the tracing process.
        res = subprocess.run([self.adb, "shell", f"perfetto --txt -c - --background -o {self.remote}"], input=self.config, capture_output=True, text=True, timeout=_ADB_TIMEOUT_S, check=True)
        self._pid = res.stdout.strip().split()[-1] if res.stdout.strip() else ""
        if not self._pid.isdigit():
            raise RuntimeError(f"perfetto did not start: {res.stdout!r} {res.stderr!r}")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        time.sleep(_PERFETTO_TAIL_S)

        # SIGTERM makes perfetto flush and finalize the file, which can take a moment after the process exits.
        _adb(self.adb, f"kill -TERM {self._pid}")
        deadline = time.monotonic() + _PERFETTO_STOP_TIMEOUT_S
        last_size = -1
        while time.monotonic() < deadline:
            alive = _adb(self.adb, f"kill -0 {self._pid} 2>/dev/null && echo alive").strip() == "alive"
            size_out = _adb(self.adb, f"stat -c %s {self.remote} 2>/dev/null").strip()
            size = int(size_out) if size_out.isdigit() else 0
            if not alive and size > 0 and size == last_size:
                break
            last_size = size
            time.sleep(_PERFETTO_STOP_POLL_S)
        subprocess.run([self.adb, "pull", self.remote, str(self.out_path)], capture_output=True, timeout=_ADB_TIMEOUT_S * 3, check=False)
        _adb(self.adb, f"rm -f {self.remote}")


class PhoneTelemetry:
    """Perfetto trace plus temperature polling for one run on the phone.

    :param adb: Path to ``adb``.
    :param trace_path: Local trace file.
    :param thermal_path: Local JSONL file for the temperature samples.
    """

    def __init__(self, adb: str, trace_path: Path, thermal_path: Path) -> None:
        self._parts: list[contextlib.AbstractContextManager[object]] = [PerfettoSession(adb, trace_path), PhoneThermalSampler(adb, thermal_path)]
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> "PhoneTelemetry":
        for part in self._parts:
            self._stack.enter_context(part)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self._stack.close()


def phone_clock_offset_s(adb: str) -> float:
    """Returns phone wall clock minus laptop wall clock, in seconds.

    The phone clock is read in the middle of an ``adb shell`` round trip, so the error is about half the round trip (around 15 ms).
    """
    t0 = time.time()
    out = _adb(adb, "date +%s.%N").strip()
    t1 = time.time()
    return float(out) - (t0 + t1) / 2
