"""Runs the Phase 0 baseline matrix: configurations x models x repetitions.

For every (model, repetition, configuration) the script starts a fresh ``llama-server`` (locally or on an Android phone over ``adb``), runs the workload, records telemetry, stops the server, and stores the server log. Results go to ``<out>/<device>/<model>/<config>/rep<k>.*``.

Configurations (all use ``--kv-unified``, ``-fa on``, ``-lv 4``):
    A  -np 2, all stages pinned to slot 0 (one lane, the stages take turns)
    B  -np 2, no pinning (the server picks the slot)
    C  -np 4 --cache-ram 0, one pinned slot per stage (best case without a RAM cache)
    D  as A with a small --cache-ram (the eviction cliff)
    E  as A plus a concurrent background task on slot 1 every 2 turns
    F  as A with --cache-ram 0 (one slot, no RAM cache: each stage overwrites the cells of the one before)
    A-tree, D-tree  as A and D with --cache-tree (Phase 2)
    F-tree  as F with --cache-tree and the RAM cache size of A

Examples:
    python run_matrix.py --target local --server-extra "-ngl 99" --models ~/models/*.gguf --device laptop
    python run_matrix.py --target adb --adb ~/Android/Sdk/platform-tools/adb --models ~/models/*.gguf --device pixel-fold \\
        --preset edge --ctx 4096 --turns 12 --warmup 2 --taskset f0 --server-extra "-t 4" --cache-ram-default 1024 --d-cache-ram 32
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Protocol

import requests

from orchestration_bench import add_run_args, run_rep
from runner import NO_SLOT, PinMode, RunConfig, ServerClient, output_path
from telemetry import LocalSampler, PhoneTelemetry, parse_hal_temperatures, phone_clock_offset_s
from workload import Layout

_READY_TIMEOUT_S = 900
_STOP_TIMEOUT_S = 30
_POLL_S = 0.5
_START_POLLS = 20
_COOL_POLL_S = 15
_KIB_PER_MIB = 1024
_BG_EVERY_E = 2
_BG_SLOT_E = 1
_REMOTE_LOG = "server.log"
_SERVER_BINARY = "llama-server"

# Failures of one run that should not stop the remaining runs of an unattended matrix.
_RUN_ERRORS: tuple[type[BaseException], ...] = (RuntimeError, TimeoutError, OSError, subprocess.SubprocessError, requests.RequestException)


@dataclass(frozen=True)
class Config:
    """One matrix configuration: server arguments plus client settings.

    :param n_parallel: Number of server slots (``-np``).
    :param cache_ram_mib: ``--cache-ram`` value, or None for the server default (8192 MiB).
    :param run: Client settings. ``n_turns`` and ``n_warmup`` are replaced by the CLI values.
    :param server_extra: Server arguments of this configuration only.
    """

    n_parallel: int
    cache_ram_mib: int | None
    run: RunConfig
    server_extra: tuple[str, ...] = ()


def configs(d_cache_ram_mib: int, default_cache_ram_mib: int | None) -> dict[str, Config]:
    """Returns the matrix configurations by name.

    :param d_cache_ram_mib: Small RAM cache of config D.
    :param default_cache_ram_mib: RAM cache of A, B, and E (C and F use 0), or None for the server default. A phone needs a lower value than the default 8192 MiB.
    :return: Configuration name to configuration.
    """
    base = RunConfig(n_turns=0, n_warmup=0, pin=PinMode.SLOT0, bg_every=0, bg_slot=NO_SLOT, bg_concurrent=False)
    return {
        "A": Config(2, default_cache_ram_mib, base),
        "B": Config(2, default_cache_ram_mib, replace(base, pin=PinMode.NONE)),
        "C": Config(4, 0, replace(base, pin=PinMode.PER_STAGE)),
        "D": Config(2, d_cache_ram_mib, base),
        "E": Config(2, default_cache_ram_mib, replace(base, bg_every=_BG_EVERY_E, bg_slot=_BG_SLOT_E, bg_concurrent=True)),
        "F": Config(2, 0, base),
        "A-tree": Config(2, default_cache_ram_mib, base, ("--cache-tree",)),
        "D-tree": Config(2, d_cache_ram_mib, base, ("--cache-tree",)),
        "F-tree": Config(2, default_cache_ram_mib, base, ("--cache-tree",)),
    }


def server_args(model: str, cfg: Config, ctx: int, port: int, extra: list[str]) -> list[str]:
    """Returns the ``llama-server`` arguments of one configuration."""
    args = ["-m", model, "-c", str(ctx), "-np", str(cfg.n_parallel), "--kv-unified", "-fa", "on", "-lv", "4", "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
    if cfg.cache_ram_mib is not None:
        args += ["--cache-ram", str(cfg.cache_ram_mib)]
    return args + list(cfg.server_extra) + extra


def run_order(models: list[str], names: list[str], reps: int) -> list[tuple[str, int, str]]:
    """Returns the (model, repetition, configuration) runs in execution order.

    Each repetition runs every configuration once, and the start configuration rotates with the repetition. A device that heats up over time then slows all configurations about equally, instead of always the last ones.

    :param models: Model paths, run one after another.
    :param names: Configuration names.
    :param reps: Repetitions per configuration.
    :return: Runs in order.
    """
    out: list[tuple[str, int, str]] = []
    for model in models:
        for rep in range(reps):
            shift = rep % len(names)
            out.extend((model, rep, name) for name in names[shift:] + names[:shift])
    return out


class Server(Protocol):
    """A started ``llama-server`` process with its telemetry."""

    def telemetry(self, prefix: Path) -> AbstractContextManager[object]:
        """Returns the telemetry recorder of the run with output prefix ``prefix``."""
        ...

    def run_meta(self) -> dict[str, object]:
        """Returns device facts to store in the run metadata."""
        ...

    def is_running(self) -> bool:
        """Returns False once the server process has exited."""
        ...

    def stop(self, log_path: Path) -> None:
        """Stops the server and makes sure its log is at ``log_path``."""
        ...


class LocalServer:
    """``llama-server`` started as a local child process. The log is written directly to ``log_path``."""

    def __init__(self, binary: str, args: list[str], log_path: Path) -> None:
        self._log: IO[bytes] = log_path.open("wb")
        self._proc = subprocess.Popen([binary, *args], stdout=self._log, stderr=subprocess.STDOUT)

    def telemetry(self, prefix: Path) -> LocalSampler:
        return LocalSampler(self._proc.pid, output_path(prefix, ".samples.jsonl"))

    def run_meta(self) -> dict[str, object]:
        return {}

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def stop(self, log_path: Path) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._log.close()


class AdbServer:
    """``llama-server`` started on an Android phone in ``remote_dir``. The log is pulled when the server stops.

    ``adb shell`` does not return while the remote command keeps the session open, even when backgrounded, so the shell runs as a local child process that is not waited on.

    :param adb: Path to ``adb``.
    :param remote_dir: Directory on the phone with ``llama-server`` and ``models/``.
    :param args: Server arguments.
    :param port: Port to forward from the laptop to the phone.
    :param taskset: CPU mask for ``taskset`` (for example ``f0`` = cores 4-7), or an empty string.
    """

    def __init__(self, adb: str, remote_dir: str, args: list[str], port: int, taskset: str) -> None:
        self.adb = adb
        self.remote_dir = remote_dir

        # A server left over from an interrupted run would hold the port and the memory.
        self._shell(f"pkill -x {_SERVER_BINARY}")
        subprocess.run([adb, "forward", f"tcp:{port}", f"tcp:{port}"], check=True, capture_output=True)
        self.mem_available_mib = phone_mem_available_mib(adb)
        self.clock_offset_s = phone_clock_offset_s(adb)

        prefix = f"taskset {taskset} " if taskset else ""
        cmd = f"cd {remote_dir} && exec {prefix}./{_SERVER_BINARY} {shlex.join(args)} > {_REMOTE_LOG} 2>&1 < /dev/null"
        self._shell_proc = subprocess.Popen([adb, "shell", cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Look up the pid, so the server can be stopped by pid.
        self.pid = 0
        for _ in range(_START_POLLS):
            out = self._shell(f"pidof {_SERVER_BINARY}").split()
            if out:
                self.pid = int(out[0])
                break
            time.sleep(_POLL_S)
        if not self.pid:
            self._shell_proc.kill()
            raise RuntimeError(f"{_SERVER_BINARY} did not start on the phone, see {remote_dir}/{_REMOTE_LOG}")

    def _shell(self, cmd: str) -> str:
        return subprocess.run([self.adb, "shell", cmd], capture_output=True, text=True, check=False).stdout

    def telemetry(self, prefix: Path) -> PhoneTelemetry:
        return PhoneTelemetry(self.adb, output_path(prefix, ".pftrace"), output_path(prefix, ".thermal.jsonl"))

    def run_meta(self) -> dict[str, object]:
        return {"phone_mem_available_mib": self.mem_available_mib, "phone_clock_offset_s": self.clock_offset_s, "server_pid": self.pid}

    def is_running(self) -> bool:
        return str(self.pid) in self._shell(f"pidof {_SERVER_BINARY}").split()

    def stop(self, log_path: Path) -> None:
        self._shell(f"kill {self.pid}")
        for _ in range(int(_STOP_TIMEOUT_S / _POLL_S)):
            if not self._shell(f"pidof {_SERVER_BINARY}").strip():
                break
            time.sleep(_POLL_S)
        self._shell_proc.kill()
        self._shell_proc.wait()
        subprocess.run([self.adb, "pull", f"{self.remote_dir}/{_REMOTE_LOG}", str(log_path)], capture_output=True, check=False)


def phone_mem_available_mib(adb: str) -> float | None:
    """Returns ``MemAvailable`` of the phone in MiB, or None if it cannot be read."""
    out = subprocess.run([adb, "shell", "grep MemAvailable /proc/meminfo"], capture_output=True, text=True, check=False).stdout.split()
    return int(out[1]) / _KIB_PER_MIB if len(out) >= 2 and out[1].isdigit() else None


def wait_phone_cool(adb: str, max_big_c: float) -> None:
    """Blocks until the phone's big cluster is below ``max_big_c``."""
    while True:
        out = subprocess.run([adb, "shell", "dumpsys thermalservice"], capture_output=True, text=True, check=False).stdout
        big = parse_hal_temperatures(out).get("BIG", 0.0)
        if big < max_big_c:
            return
        print(f"  waiting for the phone to cool: BIG = {big:.1f} C (limit {max_big_c:.1f} C)", flush=True)
        time.sleep(_COOL_POLL_S)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 0 baseline matrix")
    parser.add_argument("--target", choices=["local", "adb"], default="local")
    parser.add_argument("--models", nargs="+", required=True, help="local GGUF paths; with --target adb the phone must have the same file names in <remote-dir>/models")
    parser.add_argument("--configs", default="A,B,C,D,E,F")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--d-cache-ram", type=int, default=128, help="--cache-ram (MiB) of config D")
    parser.add_argument("--cache-ram-default", type=int, default=None, help="--cache-ram (MiB) of configs A, B, E (default: server default)")
    parser.add_argument("--device", required=True, help="device name, used as the first directory level of the results")
    parser.add_argument("--out", type=Path, default=Path("bench-results/phase0"))
    parser.add_argument("--server-extra", default="", help="extra server arguments, for example '-ngl 99' or '-t 4'")
    parser.add_argument("--server-bin", default="build-laptop-cuda-release/bin/llama-server", help="local server binary (--target local)")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--remote-dir", default="/data/local/tmp/llama.cpp")
    parser.add_argument("--taskset", default="", help="CPU mask on the phone, for example f0")
    parser.add_argument("--max-big-temp", type=float, default=None, help="phone: wait until the big cores are below this temperature before each run (default: no wait)")
    add_run_args(parser)
    args = parser.parse_args()

    matrix = configs(args.d_cache_ram, args.cache_ram_default)
    extra = shlex.split(args.server_extra)
    for model_path, rep, name in run_order(args.models, args.configs.split(","), args.reps):
        model_name = Path(model_path).stem
        model_arg = model_path if args.target == "local" else f"models/{Path(model_path).name}"
        cfg = matrix[name]
        run_cfg = replace(cfg.run, n_turns=args.turns, n_warmup=args.warmup)
        prefix = args.out / args.device / model_name / name / f"rep{rep}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        sargs = server_args(model_arg, cfg, args.ctx, args.port, extra)
        print(f"[{args.device}] {model_name} config {name} rep {rep}: {_SERVER_BINARY} {shlex.join(sargs)}", flush=True)

        # Start a fresh server, so no cache state leaks between runs. A failed run is logged and skipped.
        try:
            if args.target == "adb":
                if args.max_big_temp is not None:
                    wait_phone_cool(args.adb, args.max_big_temp)
                server: Server = AdbServer(args.adb, args.remote_dir, sargs, args.port, args.taskset)
            else:
                server = LocalServer(args.server_bin, sargs, output_path(prefix, ".server.log"))
            try:
                client = ServerClient(f"http://127.0.0.1:{args.port}")
                client.wait_ready(_READY_TIMEOUT_S, server.is_running)
                meta: dict[str, object] = {"device": args.device, "model": model_name, "config": name, "rep": rep, "server_args": sargs, "target": args.target, "taskset": args.taskset, **server.run_meta()}
                metrics = run_rep(client, args.preset, Layout(args.layout), args.seed, run_cfg, prefix, server.telemetry(prefix), meta)
            finally:
                server.stop(output_path(prefix, ".server.log"))
        except _RUN_ERRORS:
            output_path(prefix, ".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"  FAILED, see {output_path(prefix, '.error.txt')}", flush=True)
            continue
        hits = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items() if k.endswith(".hit"))
        print(f"  turn p50 = {metrics['turn_p50']:.0f} ms, errors = {metrics['n_errors']:.0f}, {hits}", flush=True)


if __name__ == "__main__":
    main()
