"""Runs the orchestration workload once against a running ``llama-server``.

Example:
    python orchestration_bench.py --url http://127.0.0.1:8080 --preset desktop --pin slot0 --out bench-results/manual/rep0

This writes ``<out>.requests.jsonl`` and ``<out>.meta.json`` (and ``<out>.samples.jsonl`` with ``--pid``). ``run_matrix.py`` uses ``run_rep`` to start servers and loop over configurations.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path

from runner import NO_SLOT, PinMode, RunConfig, ServerClient, output_path, run, write_meta
from summary import Metrics, summarize_requests
from telemetry import LocalSampler
from workload import PRESETS, Layout, Workload

_READY_TIMEOUT_S = 600
_DEFAULT_SEED = 1234
_DEFAULT_TURNS = 23
_DEFAULT_WARMUP = 3


def run_rep(client: ServerClient, preset: str, layout: Layout, seed: int, cfg: RunConfig, out_prefix: Path, telemetry: AbstractContextManager[object] | None, meta: dict[str, object]) -> Metrics:
    """Runs one repetition and writes ``<out_prefix>.requests.jsonl`` and ``<out_prefix>.meta.json``.

    :param client: Client of a ready server.
    :param preset: Workload preset name.
    :param layout: History layout.
    :param seed: Workload seed.
    :param cfg: Client settings.
    :param out_prefix: Output path without the suffix, for example ``.../A/rep0``.
    :param telemetry: Optional telemetry recorder that runs while the requests run (it also covers prompt calibration).
    :param meta: Extra metadata to store (configuration, model, server arguments, device facts).
    :return: Request metrics of the repetition (see ``summary.summarize_requests``).
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    workload = Workload(PRESETS[preset], seed, layout, client.count_tokens)
    write_meta(output_path(out_prefix, ".meta.json"), {**meta, "preset": preset, "layout": str(layout), "seed": seed, "run": asdict(cfg), "t_start": time.time()})

    with (telemetry or contextlib.nullcontext()), output_path(out_prefix, ".requests.jsonl").open("w", encoding="utf-8") as out:
        records = run(client, workload, cfg, out)
    return summarize_requests(records)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """Adds the workload and client arguments shared by both CLIs."""
    parser.add_argument("--preset", choices=sorted(PRESETS), default="desktop")
    parser.add_argument("--layout", choices=[str(x) for x in Layout], default=str(Layout.INLINE), help="history as text in one message, or as chat turns")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--turns", type=int, default=_DEFAULT_TURNS, help="total turns, including warm-up turns")
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the orchestration workload against a running llama-server")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, required=True, help="output prefix, for example bench-results/manual/rep0")
    parser.add_argument("--pin", choices=[str(x) for x in PinMode], default=str(PinMode.SLOT0))
    parser.add_argument("--bg-every", type=int, default=0, help="run the background task every N turns (0 = off)")
    parser.add_argument("--bg-slot", type=int, default=NO_SLOT)
    parser.add_argument("--bg-concurrent", action="store_true")
    parser.add_argument("--pid", type=int, default=0, help="local server pid, to sample RSS and GPU state")
    add_run_args(parser)
    args = parser.parse_args()

    client = ServerClient(args.url)
    client.wait_ready(_READY_TIMEOUT_S)
    cfg = RunConfig(args.turns, args.warmup, PinMode(args.pin), args.bg_every, args.bg_slot, args.bg_concurrent)
    telemetry = LocalSampler(args.pid, output_path(args.out, ".samples.jsonl")) if args.pid else None
    metrics = run_rep(client, args.preset, Layout(args.layout), args.seed, cfg, args.out, telemetry, {"url": args.url})
    for key, value in metrics.items():
        print(f"{key:24s} {value:10.3f}")


if __name__ == "__main__":
    main()
