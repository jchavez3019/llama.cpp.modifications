#!/usr/bin/env python3
"""Fit restore and prefill costs from ``llama-kv-block-bench`` output and derive restore thresholds.

The cost model is the one in ``docs/radix_tree_kv_cache/design.md`` ("Restore or recompute"):

- recompute of ``k`` tokens costs ``k * t_prefill``, the slope of prefill time over ``n``;
- restore of ``k`` tokens in ``calls`` state calls costs ``calls * t_overhead + k * t_copy``;
- restore wins when ``k > T = t_overhead / (t_prefill - t_copy)``.

Usage::

    python kv_block_fit.py ../../../../bench-results/phase1/*/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Run lengths (tokens) used to pick a block size: stage tails and history chunks from the Phase 0 workloads.
DEFAULT_RUN_LENGTHS: tuple[int, ...] = (80, 300, 600, 3000)


@dataclass(frozen=True)
class TimedRow:
    """One timed measurement from the benchmark.

    :param device: Device label passed with ``--device``.
    :param model: Model description from ``llama_model_desc``.
    :param kind: Measurement kind, for example ``prefill_offset`` or ``block_restore``.
    :param n: Number of tokens.
    :param b: Block size, or 0 when the kind has no blocks.
    :param ms: Wall time in milliseconds.
    :param calls: Number of state API calls (blocks), 1 for whole-state kinds.
    """

    device: str
    model: str
    kind: str
    n: int
    b: int
    ms: float
    calls: int


@dataclass(frozen=True)
class CheckRow:
    """One correctness check: probe logits after a restore against a cold prefill.

    :param device: Device label.
    :param model: Model description.
    :param kind: ``check_whole`` or ``check_block``.
    :param n: Number of restored tokens.
    :param b: Block size, or 0.
    :param max_abs_diff: Largest absolute logit difference.
    :param argmax_match: Whether the top token is the same.
    """

    device: str
    model: str
    kind: str
    n: int
    b: int
    max_abs_diff: float
    argmax_match: bool


@dataclass(frozen=True)
class CostFit:
    """Fitted costs for one device and model.

    :param prefill_ms_per_token: Marginal prefill cost (slope of ``prefill_offset``).
    :param prefill_fixed_ms: Intercept of the prefill fit (fixed cost per decode call).
    :param restore_overhead_ms: Fixed cost per restore call.
    :param restore_ms_per_token: Copy cost per restored token.
    :param threshold_tokens: Restore threshold ``T``; ``math.inf`` when copying is not cheaper than prefill.
    """

    prefill_ms_per_token: float
    prefill_fixed_ms: float
    restore_overhead_ms: float
    restore_ms_per_token: float
    threshold_tokens: float


def load_rows(paths: list[Path]) -> tuple[list[TimedRow], list[CheckRow]]:
    """Read benchmark JSON lines from several files.

    :param paths: Files written by ``llama-kv-block-bench -o``.
    :returns: Timed rows and check rows.
    """
    timed: list[TimedRow] = []
    checks: list[CheckRow] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)

            # Check rows carry a logit comparison instead of a time.
            if d["kind"].startswith("check_"):
                checks.append(CheckRow(d["device"], d["model"], d["kind"], int(d["n"]), int(d["b"]), float(d["max_abs_diff"]), bool(d["argmax_match"])))
            else:
                timed.append(TimedRow(d["device"], d["model"], d["kind"], int(d["n"]), int(d["b"]), float(d["ms"]), int(d["calls"])))
    return timed, checks


def median_ms(rows: list[TimedRow], kind: str) -> dict[tuple[int, int], tuple[float, int]]:
    """Median time per ``(n, b)`` for one kind, which removes outlier repetitions.

    :param rows: Timed rows of one device and model.
    :param kind: Measurement kind to select.
    :returns: ``(n, b) -> (median ms, calls)``.
    """
    groups: dict[tuple[int, int], list[TimedRow]] = defaultdict(list)
    for r in rows:
        if r.kind == kind:
            groups[(r.n, r.b)].append(r)
    return {key: (statistics.median(r.ms for r in g), g[0].calls) for key, g in groups.items()}


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares line ``y = intercept + slope * x``.

    :param xs: Inputs; needs at least two distinct values.
    :param ys: Outputs.
    :returns: ``(intercept, slope)``.
    :raises ValueError: If fewer than two distinct inputs are given.
    """
    if len(set(xs)) < 2:
        raise ValueError("fit_line needs at least two distinct x values")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def fit_restore(points: list[tuple[int, int, float]]) -> tuple[float, float]:
    """Least-squares fit of ``ms = overhead * calls + per_token * tokens``, with no intercept.

    Whole-state restores (one call) and block restores (many calls) of the same token count separate the two terms.

    :param points: ``(calls, tokens, ms)`` triples.
    :returns: ``(overhead_ms, ms_per_token)``.
    :raises ValueError: If the points do not determine both terms.
    """
    # Normal equations of the 2-variable fit without intercept.
    s_cc = sum(c * c for c, _, _ in points)
    s_ct = sum(c * t for c, t, _ in points)
    s_tt = sum(t * t for _, t, _ in points)
    s_cy = sum(c * y for c, _, y in points)
    s_ty = sum(t * y for _, t, y in points)

    det = s_cc * s_tt - s_ct * s_ct
    if det == 0:
        raise ValueError("fit_restore needs points with different calls-to-tokens ratios")
    overhead = (s_cy * s_tt - s_ty * s_ct) / det
    per_token = (s_cc * s_ty - s_ct * s_cy) / det
    return overhead, per_token


def threshold(overhead_ms: float, prefill_ms_per_token: float, copy_ms_per_token: float) -> float:
    """Restore threshold ``T = overhead / (prefill - copy)``.

    :param overhead_ms: Fixed cost per restore call.
    :param prefill_ms_per_token: Marginal prefill cost.
    :param copy_ms_per_token: Copy cost per token.
    :returns: Tokens above which a restore beats recompute; ``math.inf`` if copying is not cheaper than prefill.
    """
    gain = prefill_ms_per_token - copy_ms_per_token
    if gain <= 0:
        return math.inf
    return max(overhead_ms, 0.0) / gain


def block_restore_cost(run: int, b: int, fit: CostFit) -> float:
    """Expected cost of making a run of ``run`` tokens available from ``b``-token blocks.

    A run starts and ends at arbitrary positions, while blocks sit at multiples of ``b``. On average the first block repeats ``(b - 1) / 2`` parent tokens (copied, then trimmed), and the last ``(b - 1) / 2`` tokens are recomputed instead of loading a partial block. Each block is one call with the current API.

    :param run: Run length in tokens.
    :param b: Block size in tokens.
    :param fit: Fitted costs of the device and model.
    :returns: Expected time in milliseconds.
    """
    waste = (b - 1) / 2
    calls = math.ceil(run / b)
    return calls * fit.restore_overhead_ms + (run + waste) * fit.restore_ms_per_token + waste * fit.prefill_ms_per_token


def best_block(fit: CostFit, sizes: list[int], runs: tuple[int, ...] = DEFAULT_RUN_LENGTHS) -> int:
    """Pick the block size with the lowest mean restore cost over typical run lengths.

    :param fit: Fitted costs.
    :param sizes: Candidate block sizes.
    :param runs: Run lengths to average over.
    :returns: The best block size.
    """
    return min(sizes, key=lambda b: statistics.fmean(block_restore_cost(r, b, fit) for r in runs))


def fit_costs(rows: list[TimedRow]) -> CostFit:
    """Fit prefill and restore costs for one device and model.

    :param rows: Timed rows of one device and model.
    :returns: The fitted costs and threshold.
    """
    # Prefill: the slope over n is the marginal cost of tokens that join a batch.
    prefill = median_ms(rows, "prefill_offset")
    ns = sorted(n for n, _ in prefill)
    fixed, per_token = fit_line([float(n) for n in ns], [prefill[(n, 0)][0] for n in ns])

    # Restore: whole-state restores are one call; block restores use one call per block.
    points = [(calls, n, ms) for (n, _), (ms, calls) in median_ms(rows, "whole_restore").items()]
    points += [(calls, n, ms) for (n, _), (ms, calls) in median_ms(rows, "block_restore").items()]
    if any(c > 1 for c, _, _ in points):
        overhead, copy = fit_restore(points)
    else:
        # Without block rows the two terms cannot be separated; use the line through whole restores.
        overhead, copy = fit_line([float(n) for _, n, _ in points], [ms for _, _, ms in points])

    return CostFit(per_token, fixed, overhead, copy, threshold(overhead, per_token, copy))


def summarize(timed: list[TimedRow], checks: list[CheckRow]) -> str:
    """Build a markdown report: one row per device and model.

    :param timed: All timed rows.
    :param checks: All check rows.
    :returns: Markdown text.
    """
    groups: dict[tuple[str, str], list[TimedRow]] = defaultdict(list)
    for r in timed:
        groups[(r.device, r.model)].append(r)

    lines = [
        "| Device | Model | Prefill ms/token | Restore overhead ms/call | Restore ms/token | T (tokens) | Best B | Checks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (device, model), rows in sorted(groups.items()):
        fit = fit_costs(rows)
        sizes = sorted({r.b for r in rows if r.kind == "block_restore"})
        best = str(best_block(fit, sizes)) if sizes else "-"

        # A check fails if the top token differs; the logit difference is reported for context.
        mine = [c for c in checks if c.device == device and c.model == model]
        failed = [c for c in mine if not c.argmax_match]
        diff = max((c.max_abs_diff for c in mine), default=0.0)
        check = f"{len(mine) - len(failed)}/{len(mine)} pass, max diff {diff:.3g}"

        t = "inf" if math.isinf(fit.threshold_tokens) else f"{fit.threshold_tokens:.1f}"
        lines.append(f"| {device} | {model} | {fit.prefill_ms_per_token:.4f} | {fit.restore_overhead_ms:.4f} | {fit.restore_ms_per_token:.5f} | {t} | {best} | {check} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the markdown summary for the given result files.

    :param argv: Command line arguments without the program name.
    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    timed, checks = load_rows(args.files)
    print(summarize(timed, checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
