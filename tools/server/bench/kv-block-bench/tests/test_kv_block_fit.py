"""Offline tests for the cost fits in ``kv_block_fit.py``.

The fixtures are built by hand in the JSON-lines shape that ``kv-block-bench.cpp`` writes (``emit_row`` and ``emit_check``). If the C++ output format changes, these fixtures must change with it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kv_block_fit import CostFit, TimedRow, best_block, fit_costs, fit_restore, load_rows, summarize, threshold

OVERHEAD_MS = 0.4
COPY_MS = 0.03
PREFILL_MS = 0.15
PREFILL_FIXED_MS = 5.0


def _rows(n_list: list[int], b_list: list[int]) -> list[TimedRow]:
    """Exact rows from known costs: prefill = fixed + p * n, restore = overhead * calls + copy * n."""
    rows: list[TimedRow] = []
    for n in n_list:
        rows.append(TimedRow("dev", "m", "prefill_offset", n, 0, PREFILL_FIXED_MS + PREFILL_MS * n, 1))
        rows.append(TimedRow("dev", "m", "whole_restore", n, 0, OVERHEAD_MS + COPY_MS * n, 1))
        for b in b_list:
            if n % b == 0:
                rows.append(TimedRow("dev", "m", "block_restore", n, b, OVERHEAD_MS * (n // b) + COPY_MS * n, n // b))
    return rows


def test_fit_costs_recovers_overhead_and_copy_cost_from_whole_and_block_restores() -> None:
    """
    Given: exact restore times built from a known overhead per call and copy cost per token, for whole and block restores
    When: fit_costs separates the two terms
    Then: both costs, the prefill slope, and the threshold match the values used to build the rows
    """
    fit = fit_costs(_rows([16, 64, 256, 1024], [16, 64]))

    assert fit.restore_overhead_ms == pytest.approx(OVERHEAD_MS)
    assert fit.restore_ms_per_token == pytest.approx(COPY_MS)
    assert fit.prefill_ms_per_token == pytest.approx(PREFILL_MS)
    assert fit.prefill_fixed_ms == pytest.approx(PREFILL_FIXED_MS)
    assert fit.threshold_tokens == pytest.approx(OVERHEAD_MS / (PREFILL_MS - COPY_MS))


def test_fit_restore_rejects_points_with_one_calls_to_tokens_ratio() -> None:
    """
    Given: restore points that all have the same ratio of calls to tokens
    When: fit_restore tries to separate overhead from copy cost
    Then: it raises ValueError instead of returning arbitrary numbers
    """
    with pytest.raises(ValueError):
        fit_restore([(1, 16, 1.0), (2, 32, 2.0)])


def test_threshold_is_infinite_when_copy_is_not_cheaper_than_prefill() -> None:
    """
    Given: a copy cost per token that is equal to or higher than the prefill cost
    When: the threshold is computed
    Then: it is infinite, so restores are never chosen
    """
    assert math.isinf(threshold(1.0, 0.1, 0.1))
    assert math.isinf(threshold(1.0, 0.1, 0.2))


def test_best_block_prefers_larger_blocks_when_call_overhead_dominates() -> None:
    """
    Given: one fit with a large overhead per call and one with almost none, both with cheap copies
    When: best_block picks from 16, 64, and 256 tokens
    Then: the high-overhead fit picks a larger block than the low-overhead fit, which avoids boundary recompute
    """
    sizes = [16, 64, 256]
    costly_calls = CostFit(0.15, 5.0, 2.0, 0.003, 0.0)
    cheap_calls = CostFit(0.15, 5.0, 0.001, 0.003, 0.0)

    assert best_block(costly_calls, sizes) > best_block(cheap_calls, sizes)
    assert best_block(cheap_calls, sizes) == 16


def test_summarize_reports_failed_checks_from_benchmark_output(tmp_path: Path) -> None:
    """
    Given: a results file in the benchmark's JSON-lines format with one passing and one failing block check
    When: the file is loaded and summarized
    Then: the report row says 1 of 2 checks pass and shows the largest logit difference
    """
    lines = [
        {"device": "dev", "model": "m", "kind": r.kind, "n": r.n, "b": r.b, "rep": 0, "ms": r.ms, "bytes": 1, "calls": r.calls}
        for r in _rows([64, 256], [16])
    ]
    lines.append({"device": "dev", "model": "m", "kind": "check_block", "n": 64, "b": 16, "max_abs_diff": 0.0, "argmax_match": True})
    lines.append({"device": "dev", "model": "m", "kind": "check_block", "n": 256, "b": 16, "max_abs_diff": 2.5, "argmax_match": False})
    path = tmp_path / "out.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in lines) + "\n")

    timed, checks = load_rows([path])
    report = summarize(timed, checks)

    assert len(checks) == 2
    assert "1/2 pass, max diff 2.5" in report
