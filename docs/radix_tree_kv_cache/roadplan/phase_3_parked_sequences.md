# Phase 3 plan: parked sequences and stateless slots

Phase 3 extends the tree into the KV pool (tier 0). Cached prefixes stay in the pool under parked sequence IDs that no slot uses, any slot attaches them with `seq_cp`, and slots hold nothing between requests. Results go in [phase_3_parked_sequences_results.md](phase_3_parked_sequences_results.md).

## Goal

- Any slot serves any request at the same cost, so slot choice no longer decides hits.
- BUG-001 cannot happen: there is no "empty slot" whose cache is skipped.
- Attaching a prefix that is already in the pool costs a tag change, not a copy (laptop stage 1 TTFT near config C's 19 ms, against A's 104 ms).

## Scope

Full-attention models, as in Phase 2. Hybrid and SWA models follow in Phase 4.

## Work items

### 1. Sequence IDs

- `n_seq_max = n_parallel + 1 + n_parked` (slots, scratch, parked leaves and anchors), within `LLAMA_MAX_SEQ` (256). New flag `--cache-tree-parked N`, default 32.
- Each tier-0 leaf and each anchor holds one parked ID. Running out of IDs triggers tier-0 eviction, the same as running out of cells.
- Requires `--kv-unified`.

### 2. Request lifecycle

| Step | Action |
| --- | --- |
| Start | `match`; for the tier-0 part of the path, `seq_cp(leaf_id, slot_id, 0, p_match)`; for a tier-1 part longer than \(T_1\), Phase 2 `restore`; prefill the rest |
| Run | Unchanged; new cells are tagged with the slot's ID |
| End | `insert` the request's tokens; `seq_cp(slot_id, new_leaf_id, p_attach, p_end)` hands the new cells to the tree; `seq_rm(slot_id)` |

The slot's `prompt.tokens` is cleared after the hand-off, and `get_available_slot` stops comparing slot prompts.

### 3. Read-only tree cells

Cells tagged by a parked ID are shared, so no request may change them.
- Context shift and `--cache-reuse` move or delete cells by position. With the tree on, they either stay off or copy first: `seq_rm(slot)` past the shift point, then recompute (copy-on-write by recompute).
- `seq_rm` on a slot only removes the slot's tag; cells stay if a parked ID still tags them. This is the existing behavior of `llama_kv_cells`, and a test checks it.

### 4. Tier-0 eviction

Replace `try_clear_idle_slots()` (purges a whole idle slot, no write-back) with tree eviction when the pool or the IDs run out:
1. Pick a tier-0 leaf with reference count 0 (bottom-up, same ranking as Phase 2).
2. Save its blocks to tier 1 if they are not there and its tier-1 run is at least \(T_1\).
3. `seq_rm(leaf_id)`; the node's location changes to tier 1.

Decode failure handling (`n_batch` halving) stays as a last resort.

### 5. Relation to #30970

[#30970](https://github.com/ggml-org/llama.cpp/pull/30970) (`--slot-attach`) lets a slot attach another slot's prefix cells. Before starting, read its current state: if it has merged, build attach on its code paths; if not, keep this phase independent and compare the two on config E.

## Measurements

Both devices, Llama, 3 reps: A, C, E, F with `--cache-tree --cache-tree-parked 32`, against Phase 0 and Phase 2. Also run with a small `-c` so the pool fills, to exercise tier-0 eviction (Phase 0 never filled it).

## Exit criteria

- E: background hit rate within 2 points of the ideal (40-52%).
- A with the tree: stage 1 TTFT on the laptop within 20% of C.
- Small `-c`: no decode failures, and evicted leaves are restored from tier 1 when worth it.
- Greedy outputs equal to C.

## Risks

- **Fragmentation.** Shared cells stay at fixed indices while slots come and go. Watch `find_slot` failures with free cells left.
- **Parked ID memory for hybrid models.** Recurrent memory is sized by `n_seq_max`, so each parked ID reserves a state. Measure before enabling for LFM2 in Phase 4.
