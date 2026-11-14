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
- Each tier-0 leaf and each anchor holds one parked ID. A leaf's ID tags all cells of its path from position 0, so an attach from any leaf gets the whole prefix. Cells shared by several paths carry several IDs.
- When a request extends a leaf, the old leaf becomes an inner node and its ID is freed with `seq_rm(old_id)`. The new leaf's ID still tags those cells, so no cells are lost.
- Running out of IDs triggers tier-0 eviction, the same as running out of cells. Cells usually run out first, because parked prefixes take cells that slots need for decoding.
- Requires `--kv-unified`. Within one stream, `seq_cp` only adds a tag to each cell and `seq_rm` only removes one; a cell is freed when no ID tags it.

### 2. Request lifecycle

| Step | Action |
| --- | --- |
| Start | `match`; for the tier-0 part of the path, `seq_cp(leaf_id, slot_id, 0, p_match)`; for a tier-1 part longer than \(T_1\), Phase 2 `restore`; prefill the rest |
| Run | Unchanged; new cells are tagged with the slot's ID |
| End | `insert` the request's tokens; `seq_cp(slot_id, new_leaf_id, 0, p_end)` gives the new leaf the whole path; free the ID of a leaf that became an inner node; `seq_rm(slot_id)` |

The slot's `prompt.tokens` is cleared after the hand-off, and `get_available_slot` stops comparing slot prompts.

Tier 1 is write-back: a request saves nothing to RAM. A prefix is saved to tier 1 only when its tier-0 leaf is evicted (work item 4). Phase 2 saves on every request (77 ms per turn on the laptop, in the path before the next request), and most of these saves are never used.

### 3. Read-only tree cells

Cells tagged by a parked ID are shared, so no request may change them.
- Context shift and `--cache-reuse` move or delete cells by position. With the tree on, they either stay off or copy first: `seq_rm(slot)` past the shift point, then recompute (copy-on-write by recompute).
- `seq_rm` on a slot only removes the slot's tag; cells stay if a parked ID still tags them. This is the existing behavior of `llama_kv_cells`, and a test checks it.

### 4. Tier-0 eviction

Eviction runs when `llama_decode` fails because the pool is full. `llama_decode` checks that all micro-batches fit before it computes, so a failed call changes nothing and costs one cell scan. On that failure, in place of `try_clear_idle_slots()` (purges a whole idle slot, no write-back):
1. Pick a tier-0 leaf with reference count 0 (bottom-up, same ranking as Phase 2).
2. Save its blocks to tier 1 if they are not there, its tier-1 run is at least \(T_1\), and its aged hit count is above 0. A leaf that no request matched after it was created (for example a stage 3 tail with one-off RAG documents) is dropped without a save.
3. `seq_rm(leaf_id)`; the node's location changes to tier 1.
4. Retry the decode. Repeat from step 1 while it fails. A leaf that shares most cells with other paths frees only its own tail, so one eviction may not be enough.

When no leaf can be evicted, the existing `n_batch` halving stays as a last resort.

Measure the evictions per turn and the time a batch waits for them. If these waits show in TTFT or turn time, add an up-front step later: at request start, evict until the request's new cells (prompt tokens not attached, plus `max_tokens`) are free. That step needs a new library call that counts free cells, and the failure path stays as a fallback.

### 5. Relation to #30970

[#30970](https://github.com/ggml-org/llama.cpp/pull/30970) (`--slot-attach`) lets a slot attach another slot's prefix cells. It was still open on 2026-11-10, so this phase stays independent. A comparison with it on config E is optional, because it needs a second build of that branch.

### 6. Tests

- Attach check in `kv-block-bench`: logits after attaching a parked prefix with `seq_cp` and prefilling the rest are bit-identical to a prefill with the same batch shapes.
- `seq_rm` of a slot keeps the cells that a parked ID still tags (work item 3).
- Index tests in `tests/test-server-component.cpp` for leaf IDs: an ID is freed when its leaf grows, and eviction picks only leaves with reference count 0.

## Measurements

Laptop, Llama, 3 reps: A, C, E, F with `--cache-tree --cache-tree-parked 32`, against Phase 0 and Phase 2. Also run A and F with `-c 8192` so the pool fills, to exercise tier-0 eviction (Phase 0 never filled it). One stage 3 prompt (about 4,300 tokens with the `desktop` preset) still fits.

Phone, Llama, 3 reps: A and F only, as a check that nothing gets worse. On the phone the KV pool is already in RAM, so Phase 3 removes only about 27 ms of saves and 12 ms of restores per turn, against a 20 s turn.

## Exit criteria

- E: background hit rate within 2 points of the ideal (40-52%).
- A with the tree: stage 1 TTFT on the laptop within 20% of C.
- Small `-c`: no request fails, and evicted leaves are restored from tier 1 when worth it.
- The attach check is bit-identical. Exact greedy equality with C is not a valid test on a GPU (Phase 2 results). Against a run without prompt cache, outputs differ no more often than C's, by a paired sign test over 102 requests.

## Risks

- **Fragmentation.** Shared cells stay at fixed indices while slots come and go. Watch `find_slot` failures with free cells left.
- **Parked ID memory for hybrid models.** Recurrent memory is sized by `n_seq_max`, so each parked ID reserves a state. Measure before enabling for LFM2 in Phase 4.
