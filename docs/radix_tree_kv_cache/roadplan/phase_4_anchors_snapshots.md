# Phase 4 plan: anchors and snapshot policy

Phase 4 makes SWA (Gemma) and hybrid (LFM2) models reuse shared prefixes. Their memory can only resume where a snapshot was taken during prefill, so the server takes snapshots at client-marked anchors and on a fixed grid. Results go in [phase_4_anchors_snapshots_results.md](phase_4_anchors_snapshots_results.md).

## Goal

- Gemma and LFM2 reach within 5 points of the ideal hit rate for stages 1 and 2 (Phase 0: Gemma 0-35%, LFM2 stage 3 6-18% with the `inline` layout).
- Clients can mark "this prefix will be reused" (for example the end of a system prompt) without knowing token positions.

## Work items

### 1. Anchors

- Request field: a per-message flag (for example `"cache_anchor": true` on a chat message) meaning "the prefix up to the end of this message will be reused".
- After templating, map each anchor to a token position. Reuse the approach behind `message_spans` in the chat template code; do not add a second template pass.
- An anchor becomes a node boundary in the tree (a forced split) and gets a parked ID in tier 0, so it survives when its leaves are evicted.
- Anchor priority comes from the request (default high); pinned anchors count against the pin quota (design.md, "Pins need limits").

### 2. Snapshot positions

- Grid: every `S` tokens, with `S` a multiple of `n_ubatch` and of `B`. New flag `--cache-tree-snapshot-every S`.
- Anchors: always.
- Prefill is split so that a ubatch ends exactly at each snapshot position, and the snapshot is taken there. Reuse the context checkpoint code (`ctx_checkpoints`, `common_prompt_checkpoint`) for the save itself.

### 3. What a snapshot holds

| Model type | Snapshot | API |
| --- | --- | --- |
| SWA (Gemma) | The last window of cells of the SWA layers | `llama_state_seq_get_data_ext(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` |
| Hybrid (LFM2) | The recurrent state at that position | Same flag |
| Full attention | Nothing extra | - |

A node keeps a list of snapshot positions and their data. Pruning uses recompute distance, not token count (design.md, "Pruning short runs").

### 4. Restore with snapshots

`match` returns the deepest snapshot at or before the matched length. Restore attention blocks up to there, load the snapshot, and prefill the rest. For Gemma, attention blocks outside the window are not needed for the SWA layers, but the full-attention layers still need them.

## Measurements

Both devices, Gemma and LFM2, `inline` and `turns` layouts, 3 reps, configs A, C, F with the tree, and anchors set at the end of each stage's system prompt and of the shared history.
- Sweep `S` in 512, 1024, 2048: snapshot memory against hit rate.
- Settle the Phase 0 open question: why Gemma's stage 3 reaches 51% in config C on the phone but 0% on the laptop.

## Exit criteria

- Gemma and LFM2 stage 1 and 2 within 5 points of the ideal on both devices.
- LFM2 `inline` stage 3 within 5 points of the `turns` layout.
- Parked-ID cost for LFM2 measured (Phase 3 risk).

## Risks

- **Snapshot size.** Recurrent states are large compared with one token's K/V. Admission (only snapshots likely to be reused, as in Marconi) may be needed.
- **SWA masking in saves.** The state writer (`llama_kv_cache::state_write`) skips SWA cells that are masked relative to the saved sequence's last position, so a block save through a scratch sequence keeps the wrong cells for SWA layers. Block saves for Gemma need a check or a fix here before Phase 4 relies on them.
