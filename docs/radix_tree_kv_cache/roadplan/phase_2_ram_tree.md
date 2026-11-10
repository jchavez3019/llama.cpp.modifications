# Phase 2 plan: tree over RAM with block storage

Phase 2 replaces the flat RAM prompt cache (`server_prompt_cache`, a list of whole slot states) with a radix tree whose nodes keep their K/V as `B`-token blocks in RAM. Slots still own the KV pool. Results go in [phase_2_ram_tree_results.md](phase_2_ram_tree_results.md).

## Goal

Fix the two biggest Phase 0 losses without touching how slots own cells.

- **Config F (overwrite, no RAM cache):** stages 1 and 2 should reuse 85-89% again, because their prefixes stay in the tree after the slot is overwritten.
- **Config D (small RAM cache):** a shared prefix is stored once, not once per whole prompt, so stage 3 fits.

## Scope

In scope: full-attention models (Llama), no draft model. Out of scope: parked sequences in the KV pool (Phase 3), anchors and snapshots (Phase 4), flash (Phase 5). For SWA and hybrid models, and when a draft model is loaded, the server keeps today's `server_prompt_cache` and logs why.

## Work items

### 1. Tree data structure

New files `tools/server/server-prompt-tree.h` and `server-prompt-tree.cpp`, next to `server-task.cpp` where `server_prompt_cache` lives.

| Element | Content |
| --- | --- |
| Node | Token run `[p0, p1)` and its tokens; children keyed by first token; list of block references; hit count with aging, last use, priority; reference count |
| Block | `B` tokens of K/V at positions `[k*B, (k+1)*B)`, saved with the Phase 1 method; owned by the node whose run contains the block's last token |
| Tree | Root, total bytes, byte limit (`--cache-ram`), block size `B`, restore threshold \(T_1\) |

Operations, all on the index only (no llama calls), so they can be unit tested:
- `match(tokens)`: longest matching path, the matched length, and the restorable length (block-aligned).
- `insert(tokens, n_known)`: add or split nodes for the new tokens; returns the block positions that still need data.
- `evict(bytes_needed)`: bottom-up, ranked by priority, then hits with aging, then recency, with run-based pruning (design.md, "Pruning short runs").

Operations that move data, in the same file, using a scratch sequence ID:
- `save_blocks(ctx, seq, positions)`: Phase 1 block save.
- `restore(ctx, seq, path, p_from, p_to)`: Phase 1 block restore, then `seq_rm` of positions past `p_to` (trim).

### 2. Server integration

- **Flag:** `--cache-tree` (off by default) uses the tree in place of `server_prompt_cache`, with the same `--cache-ram` limit. `--cache-tree-block N` sets `B`, and `--cache-tree-min-restore N` sets \(T_1\) (defaults from Phase 1).
- **Scratch sequence:** context creation uses `n_seq_max = n_parallel + 1`; the last ID is the scratch sequence. Requires `--kv-unified`.
- **Save point:** where `prompt_save` runs today (`get_available_slot` before a slot is reused, and idle-slot caching), call `insert` with the slot's tokens and save only blocks the tree does not have. A prefix already in the tree costs nothing to save again.
- **Load point:** where `prompt_load` runs today, compare the slot's own common prefix (tier 0) with the tree match. If the tree gives more than \(T_1\) extra tokens, `seq_rm` the slot from the block-aligned slot prefix, then `restore` the tree's blocks up to the match. The slot's `prompt.tokens` is set to the restored tokens, and `update_slots()` prefills the rest as today.
- **Telemetry:** the Phase 1 log line gains the source `tree`, blocks loaded, and restore time.

### 3. Tests

- Index tests in `tests/test-server-component.cpp` (the place for server component tests): split inside a node, match across splits, block ownership at a boundary, bottom-up eviction order, the run-based pruning cascade (the three examples in design.md).
- One end-to-end check: config F with `--cache-tree`, Llama, and greedy outputs equal to config C for the same seed.

## Measurements

Rerun the Phase 0 matrix subset on both devices, Llama only, 3 reps:

| ID | Setup | Compare with |
| --- | --- | --- |
| A-tree | A with `--cache-tree` | A |
| D-tree | D with `--cache-tree` | D |
| F-tree | F with `--cache-tree --cache-ram <A's size>` | F and C |

Report: hit rate per stage, TTFT, turn time, energy (phone), RAM used by the tree, restore time per turn.

## Exit criteria

- F-tree stage 1 and 2 hit rates within 2 points of the ideal on both devices.
- D-tree stage 3 hit rate above 50% on both devices.
- No change in greedy outputs against C.
- Restore time per turn below A's swap time (85 ms phone, 131 ms laptop).

## Risks

- **Per-block call overhead.** If Phase 1 shows it is large, add "load a list of blocks" as a library call first (a small, separate change).
- **Positions.** Blocks store absolute positions; a restore is only valid at the same positions. Context shift and `--cache-reuse` must be off with `--cache-tree` until Phase 3 adds copy-on-write.
