# Phase 5 plan: flash tier

Phase 5 adds tier 2: blocks are written to files, so prefixes survive a server restart and the phone can keep more prefixes than its RAM holds. It needs only the Phase 2 tree. Results go in [phase_5_flash_tier_results.md](phase_5_flash_tier_results.md).

## Goal

- After a restart, the first turn reuses the stage system prompts (time to first token near a warm turn, not a cold one).
- On the phone, a small RAM tier plus flash matches a large RAM tier within 10% turn time.

## Work items

### 1. Files

- `--cache-dir PATH` enables tier 2; `--cache-disk MIB` limits its size.
- One file per block, named by the chained hash of the path up to the block plus the block index. The hash includes the model file hash, the KV types, and the LoRA set, so a file can only match the model that wrote it.
- A small manifest per directory lists block files with their token spans, so the index can be rebuilt at startup without reading block data.

### 2. Moving data

| Event | Action |
| --- | --- |
| Demotion from RAM | Write the block file if the node's tier-2 run is at least \(T_2\); otherwise drop it (pruning rule) |
| Restore | Read block files into RAM buffers, then the Phase 2 restore into the slot |
| Eviction | Bottom-up by the same ranking; delete files |
| Startup | Read the manifest, rebuild nodes as tier 2 only |

Writes happen off the request path (a background thread), so a turn never waits on flash.

### 3. Safety

Files on disk are input to the server. Every read checks the header (magic, version, model hash, KV types, token count, byte size) before using any field, and bounds every size and count before allocating. A file that fails a check is deleted and treated as a miss.

## Measurements

- Phone: configs A and F with the tree, `--cache-ram 64` plus flash, against `--cache-ram 1024` (Phase 2).
- Restart test: run 3 turns, restart the server, run 3 more; compare TTFT of the first turn after restart with and without `--cache-dir`.
- Raw numbers: read and write speed of block files on both devices, and \(T_2\) from the Phase 1 fit with flash read time.

## Exit criteria

- After a restart, stage 1 and 2 hit rates of the first turn equal a warm turn.
- Phone with small RAM plus flash within 10% of large RAM in turn time.
- No crash or wrong output with corrupted, truncated, or foreign files (tests with damaged files).

## Risks

- **Page cache hides flash cost.** Reads after a write may come from RAM. Measure restore after a restart and after filling memory, not only right after a write.
- **Flash wear on the phone.** Limit writes: only blocks above \(T_2\), and only nodes with at least one hit.
