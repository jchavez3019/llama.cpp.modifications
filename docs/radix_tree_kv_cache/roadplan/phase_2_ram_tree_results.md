# Phase 2 results: Tree over RAM with block storage

Status: done, Llama only, laptop and phone. The plan is [phase_2_ram_tree.md](phase_2_ram_tree.md).

## Summary

The tree fixes the largest Phase 0 loss. With the stock flags plus `--cache-tree` (F-tree), stages 1 and 2 reuse the ideal 85-89% instead of 3-6%. The phone turn drops from 45.3 s to 20.1 s, and phone energy per turn halves. Config D is no worse than stock D, and restores are bit-identical. Three of the four exit criteria are met. The greedy-output criterion is narrowly missed on a sample of 24 requests.

A batched block save (`llama_state_seq_get_data_ranges`) was added after the first measurements. It writes each block straight into its own buffer and cuts the server save cost by 13% (laptop) and 10% (phone). The rest of the save cost is mostly new memory. Block restores stay one call per block, so `B` stays at 64.

## Exit criteria

Each criterion from the plan, checked on both devices.

| Criterion | Result | Detail |
| --- | --- | --- |
| F-tree stage 1 and 2 hit within 2 points of the ideal | met | Exactly the ideal on both devices (85.7 / 89.2% laptop, 84.9 / 88.0% phone) |
| D-tree hit rates and turn time no worse than D | met | Same hits as D. Turn 2109 vs 2112 ms (laptop), 31.3 vs 36.5 s (phone) |
| `check_tree_*` bit-identical; F-tree greedy outputs differ from no cache no more often than C's | partly | Bit-identical on both devices. F-tree differs on 13 of 24 requests, C on 10 of 24 (laptop, one run) |
| Restore time per turn below A's swap time | met | 37 vs 131 ms (laptop), 12 vs 85 ms (phone) |

## Numbers

Median of 3 reps, Llama 3.2 1B Q4_K_M. Phase 0 is the stock server with the same config. Hit rates are stage 1 / 2 / 3.

| Device | Config | Hits, Phase 0 | Hits, tree | Turn p50, Phase 0 | Turn p50, tree | Energy per turn, Phase 0 | Energy per turn, tree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | A | 85.7 / 89.2 / 56.4 | same | 1720 ms | 1710 ms | 99.0 J | 99.2 J |
| laptop | D | 85.7 / 89.2 / 0.3 | same | 2112 ms | 2109 ms | 126.7 J | 127.3 J |
| laptop | F | 4.0 / 3.0 / 0.3 | 85.7 / 89.2 / 56.4 | 2227 ms | 1714 ms | 133.4 J | 99.3 J |
| phone | A | 84.9 / 88.0 / 54.6 | same | 22.0 s | 20.5 s | 41.1 J | 42.7 J |
| phone | D | 84.9 / 88.0 / 1.5 | same | 36.5 s | 31.3 s | 64.8 J | 71.4 J |
| phone | F | 6.4 / 5.1 / 1.5 | 84.9 / 88.0 / 54.6 | 45.3 s | 20.1 s | 81.2 J | 42.7 J |

Laptop energy is GPU board power only. Config C (one slot per stage, the reference for "nothing lost") is 1589 ms on the laptop and 22.5 s on the phone.

Cache cost per turn, from the server telemetry:

| Device | Config | Save ms/turn | Load ms/turn | Phase 0 swap ms/turn | Peak RSS, Phase 0 | Peak RSS, tree |
| --- | --- | --- | --- | --- | --- | --- |
| laptop | A-tree | 87.9 | 36.7 | 131 | 1299 MiB | 3376 MiB |
| laptop | D-tree | 14.8 | 9.6 | 21 | 717 MiB | 790 MiB |
| phone | A-tree | 30.0 | 12.2 | 85 | 1872 MiB | 2193 MiB |
| phone | D-tree | 8.5 | 5.9 | 36 | 1817 MiB | 1821 MiB |

Results are in `bench-results/phase2/` (git-ignored).

### Batched block save

Saving all new blocks of a slot with one library call instead of one call per block. Results are in `bench-results/phase2-batch/` (one shared buffer) and `bench-results/phase2-batch-direct/` (one buffer per block, the final version).

`kv-block-bench` with the first version (all blocks into one shared buffer that the bench reuses), median of 3 reps (laptop: 4,096 tokens; phone: 1,024 tokens):

| Device | Whole save | `B` 16, per block | `B` 16, batched | `B` 64, per block | `B` 64, batched | Whole restore | `B` 16 restore | `B` 64 restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | 33.1 ms | 135.2 ms | 45.4 ms | 51.4 ms | 41.1 ms | 16.0 ms | 47.6 ms | 27.0 ms |
| phone | 4.0 ms | 23.7 ms | 8.7 ms | 7.6 ms | 6.8 ms | 4.1 ms | 10.2 ms | 6.2 ms |

All `check_block_batch` and `check_tree_*` checks are bit-identical on both devices.

The bench reused one warm buffer, so these numbers hide the cost of new memory. The bench now gets new memory for each save, as the server does. With new memory on the laptop at 4,096 tokens, a save is 157 ms per block and 140 ms batched at `B` 16, and 97 ms and 90 ms at `B` 64.

Server, A-tree, median of 3 reps:

| Device | Save | Save ms/turn | Load ms/turn | Turn p50 | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| laptop | per block | 87.9 | 36.7 | 1710 ms | 3376 MiB |
| laptop | batched, one shared buffer | 105.9 | 35.9 | 1727 ms | 3414 MiB |
| laptop | batched, one buffer per block | 76.9 | 36.6 | 1694 ms | 3377 MiB |
| phone | per block | 30.0 | 12.2 | 20.5 s | 2193 MiB |
| phone | batched, one shared buffer | 59.6 | 12.0 | 20.4 s | 2195 MiB |
| phone | batched, one buffer per block | 27.1 | 11.3 | 19.4 s | 2192 MiB |

Hit rates are the same in all runs. The turn time differences are within run-to-run noise.

### Restore only the missing part

The slot keeps its full blocks of the match, and the tree restores only the blocks after them. Results are in `bench-results/phase2-shared-nokeep/` and `bench-results/phase2-shared-keep/`.

The test workload adds a system prompt part that all requests share at the start: presets `desktop-shared` (1,024 tokens, laptop) and `edge-shared` (768 tokens, phone). In A, the slot then holds the shared part of the last stage, and keeps 16 blocks (laptop) or 12 blocks (phone) of each restore. A-tree, median of 3 reps:

| Device | Restore | Load ms/turn | Blocks loaded per restore | TTFT stage 1 / 2 / 3 | Turn p50 |
| --- | --- | --- | --- | --- | --- |
| laptop | from position 0 | 54.2 | 25-30 | 104 / 35 / 561 ms | 1765 ms |
| laptop | keep the slot's blocks | 35.5 | 9-14 | 96 / 28 / 555 ms | 1738 ms |
| phone | from position 0 | 24.0 | 18-19 | 1.13 / 1.15 / 14.0 s | 22.0 s |
| phone | keep the slot's blocks | 12.1 | 6-7 | 1.33 / 1.36 / 16.8 s | 25.7 s |

Hit rates are the same with and without the change. The phone ran the two configs one after the other without cooling, and its turn time rose in every rep (21.1, 22.0, 23.5, then 24.3, 25.7, 26.6 s), so only its load time is a valid comparison. `check_tree_keep` (restore into a sequence that holds another prompt with the same first block) is bit-identical on both devices.

## Insights

What the numbers alone do not say.

- **Restoring is cheap, recomputing is not.** On the phone the F-tree turn is 20.1 s against 45.3 s for F, with the same restores that A pays for. The tree turns F into A without a second slot.
- **A restore is not free on the laptop.** A-tree's stage 1 TTFT is 105 ms against C's 19 ms: C reuses the slot's own cells, A-tree copies them back from RAM. In A the slot always holds another stage's prompt and shares only the 25-token template header with the new one, so keeping the slot's blocks does not help here, and only parked sequences (Phase 3) can close this gap.
- **Keeping the slot's blocks pays off with a shared system prompt.** Many requests with one shared prompt part and more conversations than slots is the common case for agents and multi-user servers. There, a restore keeps the shared part and copies only the rest: 35% less load time on the laptop and 50% less on the phone. In a chat that only grows on its own slot, the slot already holds everything the tree has, so the tree does not restore at all.
- **The tree grows without a token cap.** On the laptop (the default `--cache-ram` of 8,192 MiB), the tree reaches 2.7 GiB and 94 nodes. Most of it is one-off stage 3 tails that are never matched again. The stock cache has a token cap (n_ctx, 16,384), which holds about 4 prompts.
- **Config D is a capacity limit, not a tree problem.** One stage 3 state is larger than D's limit (117-149 MiB on the laptop against 128 MiB). A save that large would evict every other prompt and then itself, so the tree skips it. Storing shared prefixes once does not make it fit.
- **Per-block call overhead was the save cost.** Before batching, the laptop saved 119 MiB per turn at 1.33 GiB/s, against about 4 GB/s for one whole-state save. Profiling showed one device copy per layer per block (32 per block) and an allocation per block. The batched call reads every block in one `ggml_backend_tensor_get_n_ptr` call.
- **New memory is most of the remaining save cost.** The tree keeps about 119 MiB of new blocks per turn, and the first write to each new 4 KiB page is a page fault (about 0.35 ms per MiB on the laptop, about 40 ms per turn). The first batched version wrote into one shared buffer and then copied each block out, so it paid for new memory twice and was slower than per-block saves on both devices.
- **CUDA's gather kernel needs one compact destination.** With one shared buffer, CUDA pins it and reads all blocks with one kernel. With one buffer per block, the buffers are far apart, so CUDA falls back to one `cudaMemcpyAsync` per read (about 1,900 per turn, about 12 µs each). A block pool (large slabs cut into block-sized slots, with evicted slots reused) would remove both the page faults and the fallback. The best case is about 40 ms per turn on the laptop (2.5% of a turn) and about 10 ms on the phone, which does not justify a new allocator in the tree now.
- **Per-block restore cost has no easy fix.** On CUDA, each restore block makes 32 `cudaMemcpyAsync` calls from pageable memory at about 5 µs each (Nsight Systems). Uploads have no gather kernel, so one shared reader would not remove them. On the phone the cost per block is cell bookkeeping (clear, find cells, scratch copy) of about 0.1 ms, which is small against a 20 s turn.
- **`B` 16 has lost its main reason.** Phase 1 picked `B` 16 for the phone because it assumed that the tokens after the last full block are recomputed. The leaf's short last block already saves them. What `B` still changes is per-call overhead (favours large blocks) and duplicate storage at branch points (up to `B`-1 tokens per branch, 2 MiB at `B` 64).
- **Greedy outputs drift with any cache.** Stock C matches a no-cache run on only 14 of 24 requests. K/V computed in prefill batches of other sizes round differently on the GPU, and the drift grows over a generation. A strict "same output" test is not valid for prompt caches on the GPU; the bit-identical bench check with matched batch shapes is.

## Changes to later phases

Plan or design changes that follow from these results.

- **`B` stays 64.** The block size decision in [phase_2_ram_tree.md](phase_2_ram_tree.md) ("Risks") is closed: saves are batched, restores are not, and `B` 16 no longer saves recompute.
- **Phase 3 (parked sequences)** is the way to make restores free: a prefix that stays in KV cells is restored by a metadata-only `seq_cp`. This matters more than faster copies, on both devices.
- **Eviction needs a priority for one-off tails** before the tree runs without a byte limit. A token cap like the stock cache's is the simple option.

## Open questions

Questions this phase raised and did not answer.

- Does F-tree really diverge from the no-cache run more often than C (13 vs 10 of 24), or is that noise? More requests would settle it.
- Phone D-tree is 5 s faster than D with the same hit rates. Thermal state differed between the runs (no cooling waits), so this may not be real.
- Partial saves for prompts larger than the limit (config D stage 3) need an eviction policy that does not thrash.
- Does a block pool pay off for larger models? Only the 1B model was measured. With grouped-query attention, K/V size grows more slowly than compute (Llama 3.1 8B has 4x the K/V per token and about 8x the compute), so the save share of a turn is expected to be smaller, not larger.
- Hybrid setups, where a request runs on the device or on a server with the same tree: requests share system prompts on both sides, and a server with many nodes also needs prefix-aware routing, so a request goes to the node that holds its prefix. Related work keeps shared prefixes in GPU memory (vLLM automatic prefix caching, SGLang RadixAttention), which is closer to Phase 3.
- Would batched restores help with many small blocks on other backends (Metal, Vulkan) where per-call cost differs?
