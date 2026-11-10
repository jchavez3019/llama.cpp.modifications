# Phase 1 results: measurements and block prototype

Status: all work items are done. The plan is [phase_1_measurements.md](phase_1_measurements.md).

## Summary

Restoring is cheaper than recomputing for every model on both devices, even for runs of 16 tokens. So the restore threshold \(T\) is about 0-6 tokens instead of the 10-100 tokens that design.md estimated. Block save and restore with today's library calls gives the same logits as a cold prefill (0 difference in 39 of 39 checks). The cost of blocks is the per-call overhead (about 0.1 ms per restore call and 0.4 ms per save call) and, on the phone, the tokens that do not fill a last block if they are recomputed.

## Setup

These are the runs behind the numbers. Raw JSON lines are in `bench-results/phase1/` (not in git).

| | Laptop | Phone |
| --- | --- | --- |
| Device | RTX 3060 Laptop GPU, i7-12700H, PCIe Gen 4 x8 | Pixel Fold, CPU only |
| Command | `run_laptop.sh` (waits until the GPU is at or below 60 C) | `taskset f0 llama-kv-block-bench -ngl 0 -t 4 -c 4096` |
| Prefix before `prefill_offset` | 1,500 tokens | 300 tokens |
| \(N\) | 16, 64, 256, 1024, 4096 | 16, 64, 256, 1024 |
| Reps | 5 after 1 warm-up | 3 after 1 warm-up |
| Models | Llama 3.2 1B, Gemma 3 1B, LFM2 1.2B, all Q4_K_M | same |

To reproduce on the laptop:

```bash
tools/server/bench/kv-block-bench/run_laptop.sh bench-results/phase1/laptop ~/models/*-Q4_K_M.gguf
python tools/server/bench/kv-block-bench/kv_block_fit.py bench-results/phase1/*/*.jsonl
```

## Numbers

All laptop numbers below come from runs on a PCIe Gen 4 link. Earlier runs on a link stuck at Gen 1 are archived (see "Laptop PCIe link").

### Restore against recompute

The fit is `restore = t_overhead * calls + t_copy * N` and `prefill = fixed + t_prefill * N`, with \(T = t_\text{overhead} / (t_\text{prefill} - t_\text{copy})\).

| Device | Model | \(t_\text{prefill}\) ms/token | \(t_\text{overhead}\) ms/call | \(t_\text{copy}\) ms/token | \(T\) tokens | Checks |
| --- | --- | --- | --- | --- | --- | --- |
| laptop | Llama | 0.126 | 0.105 | 0.0044 | 0.9 | 17/17 pass |
| laptop | LFM2 | 0.115 | 0.107 | 0.0017 | 0.9 | 5/5 pass |
| laptop | Gemma | 0.105 | 0.62 | 0.0010 | 5.9 (fit artifact, see below) | 5/5 pass |
| phone | Llama | 14.7 | 0.106 | 0.0049 | 0.0 | 13/13 pass |
| phone | LFM2 | 15.9 | 0.165 | 0.0020 | 0.0 | 4/4 pass |
| phone | Gemma | 28.2 | 0.352 | 0.0028 | 0.0 | 4/4 pass |

Every check has the same top token and a largest logit difference of 0.

The Gemma \(T\) on the laptop is not real. Its saved state stops growing once the SWA window is full (6.8 MB at 256 tokens, 28 MB at 4,096 instead of about 109 MB), so a line in \(N\) has a large intercept. At the smallest run measured, 16 tokens, a restore takes 0.22 ms against 1.7 ms of marginal prefill, so the real \(T\) is below 16.

The same comparison at 1,024 tokens, Llama, median time:

| Device | Recompute (prefill after the prefix) | Whole restore | Ratio |
| --- | --- | --- | --- |
| laptop | 120 ms | 4.3 ms | 28x |
| phone | 14,839 ms | 4.7 ms | about 3,000x |

### State size and copy rate

The size per token decides how much a tier can hold. The copy rate is for one whole-state restore of the largest \(N\).

| Model | Saved state | Laptop copy rate | Phone copy rate |
| --- | --- | --- | --- |
| Llama | 32 KiB per token | about 8.0 GB/s | about 7.2 GB/s |
| LFM2 | about 165 KB fixed (recurrent state) plus 12 KB per token | about 7.0 GB/s | about 5.8 GB/s |
| Gemma | 26.7 KB per token up to the SWA window, then only the global layers grow | about 6.3 GB/s | about 5.1 GB/s |

The laptop saves (VRAM to RAM) at about 5.6 GB/s and restores at about 8.0 GB/s for Llama. A Gen 4 x8 link carries about 13 GB/s in practice. The state calls copy through pageable host memory, one small copy per layer, so pinned host memory could raise the copy rate by at most about 1.5x.

### Block size (Llama only)

Block calls cost more than one whole call because of the per-call overhead: about 0.10 ms per restore call and 0.42 ms per save call on both devices. Each call scans the KV cells, makes one small copy per layer and per K/V tensor, and a save also tags and untags the scratch sequence.

| Device | Tokens | Whole restore | `B` 16 | `B` 64 | `B` 256 | Whole save | `B` 16 | `B` 64 | `B` 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | 4,096 | 16.8 ms | 44.3 ms | 26.6 ms | 19.3 ms | 24.2 ms | 138 ms | 75 ms | 36.5 ms |
| phone | 1,024 | 4.7 ms | 11.8 ms | 7.1 ms | 5.7 ms | 4.6 ms | 30.0 ms | 23.5 ms | 7.3 ms |

The fit picks `B` = 64 on the laptop and `B` = 16 on the phone, for the Phase 0 run lengths (80, 300, 600, 3,000 tokens). Both choices assume that the tokens after the last full block, about \((B-1)/2\) per restore, are recomputed. On the phone, that costs about 110 ms at `B` 16, 460 ms at `B` 64, and 1.9 s at `B` 256. Saving them as a short last block costs about one call, 0.1 ms.

How `B` affects memory: a block that crosses a node boundary belongs to each child, so a branch point with `k` children stores \((k-1) \times (p \bmod B)\) extra tokens, where `p` is the branch position. [design.md, "Blocks at a branch point"](../design.md#blocks-at-a-branch-point) explains this step by step. For Llama that is on average 0.23 MiB per extra child at `B` 16, 0.98 MiB at `B` 64, and 3.98 MiB at `B` 256. A turn of config A saves 266 MiB of state today, so this duplication stays at a few percent or less.

### Laptop PCIe link

The first laptop runs on Nov 9 used a PCIe link that stayed at Gen 1 x8 even at 100% GPU load (`nvidia-smi`: current Gen 1, max Gen 4, no replays). The laptop had run on very low battery earlier that day, and plugging it in did not restore the link. A reboot did. On Gen 1, copies ran at about 1 GB/s instead of 6-8 GB/s, and the server's swaps took about 5x longer. Those runs are archived in `bench-results/phase1-archive/laptop-gen1/` and `telemetry-gen1/`. They do not change the conclusions: a slower link only raises \(t_\text{copy}\), and \(T\) was still below 16 tokens.

### Laptop throttling

The first laptop run was also throttled and is archived in `bench-results/phase1-archive/laptop-throttled/`. The GPU was at 79 C at idle, with throttle reasons "software power cap" and "software thermal slowdown". `llama-bench` gave 812 tokens/s for Gemma pp256, against 9,331 in Phase 0. Later runs waited until the GPU was at 60 C, and had no throttle reason during the runs.

### BUG-001 fixes

Both upstream PRs fix BUG-001: the pinned background task now reuses 41.6% of its prompt, which is the ideal, against 0% for stock. Neither PR fixes BUG-002.

Each PR's net diff was applied onto `ca89968a2`, the upstream commit Phase 0 used, so only the fix differs from the stock control. [#28992](https://github.com/ggml-org/llama.cpp/pull/28992) had two conflicts with newer upstream code. Its cache check was merged with upstream's `!task.is_stateless()` check, and its tests were kept next to the newer upstream tests. Laptop, Llama, 3 reps each, no cool-down waits. Raw results are in `bench-results/phase1/bug001/`.

| Binary | Config E background hit | Background prefill tokens | RAM restores | Config A with `--slot-prompt-similarity 0`, stage 1 / 2 / 3 hit |
| --- | --- | --- | --- | --- |
| stock | 0.0% | 1,035 | 63 | 4.0% / 3.0% / 0.3% |
| [#30866](https://github.com/ggml-org/llama.cpp/pull/30866) "Pin empty slot" | 41.6% | 605 | 75 | 4.0% / 3.0% / 0.3% |
| [#28992](https://github.com/ggml-org/llama.cpp/pull/28992) "fix skipped prompt cache lookups" | 41.6% | 605 | 75 | 4.0% / 3.0% / 0.3% |

The ideal hit rates are 41.6% for the background task, and 85.7% / 89.2% / 56.4% for the three stages.

- **Stage 1 also gains.** In config E it goes from 81.4% to 85.7% (the ideal) with either PR. The stage-1 request sometimes also lands on an empty pinned slot.
- **"Skipping, slot is empty" is not a fix signal.** It stays at 13-15 per run with both PRs. The line comes from the similarity loop, which still skips empty slots; the fixes act after the loop.
- **Why #28992 does not fix BUG-002.** It asks the RAM cache for a better start (`has_better`) even when `update_cache` is false. But with similarity 0 and pinned stages, nothing ever saves a state to the RAM cache (0 RAM restores), so there is never a better entry to find. A fix must also save the outgoing state, as BUGS.md proposes.
- **The PRs differ in scope.** #30866 is one line and covers only the empty pinned slot. #28992 also covers a pinned slot that holds a shorter prefix than a cached entry, which config E does not exercise.

These runs used the Gen 1 link, so all of them, the stock control included, were about 45% slower than Phase 0. The hit rates do not depend on the link, and the three binaries compare fairly with each other; their times do not compare with Phase 0.

### Cache telemetry

The server now logs, at info level, one line per RAM cache save and one line per prompt start, in `tools/server/server-context.cpp` (fork only):

```
prompt cache save: 636 tokens, 19.883 MiB, 21.94 ms
prompt cache use: src = ram, reused = 532, prefill = 89, load = 19.54 ms, 19.883 MiB
```

`src` is `ram` when the slot state was restored from the RAM cache for this prompt, `slot` when the slot's own cells were reused, and `none` otherwise. `reused` is the final `n_past`, after checkpoints and the "evaluate at least 1 token" rule. A restore is detected by the cache losing an entry, because `server_prompt_cache::load` returns true also when no entry is better. `summary.py` adds the columns "Save ms/turn", "Load ms/turn", and "Src ram/slot/none".

Rerun of configs A and E, laptop on Gen 4, Llama, 3 reps (stock code plus telemetry; `bench-results/phase1/telemetry/`):

| Config | Turn p50 | Swap ms/turn | Save ms/turn | Load ms/turn | Src ram / slot / none | Server save rate | Server load rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1,716 ms | 132 | 90 | 42 | 64 / 4 / 1 | 3.0 GB/s | 6.0 GB/s |
| E | 1,816 ms | 128 | 96 | 44 | 63 / 4 / 14 | 2.9 GB/s | 5.6 GB/s |

- **The times match Phase 0** (config A: 1,720 ms per turn and 131 ms of swap), so the Gen 4 laptop is back to the Phase 0 baseline.
- **`reused` matches the client.** In every turn of config A, it equals the `cache_n` the client received.
- **BUG-001 is visible directly.** Config E has 14 prompts with `src = none`: the pinned background requests on an empty slot. Config A has 1, the first prompt.
- **Saves cost more than loads.** In config A, a turn saves 266 MiB (the outgoing state of every stage) and loads 253 MiB. Saves are 68-75% of the swap time.
- **Server saves run at half the benchmark's rate** (3.0 against 5.6 GB/s), while loads are close (6.0 against 8.0 GB/s). The server allocates a new buffer for every save, which may add page faults; this is not verified.

## Insights

What the numbers change in the design.

| Finding | Consequence | Phase |
| --- | --- | --- |
| \(T\) is below 16 tokens everywhere, and 0 on the phone | The restore rule and run-based pruning almost never trigger. Keep them, but tier capacity and eviction matter far more | 2 |
| A swap moves whole states: 266 MiB saved and 253 MiB loaded per turn in config A | Store shared prefixes once and skip saves of states the tree already holds | 2 |
| On Gen 4, per-call overhead dominates block costs: at `B` 16 a restore costs 2.6x and a save 5.7x one whole-state copy | Build the tree with `B` 64 and the existing calls, then add batched block calls and move to `B` 16 if they remove the overhead | 2 |
| Recomputing the tokens after the last full block costs 100 ms to 2 s per restore on the phone, depending on `B` | A leaf keeps a short last block, so nothing is recomputed | 2 |
| Duplication at branch points is about 1 MiB per extra child at `B` 64 | Memory does not decide `B`; call overhead does | 2 |
| Server saves copy at half the benchmark's rate | Check buffer allocation in the save path; saves are the larger part of the swap | 2 |
| Pageable copies already reach 5.6-8 GB/s of about 13 GB/s on Gen 4 | Pinned host memory can give at most about 1.5x on copies; batching first | 2 |
| Both upstream PRs fix BUG-001; neither fixes BUG-002 | Phase 2 and 3 runs can use either PR as the stock baseline for config E. BUG-002 stays open | 2, 3 |
| The PCIe link can stay at Gen 1 after low battery, and the GPU can throttle | Before a laptop session, check under load that the link reaches Gen 4, and start from a cool GPU (`run_laptop.sh`) | all |

## Changes to later phases

- [phase_2_ram_tree.md](phase_2_ram_tree.md): `B` defaults to 64, a leaf keeps a short last block, \(T_1\) defaults to 0, and batched block calls are a later Phase 2 step with a possible move to `B` 16.
- [design.md](../design.md), "Restore or recompute": replace the 10-100 token estimate with the measured \(T\). Not done yet.

## Open questions

- The phone runs were not checked for throttling. The phone values are comparable with Phase 0, which was measured the same way, but they are not peak values.
- Block storage for Gemma (SWA) and LFM2 (recurrent) is not tested. SWA masking relative to the scratch sequence, and recurrent state that cannot be split by position, are Phase 4 questions.
- Why do server saves copy at half the benchmark's rate?
- How close to one whole-state copy can a batched block call get, for save and for restore?
