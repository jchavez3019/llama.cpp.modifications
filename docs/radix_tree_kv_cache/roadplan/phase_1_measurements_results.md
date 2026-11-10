# Phase 1 results: measurements and block prototype

Status: in progress. The block benchmark and the fit are done on both devices. The BUG-001 PR tests and the server telemetry are still open. The plan is [phase_1_measurements.md](phase_1_measurements.md).

## Summary

Restoring is cheaper than recomputing for every model on both devices, even for runs of 16 tokens. So the restore threshold \(T\) is about 0-6 tokens instead of the 10-100 tokens that design.md estimated. Block save and restore with today's library calls gives the same logits as a cold prefill (0 difference in 39 of 39 checks). On the phone, the main cost of a block design is the tokens that do not fill a last block and must be recomputed.

## Setup

These are the runs behind the numbers. Raw JSON lines are in `bench-results/phase1/` (not in git).

| | Laptop | Phone |
| --- | --- | --- |
| Device | RTX 3060 Laptop GPU, i7-12700H | Pixel Fold, CPU only |
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

### Restore against recompute

The fit is `restore = t_overhead * calls + t_copy * N` and `prefill = fixed + t_prefill * N`, with \(T = t_\text{overhead} / (t_\text{prefill} - t_\text{copy})\).

| Device | Model | \(t_\text{prefill}\) ms/token | \(t_\text{overhead}\) ms/call | \(t_\text{copy}\) ms/token | \(T\) tokens | Checks |
| --- | --- | --- | --- | --- | --- | --- |
| laptop | Llama | 0.151 | 0.094 | 0.0339 | 0.8 | 17/17 pass |
| laptop | LFM2 | 0.139 | 0.72 | 0.0122 | 5.7 | 5/5 pass |
| laptop | Gemma | 0.125 | 4.1 | 0.0067 | 34.8 (fit artifact, see below) | 5/5 pass |
| phone | Llama | 14.7 | 0.106 | 0.0049 | 0.0 | 13/13 pass |
| phone | LFM2 | 15.9 | 0.165 | 0.0020 | 0.0 | 4/4 pass |
| phone | Gemma | 28.2 | 0.352 | 0.0028 | 0.0 | 4/4 pass |

Every check has the same top token and a largest logit difference of 0.

The Gemma \(T\) on the laptop is not real. Its saved state stops growing once the SWA window is full (6.8 MB at 256 tokens, 28 MB at 4,096 instead of about 109 MB), so a line in \(N\) has a large intercept. At the smallest run measured, 16 tokens, a restore takes 0.51 ms against 2.0 ms of marginal prefill, so the real \(T\) is below 16.

The same comparison at 1,024 tokens, Llama, median time:

| Device | Recompute (prefill after the prefix) | Whole restore | Ratio |
| --- | --- | --- | --- |
| laptop | 145 ms | 32 ms | 4.5x |
| phone | 14,839 ms | 4.7 ms | about 3,000x |

### State size and copy rate

The size per token decides how much a tier can hold.

| Model | Saved state | Laptop copy rate | Phone copy rate |
| --- | --- | --- | --- |
| Llama | 32 KiB per token | about 1.0 GB/s | about 7.2 GB/s |
| LFM2 | about 165 KB fixed (recurrent state) plus 12 KB per token | about 1.0 GB/s | about 5.8 GB/s |
| Gemma | 26.7 KB per token up to the SWA window, then only the global layers grow | about 1.0 GB/s | about 5.1 GB/s |

The laptop copies at about 1 GB/s, far below PCIe 4.0 x8 (about 13 GB/s). The state calls copy each layer's tensors through pageable host memory, so this is a software limit, not a hardware one.

### Block size (Llama only)

Block calls cost more than one whole call because of the per-call overhead (about 0.1 ms on both devices) and the scratch-sequence tagging on save.

| Device | Tokens | Whole restore | `B` 16 | `B` 64 | `B` 256 | Whole save | `B` 16 | `B` 64 | `B` 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | 4,096 | 138 ms | 163 ms | 145 ms | 142 ms | 152 ms | 313 ms | 222 ms | 168 ms |
| phone | 1,024 | 4.7 ms | 11.8 ms | 7.1 ms | 5.7 ms | 4.6 ms | 30.0 ms | 23.5 ms | 7.3 ms |

The fit picks `B` = 16 on both devices for the Phase 0 run lengths (80, 300, 600, 3,000 tokens). That choice comes from the tokens that do not fill a last block: the cost model assumes about \((B-1)/2\) of them are recomputed per restore. On the phone, that is about 110 ms at `B` 16 and about 460 ms at `B` 64, much more than any copy cost.

### Laptop throttling

The first laptop run was throttled and is archived in `bench-results/phase1-archive/laptop-throttled/`. The GPU was at 79 C at idle, with throttle reasons "software power cap" and "software thermal slowdown". `llama-bench` gave 812 tokens/s for Gemma pp256, against 9,331 in Phase 0. The rerun waited until the GPU was at 60 C, and `llama-bench` pp512 gave 8,372-8,929 tokens/s, with no throttle reason during the runs.

The throttled run's copy cost (0.0335 ms/token for Llama) matches the cool run's (0.0339), because copies use the CPU and the bus, not the GPU cores. Its prefill cost was 20% higher for Llama, and its Gemma and LFM2 numbers are not usable.

### BUG-001 fixes

Not run yet. Results for [#30866](https://github.com/ggml-org/llama.cpp/pull/30866) and [#28992](https://github.com/ggml-org/llama.cpp/pull/28992) in config E go here.

### Cache telemetry

Not done yet.

## Insights

What the numbers change in the design.

| Finding | Consequence | Phase |
| --- | --- | --- |
| \(T\) is below 16 tokens everywhere, and 0 on the phone | The restore rule and run-based pruning almost never trigger. Keep them, but tier capacity and eviction matter far more | 2 |
| Phase 0's slow laptop swap (131 ms per turn) is mostly whole-state size, at about 1 GB/s | Restoring only the shared prefix, not the whole prompt, is what makes RAM restores cheap | 2 |
| Block restores cost 3-23% more than one whole restore at `B` 64-256 on the laptop | Per-call overhead is small enough that a "load many blocks" library call is not needed yet | 2 |
| Block save at `B` 16 costs 2x (laptop) to 6.5x (phone) a whole save | Saves should happen off the request path, or use a larger `B` for saves | 2 |
| On the phone, recomputing an unfilled last block costs seconds per restore | Store a leaf's last partial block too, so nothing is recomputed. Then `B` is set by copy cost only, and 64-256 is better | 2 |
| The laptop copy rate is a software limit | Pinned host memory or one copy per tensor could give a large speed-up. That is a library change, so it goes on the upstream track only with numbers | upstream |
| Throttling slows prefill by 20% to 11x | Every laptop run starts from a cool GPU and logs its state (`run_laptop.sh`) | all |

## Changes to later phases

- [phase_2_ram_tree.md](phase_2_ram_tree.md) should save a leaf's last partial block, and start with `B` = 64 on both devices, then confirm it with the Phase 2 matrix.
- [design.md](../design.md), "Restore or recompute": replace the 10-100 token estimate with the measured \(T\). This is not done yet; it waits until the BUG-001 tests and telemetry close Phase 1.

## Open questions

- The phone runs were not checked for throttling. The phone values are comparable with Phase 0, which was measured the same way, but they are not peak values.
- Block storage for Gemma (SWA) and LFM2 (recurrent) is not tested. SWA masking relative to the scratch sequence, and recurrent state that cannot be split by position, are Phase 4 questions.
- Pinned host memory for tier 1: how much of the 13 GB/s bus can the laptop reach, and what does it cost in RAM that cannot be swapped?
