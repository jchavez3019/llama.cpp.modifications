# Phase 0: orchestration cache baseline

Phase 0 measures how well the stock `llama-server` caches a cyclic 3-stage agent workload on a laptop GPU and on a phone CPU, and how far it is from a perfect prefix cache. It changes no server code. The numbers decide which later phases are worth doing, and in which order.

To build the binaries used here, see [build.md](build.md).

## Summary

The stock server already reaches the ideal reuse for stages 1 and 2 whenever their state survives, either in a dedicated slot or in the RAM prompt cache. Four cases lose most of the possible reuse: overwriting without a RAM cache, a RAM cache that is too small, the background task (BUG-001), and models whose memory cannot roll back (sliding window and hybrid models).

- **One slot without a RAM cache is the worst case.** In config F all stages share slot 0 and `--cache-ram` is 0, so each stage overwrites the cells of the previous stage. Stages 1 and 2 reuse 0-6% of their prompts instead of 85-89%. On the phone, Llama turns take 2x longer than with one slot per stage and use 2x the energy.
- **On the phone, the RAM cache is as good as one slot per stage.** Config A (one slot plus the RAM cache) matches config C (one slot per stage) within 2% for Llama and LFM2. The swaps cost 85 ms per turn, against 22-second turns. On the laptop, the same swaps cost 131 ms per turn and make A 8% slower than C.
- **A RAM cache that is too small causes an eviction cliff.** In config D, Llama's stage 3 reuse falls from 55-56% to 0.3-1.5%. On the phone, turns get 66% longer and use 58% more energy.
- **BUG-001 reproduces.** The pinned background task gets 0% reuse on every device and model, while 40-52% is possible.
- **Gemma (sliding window) and LFM2 (hybrid) lose reuse whatever the slot setup.** Their states can only roll back to saved checkpoints, and the checkpoints land after the shared prefix. The only good cases are the LFM2 `turns` layout on the laptop, which reaches the ideal 56% in stage 3, and Gemma's stage 3 in config C on the phone (51%).

## Setup

The two devices stress opposite parts of the cache design: the laptop has a small KV pool in VRAM and a large RAM, and the phone has one shared memory but slow prefill.

| Item | Laptop | Pixel Fold |
| --- | --- | --- |
| Compute | i7-12700H, RTX 3060 Laptop (6 GB VRAM), CUDA sm_86 | Tensor G2, CPU only, `taskset f0 -t 4` (2x X1 + 2x A78) |
| Build | `laptop-cuda-release` preset, `-ngl 99` | `android-arm64-release` preset (NDK 26.3, static, `armv8.2-a+dotprod+fp16`) |
| Context | `-c 16384` | `-c 4096` |
| Workload preset | `desktop`, 23 turns (3 warm-up) | `edge`, 12 turns (2 warm-up) |
| RAM cache (A, B, E) | server default (8192 MiB) | `--cache-ram 1024` |
| RAM cache (D) | 128 MiB | 32 MiB |
| Telemetry | RSS from `/proc`, VRAM, GPU power, utilization, and temperature from NVML | Perfetto trace (power rails, RSS, CPU frequency), `dumpsys thermalservice` |

All runs use llama.cpp commit `c8357af02` and Q4_K_M GGUFs of Llama 3.2 1B Instruct (full attention), Gemma 3 1B IT (512-token sliding window), and LFM2 1.2B (hybrid: short convolutions plus attention). Every run starts a fresh server with `--kv-unified -fa on -lv 4`.

### Workload

One turn runs three stages with different static system prompts, like a router, a tool extractor, and a response writer. Every few turns a background task also runs. All text is synthetic and seeded, and each segment is calibrated to an exact token count with `/tokenize`.

| Segment (tokens) | `desktop` | `edge` |
| --- | --- | --- |
| System prompts (stage 1 / 2 / 3) | 500 / 700 / 500 | 300 / 400 / 300 |
| New RAG documents per turn (stage 3) | 3000 | 600 |
| History added per turn (stage 3) | 300 | 80 |
| Generated tokens (stage 1 / 2 / 3) | 16 / 48 / 128 | 8 / 24 / 48 |
| Background system prompt + input | 400 + 600 | 250 + 250 |

Generation uses temperature 0 and `ignore_eos`, so every run sends the same prompts and generates the same number of tokens. In the `inline` layout, the history is text inside one user message. In the `turns` layout, it is sent as real chat messages.

### Configurations

| ID | Server | Client | Question |
| --- | --- | --- | --- |
| A | `-np 2`, RAM cache on | All stages pinned to slot 0 | The stock design: one lane, states swapped through RAM |
| B | Same as A | Unpinned | Does the server's own slot selection do better? |
| C | `-np 4 --cache-ram 0` | One slot per stage | Everything stays in the KV pool |
| D | A with a small RAM cache | All stages on slot 0 | The eviction cliff (BUG-003) |
| E | Same as A | Stages on slot 0, background task on slot 1, concurrent | Background interference (BUG-001) |
| F | `-np 2 --cache-ram 0` | All stages pinned to slot 0 | One lane with no RAM cache: each stage overwrites the previous one |

Each device ran every configuration x 3 models x 3 repetitions. The laptop also ran LFM2 with the `turns` layout for A, B, C, and F. In total that is 66 laptop runs and 54 phone runs, with no failed runs and no request errors.

The phone ran without cool-down breaks. Each repetition rotates the order of the configurations, so the heat affects every configuration about equally. The big cores averaged about 1.2 GHz (maximum 2.85 GHz) and peaked at about 60 C, so the absolute phone times are throttled values.

## How to run

The commands below are run from the repository root, in the `llama-orch-bench` Conda environment. The benchmark and its options are described in [the bench README](../../tools/server/bench/orchestration-bench/README.md).

```bash
conda activate llama-orch-bench
B=tools/server/bench/orchestration-bench/run_matrix.py

# Laptop: all configurations, then the LFM2 layout comparison
python -u $B --target local --models ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf ~/models/gemma-3-1b-it-Q4_K_M.gguf ~/models/LFM2-1.2B-Q4_K_M.gguf \
  --device laptop --configs A,B,C,D,E,F --reps 3 --server-extra '-ngl 99'
python -u $B --target local --models ~/models/LFM2-1.2B-Q4_K_M.gguf --device laptop-turns --layout turns --configs A,B,C,F --reps 3 --server-extra '-ngl 99'

# Phone: binaries and models in /data/local/tmp/llama.cpp
python -u $B --target adb --adb ~/Android/Sdk/platform-tools/adb \
  --models ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf ~/models/gemma-3-1b-it-Q4_K_M.gguf ~/models/LFM2-1.2B-Q4_K_M.gguf \
  --device pixel-fold --preset edge --ctx 4096 --turns 12 --warmup 2 --configs A,B,C,D,E,F --reps 3 \
  --taskset f0 --server-extra '-t 4' --cache-ram-default 1024 --d-cache-ram 32 --port 8090

# Tables
python tools/server/bench/orchestration-bench/summary.py bench-results/phase0
```

The laptop matrix takes about 50 minutes, and the phone matrix about 7.5 hours. The two can run at the same time, because they use different ports and the phone client does little work on the laptop. The first summary of each phone run reads its trace, so the first `summary.py` call takes about 2 minutes. Later calls are fast.

Results are in `bench-results/phase0/<device>/<model>/<config>/rep<k>.*`. Each run has `requests.jsonl`, `meta.json`, `server.log`, and either `samples.jsonl` (laptop) or `pftrace` and `thermal.jsonl` (phone). Results of the earlier A-E runs are kept in `bench-results/phase0-archive-AtoE/`.

## Prefill ceilings

These are the raw `llama-bench` rates (`-fa 1`). Every token a cache saves is worth `1 / prefill rate` seconds.

| Model | Laptop pp512 | Laptop pp2048 | Laptop tg64 | Phone pp512 | Phone pp2048 | Phone tg64 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2 1B | 9383 t/s | 9241 t/s | 226 t/s | 155 t/s | 71 t/s | 15.6 t/s |
| Gemma 3 1B | 9331 t/s | 9737 t/s | 180 t/s | 46 t/s | 44 t/s | 18.2 t/s |
| LFM2 1.2B | 9123 t/s | 9296 t/s | 244 t/s | 155 t/s | 102 t/s | 23.7 t/s |

The phone was cool for the Llama and LFM2 rows and hot for the Gemma row. A thread sweep with Llama (hot) gave the best prefill with `taskset f0 -t 4` (108 t/s). Two cores gave 52 t/s, six cores 67 t/s, and eight cores 73 t/s.

A missed prompt token costs about 0.1 ms on the laptop and 6-22 ms on the phone, which is about 100x more. Cache design matters most on the phone.

## Results

### Turn latency and energy

Turn p50 is the median time of the three stages of one turn. Phone energy is the sum of all power rails over the measured turns. Laptop energy is GPU board power only, so the two are not comparable.

| Device | Model | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Laptop | Llama | 1720 ms | 1744 ms | 1589 ms | 2112 ms | 1808 ms | 2227 ms |
| Laptop | Gemma | 2268 ms | 2262 ms | 2246 ms | 2271 ms | 2348 ms | 2273 ms |
| Laptop | LFM2 | 1826 ms | 1845 ms | 1793 ms | 1841 ms | 1920 ms | 1988 ms |
| Laptop | LFM2 (`turns`) | 1415 ms | 1421 ms | 1364 ms | - | - | 2010 ms |
| Phone | Llama | 22.0 s | 23.0 s | 22.5 s | 36.5 s | 29.1 s | 45.3 s |
| Phone | Gemma | 70.6 s | 71.7 s | 48.0 s | 70.8 s | 79.3 s | 70.9 s |
| Phone | LFM2 | 26.2 s | 26.0 s | 25.9 s | 25.9 s | 31.1 s | 39.4 s |

| Device | Model | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Laptop (GPU J/turn) | Llama | 99 | 100 | 95 | 127 | 105 | 133 |
| Phone (J/turn) | Llama | 41 | 42 | 41 | 65 | 52 | 81 |
| Phone (J/turn) | Gemma | 133 | 134 | 93 | 134 | 149 | 136 |
| Phone (J/turn) | LFM2 | 50 | 50 | 50 | 60 | 60 | 74 |

On the phone, the CPU rails use 54-62% of that energy in every configuration. Peak server RSS is 1.4-2.0 GiB on the phone and 0.5-1.7 GiB on the laptop. C and F use the least, because they have no RAM cache.

### Hit rate per stage

Hit % is the share of prompt tokens the server reused (`cache_n`). Ideal % is the share a perfect prefix cache with no size limit could reuse.

| Device | Model | Stage | Ideal % | A | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Laptop | Llama | 1 | 85.7 | 85.7 | 85.7 | 85.7 | 81.4 | 4.0 |
| Laptop | Llama | 2 | 89.2 | 89.2 | 89.2 | 89.2 | 89.2 | 3.0 |
| Laptop | Llama | 3 | 56.4 | 56.4 | 56.4 | 0.3 | 56.4 | 0.3 |
| Laptop | Llama | background | 41.6 | - | - | - | 0.0 | - |
| Phone | Llama | 1 | 84.9 | 84.9 | 84.9 | 84.9 | 59.5 | 6.4 |
| Phone | Llama | 2 | 88.0 | 88.0 | 88.0 | 88.0 | 88.0 | 5.1 |
| Phone | Llama | 3 | 54.6 | 54.6 | 54.6 | 1.5 | 54.6 | 1.5 |
| Phone | Llama | background | 52.3 | - | - | - | 0.0 | - |
| Laptop | Gemma | 1 / 2 / 3 | 85.1 / 88.8 / 56.2 | 13.3 / 35.1 / 0.0 | 13.3 / 35.1 / 0.0 | 10.6 / 28.1 / 0.0 | 13.3 / 35.1 / 0.0 | 0.2 / 0.1 / 0.0 |
| Phone | Gemma | 1 / 2 / 3 | 83.9 / 87.4 / 53.8 | 0.3 / 0.2 / 0.1 | 0.3 / 0.2 / 51.4 | 0.3 / 0.2 / 0.1 | 0.2 / 0.2 / 0.1 | 0.3 / 0.2 / 0.1 |
| Laptop | LFM2 | 1 / 2 / 3 | 85.2 / 88.9 / 56.2 | 84.4 / 88.3 / 6.4 | 84.4 / 88.3 / 6.4 | 67.5 / 70.7 / 4.8 | 80.2 / 88.3 / 6.4 | 0.0 / 0.0 / 0.0 |
| Laptop | LFM2 (`turns`) | 1 / 2 / 3 | 85.2 / 88.9 / 56.2 | 84.4 / 88.3 / 56.2 | 84.4 / 88.3 / 56.2 | - | - | 0.0 / 0.0 / 0.0 |
| Phone | LFM2 | 1 / 2 / 3 | 84.2 / 87.5 / 53.9 | 82.8 / 86.5 / 18.0 | 82.8 / 86.5 / 18.0 | 49.7 / 51.9 / 10.8 | 66.2 / 86.5 / 18.0 | 0.0 / 0.0 / 0.0 |

B has the same hit rates as A in every case, so it is not in the table. The full tables, with TTFT and prefill tokens per stage, come from `summary.py`.

### Time to first token

Stage 1 TTFT shows the cost of the swaps most clearly, because stage 1 has the shortest prompt.

| Device | Model | A | C | E | F |
| --- | --- | --- | --- | --- | --- |
| Laptop | Llama | 104 ms | 19 ms | 168 ms | 73 ms |
| Phone | Llama | 1.0 s | 1.0 s | 5.2 s | 5.0 s |

On the laptop, config A is slower than F for stage 1. Swapping states through RAM before stage 1 (A) takes longer than recomputing the short stage 1 prompt on the GPU (F). On the phone, the restore is cheap and recomputing costs 4 seconds.

## Findings

Each finding maps to a later phase. Ordered by impact on the phone, where a missed token costs about 100x more.

1. **Overwriting states without a RAM cache (config F) is the largest loss.** A single slot keeps one prompt. A new request keeps only the common prefix with the slot's tokens, which is a few chat-template tokens, and frees the rest. More free cells in the pool do not help. A cache that keeps shared prefixes (the radix tree) fixes this case directly. Today the only fix is a dedicated slot per prompt (C), which needs the client to know the prompt families in advance.
2. **The RAM cache already gets most of the benefit on the phone (A = C).** On unified memory a swap is a memory copy (85 ms per turn for Llama), so restoring is always worth it. On the laptop, the swap (131 ms per turn) costs more than recomputing a short prompt on the GPU. Restore from RAM should only be used when the prefix is long enough, and Phase 1 should measure that threshold.
3. **A small RAM cache gives an eviction cliff (config D).** For Llama, the stage 3 state does not fit in D's cache at all (about 7,800 tokens x 32 KiB = 250 MB against 128 MiB on the laptop, and about 1,700 tokens x 32 KiB = 54 MB against 32 MiB on the phone). That is a capacity limit, not BUG-003. Prefix-only entries (store the shared prefix once, not every full prompt) would make those entries fit.
4. **The background task never reuses its prompt (BUG-001).** See [BUG-001](#bug-001-reproduced). The concurrent task also slows the foreground: on the phone, Llama's stage 1 TTFT goes from 1.0 s to 5.2 s, and its turns take 32% longer.
5. **Models that cannot roll back lose reuse in every configuration.** Gemma's 512-token sliding window and LFM2's recurrent state can only roll back to saved checkpoints. The server creates checkpoints near the end of each prompt, which is past the part the next prompt shares. It then erases them as invalid (for Gemma, `erased invalidated context checkpoint` appears 44-88 times per run). With the `turns` layout, LFM2's checkpoints fall at user-message starts, and stage 3 reaches the ideal. Checkpoints at stage boundaries (end of the system prompt, end of the shared history) would fix this for any layout.
6. **No slot purges.** The KV pool never filled up, so `try_clear_idle_slots()` never ran. Testing it needs a smaller `-c`.

### Open questions

- **Gemma, config C on the phone:** stage 3 reaches 51% there but 0% on the laptop. My guess is that the shorter `edge` prompts let the dedicated slot's checkpoints fall inside the shared prefix. This is not verified.
- **Laptop energy:** NVML gives GPU board power only. A whole-laptop number would need RAPL (CPU package power) as well.
- **Turns layout on the phone:** the phone runs used the `inline` layout, so LFM2's stage 3 on the phone is 18% instead of the ideal 54%.

## Bug status

[BUGS.md](../../BUGS.md) lists the bugs found by reading the server code. Phase 0 checked two of them with real runs.

### BUG-001 (reproduced)

In config E, the background task is pinned to slot 1. Idle-slot caching saves slot 1 to RAM and clears it while the foreground stages run. When the next background request arrives, the pinned request finds an empty slot and does not look in the RAM cache. Its hit rate is 0.0% on every device and model, while 40-52% is possible. The server log shows 13-15 "skipping, slot is empty" lines per laptop run in config E, and 1 in config A.

### BUG-003 (not confirmed)

In config D with LFM2 and Gemma, the logs show 12-23 RAM cache evictions per run, and stage 1 and 2 hit rates fall by 3-35 points (most for LFM2 on the phone). The evictions happen while a stage 3 state is saved, before the next state is loaded, which is the order BUG-003 describes. After them, some loads find only a poor match (`f_keep` 0.27-0.32, against 0.83-0.84 for normal loads). The logs do not show which entry was removed, so I cannot tell whether it was the entry about to be loaded. For Llama, D's losses come from the capacity limit in finding 3, not from BUG-003. A log line with the removed entry's token count would settle it.

## Phase 0 checklist

| Item | Status |
| --- | --- |
| Laptop and phone builds | done |
| Benchmark and offline tests (29 tests, pyright, flake8) | done |
| Prefill ceilings | done |
| Laptop matrix (A-F x 3 models x 3 reps, plus LFM2 `turns`) | done |
| Phone matrix (A-F x 3 models x 3 reps) | done |
| BUG-001 | reproduced |
| BUG-003 | open |
