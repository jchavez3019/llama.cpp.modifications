# Phase 0 plan: baseline

Phase 0 measures how well the stock `llama-server` caches a cyclic 3-stage agent workload on a laptop GPU and on a phone CPU. It changes no server code. Status: done. Results are in [phase_0_baseline_results.md](phase_0_baseline_results.md), and the full report with setup and run commands is [../phase_0_baseline.md](../phase_0_baseline.md).

## Goal

Get numbers that decide which later phases are worth doing, and a benchmark that every later phase reuses.

- Measure hit rate per stage, time to first token, turn time, and energy for six server and client configurations (A-F).
- Compare against an ideal prefix cache with no size limit.
- Reproduce the bugs found by reading the code (BUG-001 to BUG-004 in [BUGS.md](../../../BUGS.md)).

## Work items

| Item | Output |
| --- | --- |
| Builds | `CMakeUserPresets.json` presets for the laptop (CUDA sm_86) and the Pixel Fold (NDK 26.3, static); see [../build.md](../build.md) |
| Models | Q4_K_M GGUFs of Llama 3.2 1B (attention), Gemma 3 1B (SWA), and LFM2 1.2B (hybrid), on both devices |
| Environment | Conda env `llama-orch-bench` (`tools/server/bench/orchestration-bench/environment.yml`) |
| Benchmark | `tools/server/bench/orchestration-bench/`: workload generator, runner, matrix driver, telemetry (NVML, Perfetto), summary |
| Tests | Offline BDD tests for the harness (29 tests), pyright and flake8 clean |
| Ceilings | `llama-bench` prefill and generation rates on both devices |
| Matrices | Laptop: A-F x 3 models x 3 reps, plus LFM2 `turns` layout. Phone: A-F x 3 models x 3 reps |
| Report | [../phase_0_baseline.md](../phase_0_baseline.md) |

## Configurations

| ID | Server | Client |
| --- | --- | --- |
| A | `-np 2`, RAM cache on | All stages pinned to slot 0 |
| B | Same as A | Unpinned |
| C | `-np 4 --cache-ram 0` | One slot per stage |
| D | A with a small RAM cache | All stages on slot 0 |
| E | Same as A | Stages on slot 0, background task on slot 1, concurrent |
| F | `-np 2 --cache-ram 0` | All stages pinned to slot 0 |

## Exit criteria

All met.

- Every run completes with no request errors.
- Each finding is tied to a later phase.
- The benchmark can be rerun unchanged by later phases.
