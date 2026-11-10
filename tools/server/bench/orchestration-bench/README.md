# Orchestration cache benchmark

Replays a cyclic multi-stage agent workload against `llama-server` and measures how much of each prompt the server reuses from its caches, compared with a perfect prefix cache.

One turn runs three stages with different static system prompts: router, tool extractor, and response synthesis (with a growing chat history and new RAG documents every turn). An optional background task runs every few turns. All text is synthetic and seeded, so every run sends the same prompts.

## Files

| File | Purpose |
| --- | --- |
| `workload.py` | Seeded prompt generator, calibrated to exact token counts with `/tokenize`. Presets `desktop` and `edge`, and `desktop-shared` and `edge-shared` with a system prompt part shared by all requests. Layouts `inline` and `turns`. |
| `runner.py` | Streaming client. Pinning modes, background thread, one JSON line per request, ideal reuse per request. |
| `telemetry.py` | Local sampler (RSS from `/proc`, per-process VRAM, GPU power, utilization, and temperature with NVML via `pynvml`). Phone recorder (a Perfetto trace plus thermal polling with `dumpsys thermalservice`). |
| `phone_trace.py` | Reads power-rail energy, server RSS, and big-core frequency of the measured window from a Perfetto trace. |
| `summary.py` | Per-stage and per-run metrics, Markdown tables over a results tree. |
| `orchestration_bench.py` | One run against a server that is already running. |
| `run_matrix.py` | Starts a fresh server per run (locally or on the phone) and loops over configurations, models, and repetitions. |
| `tests/` | Offline unit tests (no server). |

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate llama-orch-bench
python -m pytest -q tests
```

## Usage

One run against a running server:

```bash
python orchestration_bench.py --url http://127.0.0.1:8080 --preset desktop --pin slot0 --out bench-results/manual/rep0
```

The full matrix on the laptop (run from the repository root):

```bash
python tools/server/bench/orchestration-bench/run_matrix.py --target local \
  --server-bin build-laptop-cuda-release/bin/llama-server --server-extra "-ngl 99" \
  --models ~/models/*.gguf --device laptop
```

On a phone over `adb` (binaries and models in `/data/local/tmp/llama.cpp`):

```bash
python tools/server/bench/orchestration-bench/run_matrix.py --target adb --adb ~/Android/Sdk/platform-tools/adb \
  --models ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --device pixel-fold \
  --preset edge --ctx 4096 --turns 12 --warmup 2 --configs A,B,C,D,E,F --reps 3 \
  --taskset f0 --server-extra "-t 4" --cache-ram-default 1024 --d-cache-ram 32 --port 8090
```

Summary tables:

```bash
python tools/server/bench/orchestration-bench/summary.py bench-results/phase0
```

A run that fails (for example, the server does not start) writes `rep<k>.error.txt` and the matrix goes on. `--max-big-temp` waits for the phone to cool before each run (default: no wait).

## Telemetry

On the laptop, `LocalSampler` samples every 100 ms. On the phone, `PhoneTelemetry` starts `/system/bin/perfetto` in the background with the config in `telemetry.PERFETTO_CONFIG` (power rails, battery counters, process RSS, CPU frequency), stops it with SIGTERM, and pulls `rep<k>.pftrace`. Open a trace in [ui.perfetto.dev](https://ui.perfetto.dev) to see the timeline. Temperatures do not show in the trace on this phone, so a thread polls `dumpsys thermalservice` every 5 s (`rep<k>.thermal.jsonl`).

Trace timestamps are phone BOOTTIME. `run_matrix.py` stores the phone-minus-laptop clock offset in `meta.json`, and `phone_trace.py` uses it plus the trace clock snapshot to map request times onto the trace. The first summary of a trace writes `rep<k>.trace_metrics.json`, so later summaries do not parse the trace again.

## Metrics

- **Hit %:** `cache_n / (cache_n + prompt_n)` from the server `timings`, weighted by tokens.
- **Ideal %:** share of prompt tokens that a perfect prefix cache with no size limit could reuse. It is the longest token prefix shared with any earlier request of the run.
- **Prefill tok:** mean tokens evaluated per request.
- **TTFT:** client time to the first generated token.
- **Turn:** sum of the three stage latencies of one turn.
- **J/turn:** energy of the measured window divided by measured turns. Phone: sum of all `power.rails.*` (CPU, GPU, DDR, display, modem, and so on; not the battery or charger loss). Laptop: GPU board power only (NVML), so the two devices are not comparable.
- **CPU J/turn:** phone only, sum of the `power.rails.cpu.*` rails.
- **Peak BIG C, Big MHz:** phone only, peak big-core temperature (HAL sensor) and mean frequency of cores 4-7.
- **Swap ms/turn:** total "prompt cache update took" time in the server log (saving slot states to RAM and loading them back), divided by all turns.
- **Log counters:** cache events counted in the server log (`-lv 4`), including warm-up turns. `Empty-slot skips` counts slots that slot selection skipped because they were empty. For a pinned request this is BUG-001 in `BUGS.md`. Unpinned selection also logs it for every empty slot it looks at, so for config B it is not a bug signal.
