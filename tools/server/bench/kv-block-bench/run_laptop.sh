#!/usr/bin/env bash
# Run llama-kv-block-bench for each model on the laptop GPU.
# The GPU throttles when hot, so each model waits for a cool GPU first, and the GPU state and a llama-bench pp512 rate are saved next to its results.
#
# usage: run_laptop.sh OUT_DIR MODEL.gguf [MODEL.gguf ...]

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 OUT_DIR MODEL.gguf [MODEL.gguf ...]" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
BIN="$ROOT/build-laptop-cuda-release/bin"
OUT="$1"
shift

MAX_START_TEMP_C="${MAX_START_TEMP_C:-60}"
MAX_WAIT_S="${MAX_WAIT_S:-1200}"

gpu_state() {
    nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader
}

wait_cool() {
    local waited=0
    while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -gt "$MAX_START_TEMP_C" ]; do
        if [ "$waited" -ge "$MAX_WAIT_S" ]; then
            echo "warning: GPU still above ${MAX_START_TEMP_C} C after ${MAX_WAIT_S} s, starting anyway" >&2
            return
        fi
        sleep 15
        waited=$((waited + 15))
    done
}

mkdir -p "$OUT"

for model in "$@"; do
    name="$(basename "$model" .gguf)"
    name="${name%-Q4_K_M}"

    wait_cool
    {
        echo "before: $(gpu_state)"
        "$BIN/llama-bench" -m "$model" -p 512 -n 0 -fa 1 -r 2 -o csv 2>/dev/null | tail -n 1
    } > "$OUT/$name.gpu.txt"

    "$BIN/llama-kv-block-bench" -m "$model" -o "$OUT/$name.jsonl" --device laptop -r 5

    echo "after: $(gpu_state)" >> "$OUT/$name.gpu.txt"
done
