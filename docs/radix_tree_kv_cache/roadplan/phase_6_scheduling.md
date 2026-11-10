# Phase 6 plan: scheduling

Phase 6 keeps background work from delaying the foreground stages. In Phase 0, the concurrent background task raised Llama's stage 1 time to first token on the phone from 1.0 s to 5.2 s. It does not depend on the tree and can start any time after Phase 1. Results go in [phase_6_scheduling_results.md](phase_6_scheduling_results.md).

## Goal

- Config E on the phone: stage 1 TTFT within 20% of config A (1.0 s), while the background task still finishes.

## Work items

### 1. Request priority

- Request field `priority` (for example `high`, `normal`, `low`; default `normal`).
- When building a batch in `update_slots()`, add decode tokens of high-priority slots first, then prefill of high-priority slots, then the rest, within `n_batch`.

### 2. Prefill chunking for low priority

- A low-priority prefill is added in chunks of at most `--prefill-chunk-low N` tokens per batch (for example 128), so one batch never carries a long background prefill ahead of a foreground request.
- High-priority requests keep today's behavior.

### 3. Lane reservation

- `--slots-reserved N` keeps N slots for high-priority requests; low-priority requests wait instead of taking them.
- With the tree (Phase 3), slots are stateless, so reservation only affects concurrency, not hits.

### 4. Telemetry

Log queue wait time per request, so TTFT can be split into waiting and prefill.

## Measurements

Config E on both devices, Llama, 3 reps: priority on and off, chunk sizes 64, 128, 256. Report stage 1 TTFT, turn time, and background completion time.

## Exit criteria

- Phone stage 1 TTFT in E within 20% of A.
- Background task completion time at most 2x its Phase 0 value.

## Risks

- **Small batches are slower per token.** Chunking the background prefill lowers its throughput; the chunk size trades foreground latency against background time.
- **Server scope.** Check `tools/server/README-dev.md` before adding request fields, so the change fits the server's defined scope.
