# Phase 1 plan: measurements and block prototype

Phase 1 measures the costs that every later decision depends on: how long a restore takes against a recompute, on each device and model, and whether block storage works with the existing library calls. It also tests the upstream BUG-001 fixes and adds cache telemetry to the server. Results go in [phase_1_measurements_results.md](phase_1_measurements_results.md).

## Goal

Replace the estimates in [design.md](../design.md) with measured values.

- \(t_\text{prefill}\), \(t_\text{copy}\), and \(t_\text{overhead}\), and from them the restore thresholds \(T_1\) (laptop, RAM to VRAM) and the restore cost on the phone (RAM to RAM).
- Whether a sequence saved as `B`-token blocks restores to the same logits as a cold prefill.
- A first value of `B` per device.
- Whether upstream PRs #30866 and #28992 fix BUG-001 in config E.
- Server log lines that say where each hit came from and how long restores and saves took.

## Work items

### 1. Block benchmark (`llama-kv-block-bench`)

Location: `tools/server/bench/kv-block-bench/`. Fork-only; built by both presets (`llama-kv-block-bench` target).

| Kind | What it times | Calls |
| --- | --- | --- |
| `prefill_cold` | Prefill \(N\) tokens into an empty cache | 1 decode |
| `prefill_offset` | Prefill \(N\) tokens after a prefix (1,500 laptop, 300 phone), the marginal cost of recompute inside a path | 1 decode |
| `whole_save`, `whole_restore` | `llama_state_seq_get_data` and `llama_state_seq_set_data` of \(N\) tokens, today's server path | 1 |
| `block_save`, `block_restore` | The same \(N\) tokens as \(N / B\) blocks | \(N / B\) |
| `check_whole`, `check_block` | Probe-token logits after a restore against a cold prefill: top token and largest difference | - |

How blocks work with today's API:
- **Save one block:** `llama_memory_seq_cp(main, scratch, k*B, (k+1)*B)` tags only that block's cells, `llama_state_seq_get_data(scratch)` saves them, `llama_memory_seq_rm(scratch)` drops the tag.
- **Restore one block:** `llama_state_seq_set_data` clears its destination first, so each block is loaded into `scratch`, then re-tagged with `seq_cp(scratch, main)`, then `seq_rm(scratch)`.

Scope:
- Block path for full-attention models only (Llama). Gemma's SWA cache masks cells relative to the scratch sequence's last position, and LFM2's recurrent state cannot be split by position. Both get prefill and whole-state numbers only.
- Grid: \(N\) in 16, 64, 256, 1024, 4096 on the laptop (16-1024 on the phone, where prefill is slow); `B` in 16, 64, 256; 5 reps on the laptop, 3 on the phone, after one warm-up.
- Context: unified KV, flash attention, `n_ubatch` 512 (the server default), 2 sequences.

### 2. Fit and thresholds (`kv_block_fit.py`)

Fits `prefill = fixed + t_prefill * N` over `prefill_offset`, and `restore = t_overhead * calls + t_copy * N` over whole and block restores together. Reports \(T = t_\text{overhead} / (t_\text{prefill} - t_\text{copy})\), the check results, and the `B` with the lowest expected restore cost over the Phase 0 run lengths (80, 300, 600, 3,000 tokens). Offline BDD tests in `tests/`.

### 3. Upstream BUG-001 fixes

For each of [#30866](https://github.com/ggml-org/llama.cpp/pull/30866) and [#28992](https://github.com/ggml-org/llama.cpp/pull/28992):
1. Fetch the PR into a local branch (`git fetch upstream pull/<n>/head:pr-<n>`), with no commits on top.
2. Build the laptop preset in a separate build directory.
3. Run config E for Llama, 3 reps, and compare the background hit rate and "skipping, slot is empty" count with Phase 0.
4. Record the results. Posting them on the PR is the contributor's own action.

### 4. Cache telemetry in the server

Add one log line per request at the point where the prompt cache is used (`get_available_slot` and `server_prompt_cache::load` in `tools/server/server-task.cpp`):
- source of the reused tokens: slot, RAM cache, or none;
- tokens reused and tokens prefilled;
- restore and save time in ms, and bytes moved.

Extend `summary.py` to read these lines from `server.log` and report restore time per turn.

## Exit criteria

- Both devices, all three models: a fit with \(t_\text{prefill}\), \(t_\text{overhead}\), \(t_\text{copy}\), and \(T\).
- Every block check passes (same top token) for Llama, or the failure is understood.
- A chosen `B` for the laptop and for the phone, with the reason.
- BUG-001 PR results recorded.
- Telemetry merged in the fork and used by one rerun of configs A and E.

## Risks

- **Per-call overhead dominates block restores.** Then "load many blocks in one call" becomes the one library change, justified with these numbers.
- **The phone is throttled.** Phase 0 phone times are throttled values. Phase 1 records them the same way, so they are comparable, not peak values.
- **Random tokens.** The benchmark uses seeded random token ids. Prefill cost does not depend on content, and the checks compare like with like.
