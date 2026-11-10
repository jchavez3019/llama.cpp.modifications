# Known Bugs

This file lists bugs found in this fork. Each entry gives the evidence, how to reproduce it, and a proposed fix. "Found by reading" means the bug was found in the code and was not yet reproduced.

Terms used below:

- **Pinned request / pinned slot:** a request that sets `id_slot` to a slot index (for example `"id_slot": 0`). The server must run it on that slot. If the slot is busy, the task waits. A request without `id_slot` (or with `-1`) is unpinned, and the server selects the slot.
- **RAM prompt cache:** `server_prompt_cache` (`--cache-ram`). It holds copies of slot KV states in host memory.
- **Idle-slot caching:** `--cache-idle-slots` (on by default). When a task is launched, every other idle slot is saved to the RAM prompt cache. With a unified KV cache, those slots are then cleared.

## Summary

| ID | Title | Severity | Status |
| --- | --- | --- | --- |
| BUG-001 | Pinned request on an empty slot never restores from the RAM prompt cache | major | found by reading |
| BUG-002 | Pinned request with `--slot-prompt-similarity 0` never uses the RAM prompt cache | minor | found by reading |
| BUG-003 | Saving the outgoing slot state can evict the entry that is about to be loaded | minor | found by reading |
| BUG-004 | RAM cache entries are not matched by LoRA set | major | suspected |

---

## BUG-001: Pinned request on an empty slot never restores from the RAM prompt cache

A pinned request on a slot with no cached tokens skips the RAM prompt cache, so it recomputes its whole prompt, even when the RAM cache holds a matching state.

**Where:** `get_available_slot()`, `tools/server/server-context.cpp` (around lines 1800-1906).

**Cause:**

1. The request pins slot 0, so `ret` is set to slot 0 before the similarity search.
2. The similarity search skips empty slots, so it does not change `ret` and `f_sim_best` stays `0`.
3. Because `ret != nullptr`, this line runs with an empty slot:

   ```cpp
   const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();
   ```

   This is `0.0f / 0`, which gives NaN. `NaN < 0.5f` is false, so `update_cache` stays `false`.
4. The LRU branch runs only when `ret == nullptr`, so it is skipped too.
5. `prompt_load()` is never called. The slot stays empty, and `update_slots()` evaluates the full prompt.

**When it happens:** any time a pinned slot is empty when its request arrives. With the default settings (`--cache-idle-slots` and a unified KV cache), this is common:

1. Request A runs on slot 0 (pinned).
2. Request B runs on slot 1. Idle-slot caching saves slot 0 to RAM and clears it.
3. Request A' (same prefix as A) is pinned to slot 0. Slot 0 is empty, so A' recomputes everything. The state saved in step 2 is never used by slot 0.

Unpinned requests are not affected: the similarity search skips the empty slot, the LRU branch selects a free slot and sets `update_cache = true`, and `prompt_load()` restores the best match. The existing test `test_clear_and_restore` (`tools/server/tests/unit/test_kv_keep_only_active.py`) sends the follow-up request without `id_slot`, so it only covers the unpinned path.

**Impact:** orchestration clients that pin stages to slots lose the cache of a slot every time another slot launches a task. With `-ffast-math`, the NaN comparison is undefined behavior.

**Reproduce (proposed test, not added yet):**

```python
def test_pinned_slot_restores_from_cache_ram_after_idle_clear():
    """
    Given: a server with 2 slots, a unified KV cache, --cache-ram enabled, and idle-slot caching on.
    When:  a long prompt runs pinned to slot 0, a short prompt runs pinned to slot 1 (which clears slot 0), and the long prompt is sent again pinned to slot 0.
    Then:  the third request restores the cached state, so timings.cache_n > 0 and timings.prompt_n is smaller than in the first request.
    """
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "id_slot": 0, "cache_prompt": True})
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completion", data={"prompt": "The quick brown fox", "id_slot": 1, "cache_prompt": True})
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={"prompt": LONG_PROMPT, "id_slot": 0, "cache_prompt": True})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n
```

This test fits in `test_kv_keep_only_active.py`, which already has the matching fixture.

**Proposed fix:** treat an empty selected slot as "keeps nothing", so the cache is consulted:

```cpp
const float f_keep = ret->prompt.tokens.empty() ? 0.0f : (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();
```

`prompt_save()` returns early for an empty slot, so the only effect is that `prompt_load()` runs.

---

## BUG-002: Pinned request with `--slot-prompt-similarity 0` never uses the RAM prompt cache

**Where:** `get_available_slot()`, same function as BUG-001.

**Cause:** with similarity disabled, the similarity block is skipped. For a pinned request, `ret` is already set, so the LRU branch is skipped too. `update_cache` stays `false` in every case, so the slot state is never saved to RAM on a switch and nothing is ever restored from RAM. This applies to empty and non-empty pinned slots.

**Impact:** with `--slot-prompt-similarity 0`, a pinned slot that alternates between different prompts (for example, orchestration stages 1, 2, 3 on slot 0) recomputes every prompt.

**Proposed fix:** for a pinned request, compute the longest common prefix of the slot directly when the similarity block is skipped, and set `update_cache` with the same `f_keep < 0.5` rule.

---

## BUG-003: Saving the outgoing slot state can evict the entry that is about to be loaded

**Where:** `get_available_slot()` calls `prompt_save()` before `prompt_load()`. `server_prompt_cache::alloc()` makes room by removing the oldest entries (`tools/server/server-task.cpp`, around lines 1795-1803).

**Cause:** during the swap, the RAM cache briefly holds the outgoing state and the target entry together. The target entry is about to leave RAM, but it still counts against `--cache-ram`. If the limit is tight, `alloc()` can remove the target entry.

**Example:** 3 prompts used in a cycle (1, 2, 3, 1, ...) on one slot, with `--cache-ram` large enough for 2 entries. Before prompt 1 runs, the slot holds state 3 and RAM holds `[E1, E2]`. Saving state 3 removes the oldest entry, `E1`, which is the entry prompt 1 needs. Every prompt misses, although after the swap only 2 entries must stay in RAM.

**Impact:** a hit rate near 0% for cyclic workloads with a small `--cache-ram`. With the default 8 GiB, this is rare.

**Proposed fix:** find the best entry for the new prompt before the save, then use one of these rules when `alloc()` makes room:

- **Strict:** never remove the target entry, and remove other entries instead. RAM use never exceeds `--cache-ram`, but another entry is lost. In the example, `E2` is removed, so prompt 1 hits and prompt 2 misses.
- **Lenient:** do not count the size of the target entry, because `load()` frees it some milliseconds later. RAM use can exceed `--cache-ram` by the size of the target entry during the swap, and every prompt in the example hits. The device must have that much free RAM. If the allocation fails, `alloc()` already catches `std::bad_alloc` and lowers the limit.

With prefix-only entries that stay in RAM on restore, a static prefix is already in the cache, so `alloc()` skips the save and this case does not occur for static prompts.

---

## BUG-004 (suspected): RAM cache entries are not matched by LoRA set

**Where:** `get_available_slot()` restores a RAM entry before `launch_slot_with_task()` compares the LoRA set of the request with `slot.lora`.

**Cause:** `server_prompt_cache_state` stores tokens and checkpoints, but not the LoRA set that produced the KV values. `launch_slot_with_task()` compares the request LoRA set with the previous LoRA set of the *slot*, not with the LoRA set of the restored entry.

**Possible impact:** a state computed with one adapter set is restored for a request that uses another, and the request then reuses KV values from the wrong adapter.

**Next step:** confirm with a test that runs the same prompt with two LoRA scales across a slot switch and compares the outputs with and without `--cache-ram 0`.
