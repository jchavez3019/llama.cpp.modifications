import io
import json

import pytest

from runner import (
    NO_SLOT,
    PIN_POLICIES,
    ChatCompletionBody,
    PinMode,
    RequestRecord,
    RunConfig,
    ServerClient,
    ServerTimings,
    _sse_chunks,
    ideal_reuse,
    planned_requests,
    run,
)
from workload import PRESETS, ChatMessage, Layout, Request, Stage, Workload


def word_count(text: str) -> int:
    """Stand-in tokenizer: one token per word."""
    return len(text.split())


class FakeClient(ServerClient):
    """Stand-in for the server: prompts are tokenized as word hashes, and completions return fixed timings without HTTP."""

    def __init__(self) -> None:
        super().__init__("http://fake")
        self.sent: list[tuple[Stage, int]] = []

    def prompt_tokens(self, messages: list[ChatMessage]) -> list[int]:
        return [hash(w) for m in messages for w in (m["role"] + " " + m["content"]).split()]

    def chat_stream(self, request: Request, id_slot: int, record: RequestRecord) -> None:
        self.sent.append((request.stage, id_slot))
        record.apply_timings(ServerTimings(cache_n=10, prompt_n=5, prompt_ms=1.0, predicted_n=2, predicted_ms=2.0))
        record.ttft_ms, record.e2e_ms = 1.0, 2.0


def edge_workload() -> Workload:
    return Workload(PRESETS["edge"], 1, Layout.INLINE, word_count)


def test_ideal_reuse_takes_longest_prefix_over_all_earlier_prompts() -> None:
    """
    Given: three prompts, where the third shares 3 tokens with the first and 1 with the second.
    When: the ideal reuse is computed.
    Then: the first prompt gets 0, and the third gets the best earlier match (3), not only the previous prompt.
    """
    assert ideal_reuse([[1, 2, 3, 4], [1, 9], [1, 2, 3, 7]]) == [0, 1, 3]


def test_planned_requests_put_background_before_stage1_every_n_turns() -> None:
    """
    Given: a run with a background task every 2 turns.
    When: the request plan of 4 turns is built.
    Then: turns 0 and 2 start with the background task, and turns 1 and 3 have only the three stages.
    """
    cfg = RunConfig(n_turns=4, n_warmup=0, pin=PinMode.NONE, bg_every=2, bg_slot=1, bg_concurrent=False)
    plan = [(r.turn, r.stage) for r in planned_requests(edge_workload(), cfg)]
    assert plan.count((0, Stage.BACKGROUND)) == 1 and plan.count((2, Stage.BACKGROUND)) == 1
    assert plan[0] == (0, Stage.BACKGROUND) and plan[1] == (0, Stage.ROUTER)
    assert len(plan) == 4 * 3 + 2


def test_pin_policies_map_stages_to_slots_and_reject_background() -> None:
    """
    Given: the three pinning modes.
    When: the slot of each foreground stage is looked up, and the slot of the background task is asked for.
    Then: slot0 sends all stages to slot 0, per-stage uses slots 0, 1, 2, none lets the server choose, and the background task raises (its slot comes from the run config).
    """
    stages = (Stage.ROUTER, Stage.EXTRACTOR, Stage.SYNTHESIS)
    assert [PIN_POLICIES[PinMode.SLOT0].slot_for(s) for s in stages] == [0, 0, 0]
    assert [PIN_POLICIES[PinMode.PER_STAGE].slot_for(s) for s in stages] == [0, 1, 2]
    assert [PIN_POLICIES[PinMode.NONE].slot_for(s) for s in stages] == [NO_SLOT] * 3
    with pytest.raises(ValueError):
        PIN_POLICIES[PinMode.SLOT0].slot_for(Stage.BACKGROUND)


def test_chat_body_omits_id_slot_only_when_server_should_choose() -> None:
    """
    Given: one body pinned to slot 1 and one without a slot.
    When: both are converted to JSON.
    Then: the pinned body has id_slot 1, the other has no id_slot key, and both stream greedily with a fixed length.
    """
    msgs: list[ChatMessage] = [{"role": "user", "content": "hi"}]
    pinned = ChatCompletionBody(msgs, 8, 1).to_json()
    free = ChatCompletionBody(msgs, 8).to_json()
    assert pinned["id_slot"] == 1
    assert "id_slot" not in free
    assert free["stream"] is True and free["temperature"] == 0.0 and free["ignore_eos"] is True and free["max_tokens"] == 8


def test_sse_parser_skips_non_data_lines_decodes_bytes_and_stops_at_done() -> None:
    """
    Given: a stream with a blank line, a comment line, a text chunk, a bytes chunk, the done marker, and a chunk after it.
    When: the stream is parsed.
    Then: only the two chunks before the done marker are returned.
    """
    lines: list[str | bytes] = ["", ": ping", 'data: {"a": 1}', b'data: {"b": 2}', "data: [DONE]", 'data: {"c": 3}']
    assert list(_sse_chunks(iter(lines))) == [{"a": 1}, {"b": 2}]


def test_server_timings_parse_known_fields_and_default_missing_ones() -> None:
    """
    Given: a timings object with cache_n, a non-numeric prompt_n, and no generation fields.
    When: it is parsed.
    Then: cache_n is kept and the other fields become 0, so a partial chunk never raises.
    """
    t = ServerTimings.from_json({"cache_n": 42, "prompt_n": "x"})
    assert t == ServerTimings(cache_n=42, prompt_n=0, prompt_ms=0.0, predicted_n=0, predicted_ms=0.0)


def test_request_record_round_trips_and_ignores_unknown_keys() -> None:
    """
    Given: a record written as JSON plus an unknown key, and an older line without optional fields.
    When: both are loaded with from_json.
    Then: the first equals the original, and the second gets the defaults for the missing fields.
    """
    rec = RequestRecord(turn=1, stage=Stage.ROUTER, warmup=False, id_slot=0, t_start=5.0, cache_n=3, prompt_tokens=4)
    loaded = RequestRecord.from_json({**json.loads(json.dumps(rec.__dict__)), "future_field": 1})
    assert loaded == rec
    old = RequestRecord.from_json({"turn": 0, "stage": "stage3", "warmup": True, "id_slot": -1, "t_start": 1.0})
    assert old.ttft_ms == -1.0 and old.error == ""


def test_run_pins_stages_per_mode_and_marks_warmup_turns() -> None:
    """
    Given: a per-stage pinning run of 3 turns with 1 warm-up turn and a sequential background task on slot 3 every 2 turns.
    When: the run executes against a fake client.
    Then: stages go to slots 0, 1, 2, the background task goes to slot 3, only turn 0 is marked warm-up, and every record is written as a JSON line with its ideal reuse.
    """
    cfg = RunConfig(n_turns=3, n_warmup=1, pin=PinMode.PER_STAGE, bg_every=2, bg_slot=3, bg_concurrent=False)
    client = FakeClient()
    out = io.StringIO()
    records = run(client, edge_workload(), cfg, out)

    assert client.sent[:4] == [(Stage.BACKGROUND, 3), (Stage.ROUTER, 0), (Stage.EXTRACTOR, 1), (Stage.SYNTHESIS, 2)]
    assert {r.turn for r in records if r.warmup} == {0}
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(lines) == len(records) == 3 * 3 + 2
    assert lines[0]["stage"] == "background"

    # Stage 1 of turn 1 shares its system prompt with stage 1 of turn 0, so its ideal reuse is above 0.
    s1_t1 = next(r for r in records if r.turn == 1 and r.stage == Stage.ROUTER)
    assert 0 < s1_t1.ideal_cached < s1_t1.client_prompt_tokens


def test_concurrent_background_task_is_recorded_once_per_scheduled_turn() -> None:
    """
    Given: a run of 4 turns with a concurrent background task every 2 turns.
    When: the run executes against a fake client.
    Then: exactly two background records exist (turns 0 and 2), so the per-turn join neither drops nor duplicates tasks.
    """
    cfg = RunConfig(n_turns=4, n_warmup=0, pin=PinMode.SLOT0, bg_every=2, bg_slot=1, bg_concurrent=True)
    records = run(FakeClient(), edge_workload(), cfg, io.StringIO())
    assert sorted(r.turn for r in records if r.stage == Stage.BACKGROUND) == [0, 2]
