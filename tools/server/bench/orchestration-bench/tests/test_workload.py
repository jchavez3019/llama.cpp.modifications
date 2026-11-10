from workload import PRESETS, Layout, Stage, Workload, calibrated_text


def fake_count(text: str) -> int:
    """Stand-in tokenizer: one token per word, plus one more for every word longer than 7 characters.

    It only makes the word-to-token ratio uneven, like a real tokenizer, so calibration has work to do.
    """
    return sum(1 + (len(w) > 7) for w in text.split())


def make_workload(layout: Layout = Layout.INLINE) -> Workload:
    return Workload(PRESETS["edge"], seed=7, layout=layout, count_tokens=fake_count)


def test_calibrated_text_reaches_exact_token_count_with_uneven_tokenizer() -> None:
    """
    Given: a tokenizer where some words cost two tokens.
    When: text is calibrated to several target sizes.
    Then: each text has exactly the target token count.
    """
    for target in (1, 17, 300, 1000):
        assert fake_count(calibrated_text(1, "seg", target, fake_count)) == target


def test_calibrated_text_is_empty_for_zero_tokens() -> None:
    """
    Given: a target of zero tokens.
    When: text is calibrated.
    Then: the text is empty and the tokenizer is never called.
    """
    def never_called(text: str) -> int:
        """Stand-in tokenizer that fails the test if calibration calls it."""
        raise AssertionError("tokenizer called")

    assert calibrated_text(1, "seg", 0, never_called) == ""


def test_calibrated_text_is_reproducible_and_differs_between_segments() -> None:
    """
    Given: a fixed seed.
    When: the same segment is built twice, and a second segment is built once.
    Then: the same name gives the same text, and a different name gives different text.
    """
    a1 = calibrated_text(3, "rag/0", 200, fake_count)
    a2 = calibrated_text(3, "rag/0", 200, fake_count)
    b = calibrated_text(3, "rag/1", 200, fake_count)
    assert a1 == a2
    assert a1 != b


def test_inline_history_of_a_turn_is_a_prefix_of_the_next_turn() -> None:
    """
    Given: an inline-layout workload.
    When: the history text of consecutive turns is built.
    Then: each history is a strict prefix of the next, so a prefix cache can reuse it (append-only history).
    """
    wl = make_workload()
    histories = [wl.history_text(t) for t in range(5)]
    assert histories[0] == ""
    for prev, nxt in zip(histories, histories[1:]):
        assert nxt.startswith(prev) and len(nxt) > len(prev)


def test_stage_system_prompts_stay_static_while_user_message_changes() -> None:
    """
    Given: a workload.
    When: the stage requests of two turns are built.
    Then: every stage keeps the same system prompt, the dynamic user message of stage 1 differs between turns, and each stage gets its preset generation length.
    """
    wl = make_workload()
    t0, t1 = wl.stage_requests(0), wl.stage_requests(1)
    assert [r.stage for r in t0] == [Stage.ROUTER, Stage.EXTRACTOR, Stage.SYNTHESIS]
    for a, b in zip(t0, t1):
        assert a.messages[0] == b.messages[0]
    assert t0[0].messages[1] != t1[0].messages[1]
    assert [r.max_tokens for r in t0] == [PRESETS["edge"].gen_router, PRESETS["edge"].gen_extractor, PRESETS["edge"].gen_synthesis]


def test_shared_preset_starts_every_system_prompt_with_the_same_part() -> None:
    """
    Given: the edge-shared preset and the edge preset with the same seed.
    When: the stage and background requests of a turn are built.
    Then: every system prompt starts with the same shared text of sys_shared tokens, ends with the edge preset's own system prompt, and the edge preset has no shared part.
    """
    shared = Workload(PRESETS["edge-shared"], seed=7, layout=Layout.INLINE, count_tokens=fake_count)
    plain = make_workload()
    reqs_shared = shared.stage_requests(0) + [shared.background_request(0)]
    reqs_plain = plain.stage_requests(0) + [plain.background_request(0)]

    prefix = calibrated_text(7, "sys/shared", PRESETS["edge-shared"].sys_shared, fake_count) + "\n"
    assert fake_count(prefix) == PRESETS["edge-shared"].sys_shared
    for a, b in zip(reqs_shared, reqs_plain):
        assert a.messages[0]["content"] == prefix + b.messages[0]["content"]
        assert not b.messages[0]["content"].startswith(prefix)


def test_turns_layout_sends_previous_turns_as_chat_messages() -> None:
    """
    Given: a turns-layout workload.
    When: the stage 3 request of turn 3 is built.
    Then: it holds the system prompt, 3 user/assistant pairs, and the final user message, and earlier pairs match the stored queries and answers.
    """
    wl = make_workload(Layout.TURNS)
    msgs = wl.stage_requests(3)[2].messages
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert msgs[1]["content"] == wl.query(0)
    assert msgs[2]["content"] == wl.answer(0)
