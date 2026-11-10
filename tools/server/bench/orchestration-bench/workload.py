"""Deterministic multi-stage orchestration workload.

One user turn runs three stages with different static system prompts (router, tool extractor, response synthesis). Stage 3 also carries a chat history that grows by one entry every turn, plus per-turn RAG documents and metadata. An optional background task runs every few turns.

All text is synthetic. It is built from a fixed word list with a seeded random generator, so the same seed gives the same prompts on every run, for every server configuration. Segment lengths are calibrated to exact token counts with the server's ``/tokenize`` endpoint.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Literal, TypedDict

Role = Literal["system", "user", "assistant"]
TokenCounter = Callable[[str], int]


class ChatMessage(TypedDict):
    """One OpenAI-style chat message, as sent in the ``messages`` array."""

    role: Role
    content: str


class Stage(StrEnum):
    """Request kinds of the workload. The values are stored in the JSONL records."""

    ROUTER = "stage1"
    EXTRACTOR = "stage2"
    SYNTHESIS = "stage3"
    BACKGROUND = "background"


class Layout(StrEnum):
    """How stage 3 sends the chat history.

    ``INLINE`` puts the history as text inside one user message. ``TURNS`` sends it as alternating user and assistant messages, so the server's checkpoints at user-message starts can land at the end of the history.
    """

    INLINE = "inline"
    TURNS = "turns"


FOREGROUND_STAGES: tuple[Stage, ...] = (Stage.ROUTER, Stage.EXTRACTOR, Stage.SYNTHESIS)

_MAX_CALIBRATION_ROUNDS = 64

_WORDS: tuple[str, ...] = (
    "system", "router", "timer", "weather", "alarm", "message", "contact", "calendar", "reminder", "music",
    "volume", "device", "battery", "network", "setting", "location", "distance", "route", "traffic", "minute",
    "hour", "today", "tomorrow", "morning", "evening", "please", "check", "create", "cancel", "update",
    "schedule", "notify", "search", "result", "document", "summary", "answer", "question", "request", "response",
    "value", "argument", "schema", "field", "string", "number", "boolean", "object", "array", "optional",
    "required", "example", "format", "output", "input", "context", "history", "user", "assistant", "tool",
    "call", "name", "description", "parameter", "default", "range", "limit", "error", "status", "success",
    "light", "kitchen", "office", "garage", "window", "door", "lock", "camera", "sensor", "temperature",
    "humidity", "forecast", "rain", "sunny", "cloudy", "wind", "storm", "travel", "flight", "hotel",
    "booking", "payment", "account", "balance", "invoice", "order", "delivery", "package", "tracking", "store",
)


@dataclass(frozen=True)
class Preset:
    """Token budget of every workload segment.

    :param sys_router: Tokens of the stage 1 system prompt.
    :param sys_extractor: Tokens of the stage 2 system prompt.
    :param sys_synthesis: Tokens of the stage 3 system prompt.
    :param rag: Tokens of the RAG documents, new every turn.
    :param meta: Tokens of the metadata (time, location), new every turn.
    :param query: Tokens of the user query, new every turn.
    :param history_answer: Tokens of the synthetic assistant answer appended to the history every turn.
    :param sys_background: Tokens of the background task system prompt.
    :param background_input: Tokens of the background task input (notifications), new every run.
    :param gen_router: Generated tokens of stage 1.
    :param gen_extractor: Generated tokens of stage 2.
    :param gen_synthesis: Generated tokens of stage 3.
    :param gen_background: Generated tokens of the background task.
    :param sys_shared: Tokens of a system prompt part that all requests share at the start (for example, common tool definitions). Zero for none.
    """

    sys_router: int
    sys_extractor: int
    sys_synthesis: int
    rag: int
    meta: int
    query: int
    history_answer: int
    sys_background: int
    background_input: int
    gen_router: int
    gen_extractor: int
    gen_synthesis: int
    gen_background: int
    sys_shared: int = 0

    def max_tokens(self, stage: Stage) -> int:
        """Returns the number of tokens to generate for a stage."""
        return {
            Stage.ROUTER: self.gen_router,
            Stage.EXTRACTOR: self.gen_extractor,
            Stage.SYNTHESIS: self.gen_synthesis,
            Stage.BACKGROUND: self.gen_background,
        }[stage]


PRESETS: dict[str, Preset] = {
    "desktop": Preset(
        sys_router=500, sys_extractor=700, sys_synthesis=500, rag=3000, meta=50, query=30, history_answer=300,
        sys_background=400, background_input=600,
        gen_router=16, gen_extractor=48, gen_synthesis=128, gen_background=64,
    ),
    "desktop-shared": Preset(
        sys_router=500, sys_extractor=700, sys_synthesis=500, rag=3000, meta=50, query=30, history_answer=300,
        sys_background=400, background_input=600,
        gen_router=16, gen_extractor=48, gen_synthesis=128, gen_background=64,
        sys_shared=1024,
    ),
    "edge": Preset(
        sys_router=300, sys_extractor=400, sys_synthesis=300, rag=600, meta=30, query=20, history_answer=80,
        sys_background=250, background_input=250,
        gen_router=8, gen_extractor=24, gen_synthesis=48, gen_background=32,
    ),
    "edge-shared": Preset(
        sys_router=300, sys_extractor=400, sys_synthesis=300, rag=600, meta=30, query=20, history_answer=80,
        sys_background=250, background_input=250,
        gen_router=8, gen_extractor=24, gen_synthesis=48, gen_background=32,
        sys_shared=768,
    ),
}


@dataclass(frozen=True)
class Request:
    """One chat completion request of the workload.

    :param turn: Turn index, starting at 0.
    :param stage: Request kind.
    :param messages: OpenAI-style chat messages.
    :param max_tokens: Number of tokens to generate.
    """

    turn: int
    stage: Stage
    messages: list[ChatMessage]
    max_tokens: int


def _stable_seed(seed: int, name: str) -> int:
    """Derives a reproducible sub-seed for one named segment.

    Python's ``hash()`` is randomized per process, so a CRC32 of the name is used instead.

    :param seed: Global workload seed.
    :param name: Segment name, for example ``rag/7``.
    :return: Sub-seed for ``random.Random``.
    """
    return (seed * 1_000_003) ^ zlib.crc32(name.encode("utf-8"))


def _words(seed: int, name: str, n_words: int) -> list[str]:
    """Returns ``n_words`` seeded words for one segment.

    :param seed: Global workload seed.
    :param name: Segment name.
    :param n_words: Number of words.
    :return: Word list. The same arguments always give the same list.
    """
    rng = random.Random(_stable_seed(seed, name))
    return [rng.choice(_WORDS) for _ in range(n_words)]


def calibrated_text(seed: int, name: str, n_tokens: int, count_tokens: TokenCounter) -> str:
    """Builds seeded text whose token count is as close as possible to ``n_tokens``.

    The text is a prefix of a long seeded word sequence. The number of words is adjusted until the tokenizer reports the target count, or until the closest count is found.

    :param seed: Global workload seed.
    :param name: Segment name. Different names give different text.
    :param n_tokens: Target token count. Zero gives an empty string.
    :param count_tokens: Function that returns the token count of a text (no special tokens).
    :return: The calibrated text.
    """
    if n_tokens <= 0:
        return ""

    # Every word is at least one token, so n_tokens words always reach the target.
    pool = _words(seed, name, n_tokens)

    # Search the number of words: start with all of them and move by the remaining token difference.
    n_words = n_tokens
    best_text = ""
    best_diff = n_tokens
    for _ in range(_MAX_CALIBRATION_ROUNDS):
        text = " ".join(pool[:n_words])
        diff = count_tokens(text) - n_tokens
        if abs(diff) < best_diff:
            best_text, best_diff = text, abs(diff)
        if diff == 0:
            break

        # Step by half the difference (at least one word) to converge without oscillating.
        step = max(1, abs(diff) // 2)
        n_words = max(1, min(len(pool), n_words - step if diff > 0 else n_words + step))

    return best_text


class Workload:
    """Generates the requests of every turn for one preset, seed, and layout.

    :param preset: Token budgets.
    :param seed: Global seed. The same seed gives the same prompts.
    :param layout: History layout of stage 3.
    :param count_tokens: Token counter used for calibration.
    """

    def __init__(self, preset: Preset, seed: int, layout: Layout, count_tokens: TokenCounter) -> None:
        self.preset = preset
        self.seed = seed
        self.layout = layout
        self._count = count_tokens
        self._cache: dict[str, str] = {}

    def _text(self, name: str, n_tokens: int) -> str:
        """Returns the calibrated text of one segment, computed once per name.

        :param name: Segment name.
        :param n_tokens: Target token count.
        :return: The text.
        """
        if name not in self._cache:
            self._cache[name] = calibrated_text(self.seed, name, n_tokens, self._count)
        return self._cache[name]

    def system(self, name: str, n_tokens: int) -> str:
        """Returns a system prompt: the shared part (if the preset has one), then the request's own part.

        :param name: Segment name of the request's own part.
        :param n_tokens: Tokens of the request's own part.
        :return: The system prompt text.
        """
        own = self._text(name, n_tokens)
        if self.preset.sys_shared <= 0:
            return own
        return f"{self._text('sys/shared', self.preset.sys_shared)}\n{own}"

    def query(self, turn: int) -> str:
        """Returns the user query of a turn."""
        return self._text(f"query/{turn}", self.preset.query)

    def answer(self, turn: int) -> str:
        """Returns the synthetic assistant answer that the client stores in its history for a turn."""
        return self._text(f"answer/{turn}", self.preset.history_answer)

    def history_text(self, turn: int) -> str:
        """Returns the inline history before ``turn``, as the client renders it.

        Every entry ends with a newline, so the text of turn ``n`` is always a prefix of the text of turn ``n + 1``.

        :param turn: Turn index. Turn 0 has an empty history.
        :return: The history text.
        """
        return "".join(f"user_{i}: {self.query(i)}\nassistant_{i}: {self.answer(i)}\n" for i in range(turn))

    def stage_requests(self, turn: int) -> list[Request]:
        """Returns the three foreground requests of a turn, in execution order.

        Dynamic values (metadata and query) are placed in the user message, after the static system prompt. The static prefix therefore ends at the start of the last user message, where the server already creates a context checkpoint.

        :param turn: Turn index.
        :return: Requests for stage 1, stage 2, and stage 3.
        """
        p = self.preset
        meta = self._text(f"meta/{turn}", p.meta)
        query = self.query(turn)
        dynamic = f"metadata: {meta}\nuser query: {query}"

        router: list[ChatMessage] = [
            {"role": "system", "content": self.system("sys/router", p.sys_router)},
            {"role": "user", "content": dynamic},
        ]
        extractor: list[ChatMessage] = [
            {"role": "system", "content": self.system("sys/extractor", p.sys_extractor)},
            {"role": "user", "content": dynamic},
        ]

        # Stage 3: system prompt, then the history, then the per-turn documents and query.
        documents = f"<documents>\n{self._text(f'rag/{turn}', p.rag)}\n</documents>\n{dynamic}"
        synthesis: list[ChatMessage] = [{"role": "system", "content": self.system("sys/synthesis", p.sys_synthesis)}]
        if self.layout == Layout.INLINE:
            synthesis.append({"role": "user", "content": f"<history>\n{self.history_text(turn)}</history>\n{documents}"})
        else:
            for i in range(turn):
                synthesis.append({"role": "user", "content": self.query(i)})
                synthesis.append({"role": "assistant", "content": self.answer(i)})
            synthesis.append({"role": "user", "content": documents})

        return [
            Request(turn, Stage.ROUTER, router, p.max_tokens(Stage.ROUTER)),
            Request(turn, Stage.EXTRACTOR, extractor, p.max_tokens(Stage.EXTRACTOR)),
            Request(turn, Stage.SYNTHESIS, synthesis, p.max_tokens(Stage.SYNTHESIS)),
        ]

    def background_request(self, turn: int) -> Request:
        """Returns the background task request (for example, summarizing device notifications).

        :param turn: Turn index at which the task starts.
        :return: The request.
        """
        p = self.preset
        messages: list[ChatMessage] = [
            {"role": "system", "content": self.system("sys/background", p.sys_background)},
            {"role": "user", "content": self._text(f"notifications/{turn}", p.background_input)},
        ]
        return Request(turn, Stage.BACKGROUND, messages, p.max_tokens(Stage.BACKGROUND))
