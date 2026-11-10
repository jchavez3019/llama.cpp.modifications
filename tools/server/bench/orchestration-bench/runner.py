"""HTTP client and turn runner for the orchestration workload.

The runner sends the requests of every turn to a running ``llama-server`` and writes one JSON line per request. Before the timed loop it tokenizes every prompt (through ``/apply-template`` and ``/tokenize``), so it can report how many tokens a perfect prefix cache could reuse.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import IO, Iterator, Mapping, Sequence, Union

import requests

from workload import ChatMessage, Request, Stage, Workload

JsonValue = Union[None, bool, int, float, str, Sequence["JsonValue"], Mapping[str, "JsonValue"]]
JsonObject = Mapping[str, JsonValue]

# Slot id meaning "let the server choose". The server itself uses -1 the same way.
NO_SLOT = -1

_HTTP_TIMEOUT_S = 1800
_HEALTH_TIMEOUT_S = 5
_HEALTH_POLL_S = 0.5
_HTTP_OK = 200
_MS_PER_S = 1000.0
_SSE_PREFIX = "data: "
_SSE_DONE = "[DONE]"


class PinMode(StrEnum):
    """How the client assigns the foreground stages to server slots."""

    NONE = "none"
    SLOT0 = "slot0"
    PER_STAGE = "per-stage"


@dataclass(frozen=True)
class PinPolicy:
    """Slot of each foreground stage. ``NO_SLOT`` lets the server choose.

    :param router: Slot of stage 1.
    :param extractor: Slot of stage 2.
    :param synthesis: Slot of stage 3.
    """

    router: int
    extractor: int
    synthesis: int

    def slot_for(self, stage: Stage) -> int:
        """Returns the slot of a foreground stage.

        :raises ValueError: For the background task, whose slot comes from ``RunConfig.bg_slot``.
        """
        if stage == Stage.ROUTER:
            return self.router
        if stage == Stage.EXTRACTOR:
            return self.extractor
        if stage == Stage.SYNTHESIS:
            return self.synthesis
        raise ValueError(f"no pinned slot for stage {stage}")


PIN_POLICIES: dict[PinMode, PinPolicy] = {
    PinMode.NONE: PinPolicy(router=NO_SLOT, extractor=NO_SLOT, synthesis=NO_SLOT),
    PinMode.SLOT0: PinPolicy(router=0, extractor=0, synthesis=0),
    PinMode.PER_STAGE: PinPolicy(router=0, extractor=1, synthesis=2),
}


def messages_json(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Returns chat messages as plain JSON objects."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


@dataclass(frozen=True)
class ChatCompletionBody:
    """Body of a streaming ``/v1/chat/completions`` request.

    Generation uses temperature 0 and ``ignore_eos``, so every run generates exactly ``max_tokens`` tokens and stays comparable across configurations.

    :param messages: Chat messages.
    :param max_tokens: Tokens to generate.
    :param id_slot: Slot to pin the request to, or ``NO_SLOT``.
    """

    messages: list[ChatMessage]
    max_tokens: int
    id_slot: int = NO_SLOT

    def to_json(self) -> dict[str, JsonValue]:
        """Returns the JSON body. ``id_slot`` is left out when the server should choose the slot."""
        body: dict[str, JsonValue] = {
            "messages": messages_json(self.messages),
            "max_tokens": self.max_tokens,
            "stream": True,
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": True,
        }
        if self.id_slot != NO_SLOT:
            body["id_slot"] = self.id_slot
        return body


@dataclass(frozen=True)
class ServerTimings:
    """The ``timings`` object of the last stream chunk.

    Fields follow ``server_slot_stats`` / ``timings`` in ``tools/server/server-common.cpp``. Keep them in sync if the server changes.

    :param cache_n: Prompt tokens reused from the cache.
    :param prompt_n: Prompt tokens evaluated in prefill.
    :param prompt_ms: Server prefill time.
    :param predicted_n: Generated tokens.
    :param predicted_ms: Server generation time.
    """

    cache_n: int
    prompt_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float

    @staticmethod
    def from_json(obj: JsonObject) -> "ServerTimings":
        """Parses the ``timings`` object. Missing or non-numeric fields become 0."""
        return ServerTimings(
            cache_n=int(_num(obj, "cache_n")),
            prompt_n=int(_num(obj, "prompt_n")),
            prompt_ms=_num(obj, "prompt_ms"),
            predicted_n=int(_num(obj, "predicted_n")),
            predicted_ms=_num(obj, "predicted_ms"),
        )


@dataclass
class RequestRecord:
    """Measurements of one request, written as one JSON line.

    :param turn: Turn index.
    :param stage: Request kind (a ``Stage`` value).
    :param warmup: True if the turn is a warm-up turn, excluded from summaries.
    :param id_slot: Requested slot, or ``NO_SLOT`` when the server selects the slot.
    :param t_start: Wall-clock start time on the client (seconds since the epoch).
    :param ttft_ms: Client time to the first generated content, or -1 if none arrived.
    :param e2e_ms: Client time to the end of the stream.
    :param prompt_tokens: Prompt tokens of the request (``cache_n + prompt_n``).
    :param cache_n: Prompt tokens reused from the cache.
    :param prompt_n: Prompt tokens evaluated in prefill.
    :param prompt_ms: Server prefill time.
    :param predicted_n: Generated tokens.
    :param predicted_ms: Server generation time.
    :param ideal_cached: Longest token prefix shared with any earlier request of the run. A perfect prefix cache with no size limit reuses this many tokens.
    :param client_prompt_tokens: Prompt token count computed by the client before the run.
    :param error: Error message, or an empty string.
    """

    turn: int
    stage: str
    warmup: bool
    id_slot: int
    t_start: float
    ttft_ms: float = -1.0
    e2e_ms: float = -1.0
    prompt_tokens: int = 0
    cache_n: int = 0
    prompt_n: int = 0
    prompt_ms: float = 0.0
    predicted_n: int = 0
    predicted_ms: float = 0.0
    ideal_cached: int = 0
    client_prompt_tokens: int = 0
    error: str = ""

    @staticmethod
    def from_json(obj: Mapping[str, object]) -> "RequestRecord":
        """Builds a record from one JSON line. Unknown keys are ignored, missing optional keys keep their defaults."""
        known = {f.name for f in fields(RequestRecord)}
        return RequestRecord(**{k: v for k, v in obj.items() if k in known})  # type: ignore[arg-type]

    def apply_timings(self, timings: ServerTimings) -> None:
        """Copies the server timings into the record."""
        self.cache_n = timings.cache_n
        self.prompt_n = timings.prompt_n
        self.prompt_ms = timings.prompt_ms
        self.predicted_n = timings.predicted_n
        self.predicted_ms = timings.predicted_ms
        self.prompt_tokens = timings.cache_n + timings.prompt_n


class ServerClient:
    """Thin client for the ``llama-server`` endpoints that the benchmark uses.

    :param base_url: Server URL, for example ``http://127.0.0.1:8080``.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def wait_ready(self, timeout_s: float) -> None:
        """Blocks until ``/health`` returns 200 (the model is loaded).

        :param timeout_s: Maximum wait time.
        :raises TimeoutError: If the server is not ready in time.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if self.session.get(f"{self.base_url}/health", timeout=_HEALTH_TIMEOUT_S).status_code == _HTTP_OK:
                    return
            except requests.RequestException:
                pass
            time.sleep(_HEALTH_POLL_S)
        raise TimeoutError(f"server at {self.base_url} not ready after {timeout_s} s")

    def tokenize(self, text: str, add_special: bool) -> list[int]:
        """Tokenizes text with the model's tokenizer.

        :param text: Text to tokenize. Special-token text (for example ``<|eot_id|>``) is parsed as special tokens.
        :param add_special: Add BOS and other special tokens, as the server does for a chat prompt.
        :return: Token ids.
        """
        body = {"content": text, "add_special": add_special, "parse_special": True}
        res = self.session.post(f"{self.base_url}/tokenize", json=body, timeout=_HTTP_TIMEOUT_S)
        res.raise_for_status()
        return [int(t) for t in res.json()["tokens"]]

    def count_tokens(self, text: str) -> int:
        """Returns the token count of plain text, without special tokens."""
        return len(self.tokenize(text, add_special=False))

    def prompt_tokens(self, messages: list[ChatMessage]) -> list[int]:
        """Returns the tokens of the full chat prompt, after the model's chat template is applied.

        :param messages: Chat messages.
        :return: Token ids, as the server tokenizes them for a completion.
        """
        res = self.session.post(f"{self.base_url}/apply-template", json={"messages": messages_json(messages)}, timeout=_HTTP_TIMEOUT_S)
        res.raise_for_status()
        return self.tokenize(res.json()["prompt"], add_special=True)

    def chat_stream(self, request: Request, id_slot: int, record: RequestRecord) -> None:
        """Sends one streaming chat completion and fills ``record`` with the measurements.

        :param request: The request to send.
        :param id_slot: Slot to pin the request to, or ``NO_SLOT``.
        :param record: Record to fill. Errors are stored in ``record.error`` instead of raised, so a failure in the background thread is never lost.
        """
        body = ChatCompletionBody(request.messages, request.max_tokens, id_slot).to_json()
        t0 = time.perf_counter()
        try:
            with self.session.post(f"{self.base_url}/v1/chat/completions", json=body, stream=True, timeout=_HTTP_TIMEOUT_S) as res:
                res.raise_for_status()
                for chunk in _sse_chunks(res.iter_lines(decode_unicode=True)):
                    if "error" in chunk:
                        record.error = json.dumps(chunk["error"])
                        break
                    if record.ttft_ms < 0 and _has_content(chunk):
                        record.ttft_ms = (time.perf_counter() - t0) * _MS_PER_S
                    timings = chunk.get("timings")
                    if isinstance(timings, Mapping):
                        record.apply_timings(ServerTimings.from_json(timings))
        except (requests.RequestException, ValueError) as exc:
            record.error = f"{type(exc).__name__}: {exc}"
        record.e2e_ms = (time.perf_counter() - t0) * _MS_PER_S


def _sse_chunks(lines: Iterator[str | bytes]) -> Iterator[JsonObject]:
    """Parses server-sent event lines into JSON chunks.

    :param lines: Response lines.
    :return: One dict per ``data:`` line, until ``[DONE]``.
    :raises ValueError: If a data line is not valid JSON.
    """
    for raw in lines:
        line = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        if not line.startswith(_SSE_PREFIX):
            continue
        payload = line[len(_SSE_PREFIX):]
        if payload == _SSE_DONE:
            return
        yield json.loads(payload)


def _has_content(chunk: JsonObject) -> bool:
    """Returns True if a stream chunk carries generated text. The first chunk can carry only the role."""
    choices = chunk.get("choices")
    if not isinstance(choices, Sequence) or not choices or not isinstance(choices[0], Mapping):
        return False
    delta = choices[0].get("delta")
    if not isinstance(delta, Mapping):
        return False
    return bool(delta.get("content") or delta.get("reasoning_content"))


def _num(obj: JsonObject, key: str) -> float:
    """Returns a numeric field of a JSON object, or 0 if it is absent or not a number."""
    value = obj.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def common_prefix_len(a: list[int], b: list[int]) -> int:
    """Returns the length of the longest common prefix of two token lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass(frozen=True)
class RunConfig:
    """Client-side settings of one run.

    :param n_turns: Total turns, including warm-up turns.
    :param n_warmup: First turns that are excluded from summaries.
    :param pin: Slot pinning mode for the foreground stages.
    :param bg_every: Run the background task every this many turns. 0 disables it.
    :param bg_slot: Slot for the background task, or ``NO_SLOT``.
    :param bg_concurrent: Run the background task in parallel with the stages instead of before them.
    """

    n_turns: int
    n_warmup: int
    pin: PinMode
    bg_every: int
    bg_slot: int
    bg_concurrent: bool


@dataclass(frozen=True)
class PlannedRequest:
    """A request with its token data, computed before the timed loop.

    :param request: The request.
    :param prompt_tokens: Prompt token count after the chat template.
    :param ideal_cached: Tokens a perfect prefix cache could reuse (see ``ideal_reuse``).
    """

    request: Request
    prompt_tokens: int
    ideal_cached: int


def planned_requests(workload: Workload, cfg: RunConfig) -> list[Request]:
    """Returns all requests of a run, in the order they start.

    A background task of turn ``t`` is listed before stage 1 of turn ``t``.

    :param workload: The workload.
    :param cfg: Run settings.
    :return: Requests in start order.
    """
    out: list[Request] = []
    for turn in range(cfg.n_turns):
        if cfg.bg_every > 0 and turn % cfg.bg_every == 0:
            out.append(workload.background_request(turn))
        out.extend(workload.stage_requests(turn))
    return out


def ideal_reuse(prompts: list[list[int]]) -> list[int]:
    """Computes, for every prompt, the longest prefix shared with any earlier prompt.

    :param prompts: Token lists in start order.
    :return: Reusable token count per prompt. The first prompt always gets 0.
    """
    out: list[int] = []
    for i, cur in enumerate(prompts):
        best = 0
        for prev in prompts[:i]:
            best = max(best, common_prefix_len(cur, prev))
        out.append(best)
    return out


def run(client: ServerClient, workload: Workload, cfg: RunConfig, out: IO[str]) -> list[RequestRecord]:
    """Runs all turns and writes one JSON line per request to ``out``.

    At most one background thread exists at a time: it is joined at the end of its turn, so a slow background task delays the next turn instead of piling up.

    :param client: Server client.
    :param workload: The workload.
    :param cfg: Run settings.
    :param out: Text stream for the JSONL records.
    :return: All records, in completion order.
    """
    # Tokenize every prompt before the timed loop, so tokenization does not disturb the measurements.
    requests_ = planned_requests(workload, cfg)
    prompts = [client.prompt_tokens(req.messages) for req in requests_]
    plan = [PlannedRequest(req, len(tok), reuse) for req, tok, reuse in zip(requests_, prompts, ideal_reuse(prompts))]

    records: list[RequestRecord] = []
    lock = threading.Lock()
    pins = PIN_POLICIES[cfg.pin]

    def send(item: PlannedRequest, id_slot: int) -> None:
        req = item.request
        rec = RequestRecord(turn=req.turn, stage=req.stage, warmup=req.turn < cfg.n_warmup, id_slot=id_slot, t_start=time.time(), ideal_cached=item.ideal_cached, client_prompt_tokens=item.prompt_tokens)
        client.chat_stream(req, id_slot, rec)

        # The background thread and the main thread both write records, so writes are serialized.
        with lock:
            records.append(rec)
            out.write(json.dumps(asdict(rec)) + "\n")
            out.flush()

    # Each turn: start the background task (in parallel or before), then run the stages one after another.
    by_turn: dict[int, list[PlannedRequest]] = {}
    for item in plan:
        by_turn.setdefault(item.request.turn, []).append(item)
    for turn in range(cfg.n_turns):
        bg_thread: threading.Thread | None = None
        for item in by_turn[turn]:
            if item.request.stage != Stage.BACKGROUND:
                send(item, pins.slot_for(item.request.stage))
            elif cfg.bg_concurrent:
                bg_thread = threading.Thread(target=send, args=(item, cfg.bg_slot), daemon=True)
                bg_thread.start()
            else:
                send(item, cfg.bg_slot)
        if bg_thread is not None:
            bg_thread.join()
    return records


def output_path(prefix: Path, suffix: str) -> Path:
    """Returns ``<prefix><suffix>``, for example ``rep0`` + ``.meta.json``. ``Path.with_suffix`` would cut names that contain dots."""
    return prefix.parent / f"{prefix.name}{suffix}"


def write_meta(path: Path, meta: Mapping[str, object]) -> None:
    """Writes the run metadata (configuration, model, device, clock offset) next to the records."""
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
