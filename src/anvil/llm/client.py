"""LLM client contract and the OpenAI-compatible implementation behind it.

``chat`` speaks the OpenAI chat-completions dialect over httpx and hides three
provider quirks from the agent:

* transient failures (429, 5xx, network) are retried with jittered backoff;
* token usage is always reported, estimated at chars/4 when the provider omits it;
* tools work whether or not the provider supports native tool calling (see
  ``tool_mode`` in config.yaml).

Tool calls always come back normalised as ``{"id", "tool", "args"}`` (plus an
``error`` key when native arguments were not valid JSON). The agent keeps its
history in OpenAI format (assistant ``tool_calls`` whose ``function.arguments``
is a JSON string, then ``tool`` messages carrying ``tool_call_id``); in text
mode the client rewrites that history on the way out.
"""

from __future__ import annotations

import email.utils
import itertools
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

import httpx

from anvil.llm.config import LLMConfig
from anvil.llm.errors import LLMConfigError, LLMError
from anvil.llm.messages import normalize_messages
from anvil.llm.reasoning import Reply, content_text, split_message
from anvil.llm.toolcalls import adapt_messages_for_text_mode, loads_lenient, parse_text_tool_calls

logger = logging.getLogger("anvil.llm")
logger.addHandler(logging.NullHandler())  # silent unless the app configures logging (keeps the TUI clean)

API_KEY_ENV = "AI_API_KEY"
_REDACTED = "[REDACTED]"
# Statuses that, in auto mode, may mean "this provider does not accept the tools parameter".
_POSSIBLE_TOOL_REJECTIONS = (400, 404, 422)
_MISSES_BEFORE_TEXT_MODE = 2
_SWALLOWED_BEFORE_TEXT_MODE = 2
_EMPTY_TEXT_RETRIES = 2
_RETRY_TEMPERATURE = 0.5  # a greedy repeat of a request whose reply was swallowed is swallowed the same way
_REMINDER_WITH_TOOLS = (
    "(Your previous reply was empty. Reply now with exactly one tool call: a single fenced ```json block "
    'containing {"tool": "<tool name>", "args": {...}}, and nothing after it.)'
)
_REMINDER_PLAIN = "(Your previous reply was empty. Please answer in plain text.)"
# Parameters worth retrying without when a 400 names one of them (``tools`` is handled by the tool-mode fallback).
_DROPPABLE_PARAMS = (
    "tool_choice", "parallel_tool_calls", "temperature", "top_p", "response_format", "max_tokens", "max_completion_tokens",
)
# DashScope's open-source Qwen3 models refuse non-streaming calls unless thinking is switched off explicitly.
_PARAM_FIXES: dict[str, Any] = {"enable_thinking": False}
_ROLE_COMPLAINT = re.compile(
    r"alternat|consecutive|must be at the beginning|system message must|first message|unexpected role|invalid role|"
    r"role\W+\w+\W+(?:is )?not (?:supported|allowed)|roles? must",
    re.IGNORECASE,
)
_MAX_COMPAT_RETRIES = 4


@dataclass
class LLMResponse:
    """A single model reply: free text, requested tool calls, and token usage."""

    text: str
    tool_calls: list[dict]
    usage: dict


class LLMClient(Protocol):
    """Anything that can answer a chat request; implemented by the real client and by MockLLM."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Send ``messages`` (and optional tool schemas) and return the model's reply."""
        ...


@dataclass
class UsageTotals:
    """Cumulative token usage and number of successful model calls on one client."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    reasoning_tokens: int = 0  # as reported by the provider (already inside completion_tokens)
    reasoning_replies: int = 0  # replies that carried reasoning, which was removed from the text
    ignored_calls: int = 0  # tool calls beyond the first in a text-mode reply, which were not run


class _Retryable(Exception):
    """Internal signal: this attempt failed in a way worth retrying."""

    def __init__(self, error: LLMError, retry_after: float | None = None) -> None:
        super().__init__(str(error))
        self.error = error
        self.retry_after = retry_after


class OpenAICompatClient(LLMClient):
    """Chat-completions client for any OpenAI-compatible provider.

    The API key is held privately, only ever sent in the Authorization header,
    and scrubbed from every error message and from ``repr``.
    """

    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._url = f"{config.base_url}/chat/completions"
        self._http = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds, connect=config.connect_timeout_seconds),
            transport=transport,
        )
        self._sleep = sleep
        self._rng = rng
        self._call_ids = itertools.count(1)
        self._use_text = config.tool_mode == "text"
        self._native_misses = 0
        self._swallowed_replies = 0
        self.totals = UsageTotals()
        self.reasoning_seen = False  # any reply so far carried <think> tags or a reasoning field
        self.workarounds: list[str] = []  # 400s the client got around; kept for the run (see ``_compat_fix``)
        self._dropped: set[str] = set()  # request parameters the endpoint rejected
        self._forced: dict[str, Any] = {}  # parameters the endpoint demanded
        self._max_tokens_key = "max_tokens"
        self._strict_roles = False  # the endpoint rejected the message roles: normalise native histories too

    def __repr__(self) -> str:
        return (
            f"OpenAICompatClient(model={self._config.model!r}, base_url={self._config.base_url!r}, "
            f"tool_mode={self.active_tool_mode!r})"
        )

    @property
    def active_tool_mode(self) -> str:
        """``"native"`` or ``"text"``: how tools are being sent right now (auto may switch to text)."""
        return "text" if self._use_text else "native"

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Send ``messages`` (and optional OpenAI-format ``tools``) and return the reply.

        Raises ``LLMError`` once retries are exhausted or on a non-retryable failure.
        """
        tools = list(tools) if tools else None
        if self._use_text:
            return self._chat_text(messages, tools)
        try:
            return self._chat_native(messages, tools)
        except LLMError as err:
            if not tools or self._config.tool_mode != "auto" or err.status_code not in _POSSIBLE_TOOL_REJECTIONS:
                raise
            logger.warning("HTTP %s with tools attached; trying text-mode tool calling", err.status_code)
            response = self._chat_text(messages, tools)
            self._use_text = True  # only once text mode has actually worked
            return response

    # ---- the two tool dialects -----------------------------------------------------------

    def _chat_native(self, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        payload = self._payload(messages)
        if tools:
            payload["tools"] = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        data = self._request(payload)
        message = data["choices"][0]["message"]
        reply = split_message(message, strip=self._config.strip_reasoning)
        text = reply.text
        calls = self._native_calls(message)
        if tools and self._config.tool_mode == "auto":
            text, calls = self._rescue_native_miss(text, calls, tools)
            if not calls and not text.strip() and _swallowed(data, reply):
                return self._repeat_in_text_mode(messages, tools, payload, data, reply)
        return self._finish(text, calls, payload, data, reply)

    def _repeat_in_text_mode(
        self, messages: list[dict], tools: list[dict], payload: dict[str, Any], data: dict, reply: Reply
    ) -> LLMResponse:
        """Repeat a request whose reply the endpoint billed for but returned nothing of.

        Some providers run their own tool-call parser over the model's output and, when it cannot read it, return
        ``content: null`` without a call, whether the model was calling a tool or answering. Text mode has no server-side
        parser to swallow anything. The wasted call stays in the totals and is added to the response's usage, so the
        caller's budget sees everything that was billed. After ``_SWALLOWED_BEFORE_TEXT_MODE`` such replies the client
        stays in text mode.
        """
        wasted = self._finish("", [], payload, data, reply).usage
        self._swallowed_replies += 1
        if self._swallowed_replies >= _SWALLOWED_BEFORE_TEXT_MODE:
            logger.warning("the endpoint swallowed %d replies; switching to text mode for good", self._swallowed_replies)
            self._use_text = True
        else:
            logger.warning(
                "the endpoint billed %s tokens for an empty reply; repeating the request in text mode",
                wasted["completion_tokens"],
            )
        response = self._chat_text(messages, tools)
        response.usage = _sum_usage(wasted, response.usage)
        return response

    def _chat_text(self, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        """One text-mode request; if the reply was billed but came back empty, ask again (twice at most).

        Some providers strip what they take for a tool call out of the reply even when no tools were sent, leaving
        ``content: null``. The agent would answer that with a nudge and, on a second one, end the phase; here the
        request is repeated first, with a reminder added for that call only and a warmer temperature. What the
        empty attempts cost is added to the response's usage.
        """
        response, empty = self._text_attempt(messages, tools)
        retries = 0
        while empty and retries < _EMPTY_TEXT_RETRIES:
            retries += 1
            logger.warning("empty reply in text mode (%d tokens billed); asking again (%d of %d)",
                           response.usage["completion_tokens"], retries, _EMPTY_TEXT_RETRIES)
            wasted = response.usage
            response, empty = self._text_attempt(messages, tools, reminder=True)
            response.usage = _sum_usage(wasted, response.usage)
        return response

    def _text_attempt(
        self, messages: list[dict], tools: list[dict] | None, *, reminder: bool = False
    ) -> tuple[LLMResponse, bool]:
        """One text-mode request and whether its reply was swallowed (empty, yet billed)."""
        if reminder:
            messages = messages + [{"role": "user", "content": _REMINDER_WITH_TOOLS if tools else _REMINDER_PLAIN}]
        payload = self._payload(normalize_messages(adapt_messages_for_text_mode(messages, tools)))
        if reminder and "temperature" in payload:
            payload["temperature"] = max(float(payload["temperature"] or 0), _RETRY_TEMPERATURE)
        data = self._request(payload)
        message = data["choices"][0]["message"]
        reply = split_message(message, strip=self._config.strip_reasoning)
        text = reply.text
        calls: list[dict] = []
        if tools:
            text, calls = self._first_text_call(text, tools)
        empty = not calls and not text.strip() and _swallowed(data, reply)
        return self._finish(text, calls, payload, data, reply), empty

    def _first_text_call(self, text: str, tools: list[dict]) -> tuple[str, list[dict]]:
        """The first tool call written in ``text`` (the text-mode protocol is one call per reply) and the prose around it.

        Further calls are dropped from the text and counted; the count travels on the call as ``ignored_calls`` so
        the agent can tell the model that only the first one ran.
        """
        parsed = parse_text_tool_calls(text, tools)
        if not parsed.calls:
            return text, []
        first = parsed.calls[0]
        call = self._new_call(first["tool"], first["args"])
        extra = len(parsed.calls) - 1
        if extra:
            call["ignored_calls"] = extra
            self.totals.ignored_calls += extra
            logger.warning("the reply contained %d tool calls; only the first will run", extra + 1)
        return parsed.text, [call]

    def _native_calls(self, message: dict) -> list[dict]:
        calls = []
        raw_calls = message.get("tool_calls") or []
        legacy = message.get("function_call")  # the pre-2023 form, still returned by some servers
        if not raw_calls and isinstance(legacy, dict):
            raw_calls = [{"function": legacy}]
        for raw in raw_calls:
            fn = raw.get("function") if isinstance(raw, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if not isinstance(name, str) or not name:
                continue
            args_raw = fn.get("arguments")
            call = self._new_call(name, {}, call_id=raw.get("id"))
            if isinstance(args_raw, dict):
                call["args"] = args_raw
            elif isinstance(args_raw, str) and args_raw.strip():
                try:
                    parsed = loads_lenient(args_raw)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    call["args"] = parsed
                else:
                    call["error"] = f"arguments were not a valid JSON object: {args_raw[:200]!r}"
            elif args_raw is not None and not isinstance(args_raw, str):
                call["error"] = f"arguments were not a valid JSON object: {json.dumps(args_raw, default=str)[:200]}"
            calls.append(call)
        return calls

    def _rescue_native_miss(self, text: str, calls: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        """Auto mode: spot replies that put a tool call in the text instead of ``tool_calls``.

        The call is rescued from the text. After two such replies in a row the
        client switches to text mode for good.
        """
        remaining, rescued = self._first_text_call(text, tools) if not calls else (text, [])
        if not rescued:
            self._native_misses = 0
            return text, calls
        self._native_misses += 1
        if self._native_misses >= _MISSES_BEFORE_TEXT_MODE:
            logger.warning("provider ignored native tool calling twice in a row; switching to text mode")
            self._use_text = True
        return remaining, rescued

    def _new_call(self, tool: str, args: dict, call_id: Any = None) -> dict:
        return {"id": call_id or f"call_{next(self._call_ids)}", "tool": tool, "args": args}

    # ---- request / retry -----------------------------------------------------------------

    def _payload(self, messages: list[dict]) -> dict[str, Any]:
        """The request body, minus what this endpoint has already rejected and plus what it has demanded."""
        cfg = self._config
        if self._strict_roles:
            messages = normalize_messages(messages, native=True)
        payload: dict[str, Any] = {"model": cfg.model, "messages": messages}
        if "temperature" not in self._dropped:
            payload["temperature"] = cfg.temperature
        if cfg.max_output_tokens is not None and self._max_tokens_key not in self._dropped:
            payload[self._max_tokens_key] = cfg.max_output_tokens
        payload.update({k: v for k, v in cfg.extra_params.items() if k not in self._dropped})
        payload.update(self._forced)
        return payload

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST, and when the endpoint answers 400 about something we can leave out or fix, do that and try again.

        See ``_compat_fix``. Each fix is remembered, so it costs one extra request per run, not one per call.
        """
        for _ in range(_MAX_COMPAT_RETRIES + 1):
            try:
                return self._request_with_retries(payload)
            except LLMError as err:
                repaired = self._compat_fix(err, payload)
                if repaired is None:
                    raise
                payload = repaired
        raise AssertionError("unreachable: _compat_fix stops changing the payload")  # pragma: no cover

    def _compat_fix(self, err: LLMError, payload: dict[str, Any]) -> dict[str, Any] | None:
        """A repaired copy of ``payload`` for a 400/422 that names something fixable, or ``None`` if there is nothing to fix.

        * a parameter we sent that the endpoint does not support (``temperature``, ``top_p``, ``tool_choice``,
          ``parallel_tool_calls``, ``response_format``, ``max_tokens``, or a configured extra) is dropped;
        * ``max_tokens`` rejected in favour of ``max_completion_tokens`` is renamed;
        * DashScope's ``enable_thinking`` demand is met;
        * a complaint about message roles (must alternate, system must be first) turns on message normalisation.
        The ``tools`` parameter is not handled here: in ``auto`` mode a rejection switches to text-mode tool calling.
        """
        if err.status_code not in (400, 422) or err.retryable:
            return None
        text = f"{err} {err.detail}".lower()
        fixed = dict(payload)
        notes: list[str] = []

        if "max_tokens" in fixed and "max_completion_tokens" in text and self._max_tokens_key == "max_tokens":
            self._max_tokens_key = "max_completion_tokens"
            fixed["max_completion_tokens"] = fixed.pop("max_tokens")
            notes.append("renamed max_tokens to max_completion_tokens")
        else:
            candidates = [*_DROPPABLE_PARAMS, *self._config.extra_params]
            for name in dict.fromkeys(candidates):
                if name in fixed and name not in ("model", "messages", "tools") and _mentions(text, name):
                    self._dropped.add(name)
                    del fixed[name]
                    notes.append(f"dropped parameter '{name}'")
        for name, value in _PARAM_FIXES.items():
            if name not in fixed and _mentions(text, name):
                self._forced[name] = fixed[name] = value
                notes.append(f"set {name}={value}")
        if not self._strict_roles and _ROLE_COMPLAINT.search(text):
            self._strict_roles = True
            fixed["messages"] = normalize_messages(fixed["messages"], native=True)
            notes.append("normalised the message roles")

        if not notes:
            return None
        why = f"HTTP {err.status_code}: {_excerpt(err.detail or str(err), 160)}"
        for note in notes:
            self.workarounds.append(f"{note} ({why})")
        logger.warning("%s; retrying (%s)", why, "; ".join(notes))
        return fixed

    def _request_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries; return the parsed body of the first good response.

        A good body is guaranteed to have ``choices[0].message`` as a dict.
        """
        attempts = self._config.max_attempts
        for attempt in range(1, attempts + 1):
            try:
                return self._send_once(payload, attempt)
            except _Retryable as failure:
                if attempt == attempts:
                    raise failure.error from None
                delay = self._delay(attempt, failure.retry_after)
                logger.warning("%s; retry %d/%d in %.1fs", failure.error, attempt, attempts - 1, delay)
                self._sleep(delay)
        raise AssertionError("unreachable: max_attempts >= 1")  # pragma: no cover

    def _send_once(self, payload: dict[str, Any], attempt: int) -> dict:
        try:
            response = self._http.post(
                self._url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.TransportError as exc:
            error = self._error(f"network error: {type(exc).__name__}: {exc}", attempt, retryable=True)
            raise _Retryable(error) from None
        except httpx.RequestError as exc:
            raise self._error(f"request failed: {type(exc).__name__}: {exc}", attempt) from None

        status = response.status_code
        if not response.is_success:
            detail = _excerpt(response.text)
            transient = status == 429 or status >= 500
            error = self._error(
                f"LLM request failed with HTTP {status}", attempt, status=status, retryable=transient, detail=detail
            )
            if transient:
                raise _Retryable(error, _retry_after_seconds(response.headers.get("retry-after")))
            raise error

        try:
            data = response.json()
        except ValueError:
            raise _Retryable(self._error("response body was not valid JSON", attempt, retryable=True)) from None
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            if isinstance(choices[0].get("message"), dict):
                return data
        raise self._body_error(data, attempt)

    def _body_error(self, data: Any, attempt: int) -> LLMError | _Retryable:
        """Classify a 200 response that carries no completion (some gateways report errors this way)."""
        error = data.get("error") if isinstance(data, dict) else None
        if not isinstance(error, dict):
            return _Retryable(self._error("response contained no completion", attempt, retryable=True))
        code = error.get("code")
        transient = isinstance(code, int) and (code == 429 or code >= 500)
        failure = self._error(
            "provider returned an error payload",
            attempt,
            status=code if isinstance(code, int) else None,
            retryable=transient,
            detail=_excerpt(json.dumps(error)),
        )
        return _Retryable(failure) if transient else failure

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with equal jitter; a Retry-After hint sets the floor. Always capped."""
        cfg = self._config
        backoff = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * 2 ** (attempt - 1))
        delay = backoff / 2 + self._rng() * backoff / 2
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, cfg.backoff_max_seconds)

    # ---- usage ---------------------------------------------------------------------------

    def _finish(
        self, text: str, calls: list[dict], payload: dict[str, Any], data: dict, reply: Reply | None = None
    ) -> LLMResponse:
        usage = self._usage(data.get("usage"), payload, data["choices"][0]["message"])
        self.totals.prompt_tokens += usage["prompt_tokens"]
        self.totals.completion_tokens += usage["completion_tokens"]
        self.totals.total_tokens += usage["total_tokens"]
        self.totals.reasoning_tokens += usage.get("reasoning_tokens", 0)
        self.totals.calls += 1
        if reply is not None and reply.found:
            self.reasoning_seen = True
            self.totals.reasoning_replies += 1
        return LLMResponse(text=text, tool_calls=calls, usage=usage)

    @staticmethod
    def _usage(reported: Any, payload: dict[str, Any], message: dict) -> dict:
        """Provider usage where given, chars/4 estimates for whatever is missing."""
        reported = reported if isinstance(reported, dict) else {}
        prompt = _as_count(reported.get("prompt_tokens"))
        completion = _as_count(reported.get("completion_tokens"))
        total = _as_count(reported.get("total_tokens"))
        estimated = prompt is None or completion is None
        if prompt is None:
            prompt = _tokens_from_chars(_prompt_chars(payload))
        if completion is None:
            completion = _tokens_from_chars(_completion_chars(message))
        if total is None:
            total = prompt + completion
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "estimated": estimated,
        }
        details = reported.get("completion_tokens_details")
        reasoning = _as_count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        if reasoning is not None:
            usage["reasoning_tokens"] = reasoning  # informational: providers already include it in completion_tokens
        return usage

    # ---- errors --------------------------------------------------------------------------

    def _error(
        self,
        message: str,
        attempt: int,
        *,
        status: int | None = None,
        retryable: bool = False,
        detail: str = "",
    ) -> LLMError:
        """Build an ``LLMError`` whose text can never contain the API key."""
        message, detail = self._redact(message), self._redact(detail)
        full = f"{message}: {detail}" if detail else message
        return LLMError(full, status_code=status, retryable=retryable, attempts=attempt, detail=detail)

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, _REDACTED) if self._api_key else text


def make_client(
    config: LLMConfig | Mapping[str, Any],
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> OpenAICompatClient:
    """Build the LLM client from the parsed ``config.yaml`` (or an ``LLMConfig``).

    The key is read only from the ``AI_API_KEY`` environment variable;
    ``AI_MODEL`` / ``AI_BASE_URL`` override the config. Raises ``LLMConfigError``
    if the key is missing or malformed or the config is invalid. ``transport``,
    ``sleep`` and ``rng`` exist so tests can run offline and instantly.
    """
    llm_config = config if isinstance(config, LLMConfig) else LLMConfig.from_mapping(config)
    return OpenAICompatClient(llm_config, _read_api_key(), transport=transport, sleep=sleep, rng=rng)


def _read_api_key() -> str:
    try:
        key = os.environ[API_KEY_ENV]
    except KeyError:
        raise LLMConfigError(
            f"{API_KEY_ENV} environment variable is not set; export it before running (see .env.example)"
        ) from None
    key = key.strip()
    if not key:
        raise LLMConfigError(f"{API_KEY_ENV} environment variable is empty")
    if not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
        raise LLMConfigError(
            f"{API_KEY_ENV} contains spaces, line breaks or non-ASCII characters; check how it was exported"
        )
    return key


def _mentions(text: str, name: str) -> bool:
    """Whether ``name`` appears in ``text`` as a whole identifier (``tools`` must not match ``tool_choice``)."""
    return re.search(rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])", text) is not None


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header (delay in seconds or an HTTP date); ``None`` if absent or unparseable."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds)


def _excerpt(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _as_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _swallowed(data: dict, reply: Reply) -> bool:
    """A reply that cost tokens and came back with nothing: no text, no calls, no reasoning, and not cut off."""
    spent = _as_count((data.get("usage") or {}).get("completion_tokens")) or 0
    return spent > 0 and not reply.found and data["choices"][0].get("finish_reason") == "stop"


def _sum_usage(first: dict, second: dict) -> dict:
    """The usage of two calls as one: token counts added, ``estimated`` if either was."""
    merged = dict(second)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        if key in first or key in second:
            merged[key] = first.get(key, 0) + second.get(key, 0)
    merged["estimated"] = bool(first.get("estimated") or second.get("estimated"))
    return merged


def _tokens_from_chars(chars: int) -> int:
    return (chars + 3) // 4


def _prompt_chars(payload: dict[str, Any]) -> int:
    total = 0
    for msg in payload["messages"]:
        content = msg.get("content")
        total += len(content) if isinstance(content, str) else len(json.dumps(content))
        if msg.get("tool_calls"):
            total += len(json.dumps(msg["tool_calls"]))
    if payload.get("tools"):
        total += len(json.dumps(payload["tools"]))
    return total


def _completion_chars(message: dict) -> int:
    """Characters the model generated, reasoning included: it is billed even though it is not kept."""
    total = len(content_text(message.get("content")))
    for field in ("reasoning_content", "reasoning"):
        if isinstance(message.get(field), str):
            total += len(message[field])
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") if isinstance(raw, dict) else None
        if isinstance(fn, dict):
            args = fn.get("arguments")
            total += len(str(fn.get("name", ""))) + (len(args) if isinstance(args, str) else len(json.dumps(args)))
    return total
