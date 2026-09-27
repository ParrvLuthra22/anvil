"""HTTP 402 from the provider: "retry after in-flight requests settle" is retried, "out of credit" is not.

Both bodies below are what OpenRouter actually returned during real runs on a free-tier key: the first when several requests were in
flight or right after a long run (retry after ~2 minutes), the second once the balance was gone.
"""

from __future__ import annotations

import httpx
import pytest

from anvil.agent.recovery import llm_failure_advice
from anvil.llm import LLMError, make_client
from anvil.llm.client import IN_FLIGHT_MAX_SECONDS, IN_FLIGHT_TRIES
from tests.test_llm_client import CONFIG, KEY, USER, Server, ok

IN_FLIGHT_BODY = {
    "error": {
        "message": (
            "This request would exceed your available credits given your current in-flight requests. "
            "Retry after in-flight requests settle, or add credits."
        ),
        "code": 402,
        "metadata": {
            "reason": "in_flight_budget_exhausted",
            "limit_source": "openrouter_in_flight_budget",
            "remedy_hint": "Retry after your in-flight requests settle (see the Retry-After header).",
            "headers": {"Retry-After": "120"},
        },
    }
}
CREDIT_BODY = {
    "error": {
        "message": (
            "This request requires more credits, or fewer max_tokens. You requested up to 2048 tokens, but can only afford 248. "
            "To increase, visit https://openrouter.ai/settings/credits and upgrade to a paid account"
        ),
        "code": 402,
        "metadata": {"limit_source": "openrouter_credits"},
    }
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)


def in_flight(**headers) -> httpx.Response:
    return httpx.Response(402, json=IN_FLIGHT_BODY, headers=headers)


def credit() -> httpx.Response:
    return httpx.Response(402, json=CREDIT_BODY)


def build(handler, rng=lambda: 1.0, **overrides):
    sleeps: list[float] = []
    client = make_client({**CONFIG, **overrides}, transport=httpx.MockTransport(handler), sleep=sleeps.append, rng=rng)
    return client, sleeps


# ---- "retry after in-flight requests settle" ---------------------------------------------------------------------------


def test_an_in_flight_402_is_retried_with_backoff_and_then_succeeds():
    server = Server(in_flight(), in_flight(), ok("finally"))
    client, sleeps = build(server)
    assert client.chat(USER).text == "finally"
    assert len(server.requests) == 3 and sleeps == [4.0, 8.0]


def test_the_waits_are_jittered_between_half_and_all_of_the_backoff():
    lows, highs = [], []
    for rng, sink in ((lambda: 0.0, lows), (lambda: 1.0, highs)):
        client, sleeps = build(Server(in_flight(), in_flight(), in_flight(), ok()), rng=rng)
        client.chat(USER)
        sink.extend(sleeps)
    assert lows == [2.0, 4.0, 8.0] and highs == [4.0, 8.0, 16.0]


def test_every_wait_stays_between_two_and_thirty_seconds_whatever_the_jitter_and_the_hint():
    for rng in (lambda: 0.0, lambda: 0.37, lambda: 1.0):
        for hint in ({}, {"Retry-After": "0"}, {"Retry-After": "1"}, {"Retry-After": "120"}, {"Retry-After": "9999"}):
            client, sleeps = build(Server(in_flight(**hint), in_flight(**hint), in_flight(**hint), in_flight(**hint), ok()), rng=rng)
            client.chat(USER)
            assert len(sleeps) == 4
            assert all(2.0 <= s <= IN_FLIGHT_MAX_SECONDS for s in sleeps), (hint, sleeps)


def test_a_long_retry_after_is_honoured_up_to_the_ceiling_and_a_short_one_is_a_floor():
    client, sleeps = build(Server(in_flight(**{"Retry-After": "120"}), in_flight(**{"Retry-After": "120"}), ok()))
    client.chat(USER)
    assert sleeps == [30.0, 30.0], "the provider asked for 120 s; the harness waits at most 30 s per try"
    client, sleeps = build(Server(in_flight(**{"Retry-After": "3"}), ok()), rng=lambda: 0.0)
    client.chat(USER)
    assert sleeps == [3.0], "a hint above the jittered backoff is the floor"


def test_the_wait_doubles_up_to_the_ceiling():
    client, sleeps = build(Server(*[in_flight()] * (IN_FLIGHT_TRIES - 1), ok()))
    client.chat(USER)
    assert sleeps == [4.0, 8.0, 16.0, 30.0]


def test_five_tries_in_all_then_a_clear_error_that_says_what_to_do():
    server = Server(in_flight())  # the last item repeats: it never settles
    client, sleeps = build(server)
    with pytest.raises(LLMError) as caught:
        client.chat(USER)
    err = caught.value

    assert len(server.requests) == IN_FLIGHT_TRIES == 5 and len(sleeps) == 4
    assert err.status_code == 402 and err.retryable is True and err.attempts == 5
    assert "in-flight budget stayed exhausted" in str(err) and "5 tries" in str(err) and "58s" in str(err)
    assert "Add credit" in str(err) and "max_output_tokens" in str(err)
    assert "\n" not in str(err), "one line"
    assert "in_flight_budget_exhausted" in err.detail, "the provider's own words are kept for the trace"
    assert "in-flight budget stayed exhausted" in llm_failure_advice(err)


def test_in_flight_retries_do_not_use_up_the_attempts_that_429_and_5xx_share():
    server = Server(httpx.Response(429, text="slow"), in_flight(), in_flight(), in_flight(), ok("through"))
    client, sleeps = build(server, llm_max_attempts=2)
    assert client.chat(USER).text == "through"
    assert len(server.requests) == 5 and len(sleeps) == 4


def test_an_in_flight_wait_before_a_429_does_not_use_up_the_429s_attempts_either():
    server = Server(in_flight(), httpx.Response(429, text="slow"), ok("through"))
    client, sleeps = build(server, llm_max_attempts=2)
    assert client.chat(USER).text == "through" and len(server.requests) == 3


def test_and_the_in_flight_budget_is_not_cut_short_by_a_small_max_attempts():
    server = Server(in_flight(), in_flight(), ok("through"))
    client, _ = build(server, llm_max_attempts=1)
    assert client.chat(USER).text == "through"


def test_a_402_reported_inside_a_200_response_is_classified_the_same_way():
    server = Server(httpx.Response(200, json=IN_FLIGHT_BODY), ok("through"))
    client, sleeps = build(server)
    assert client.chat(USER).text == "through" and sleeps == [4.0]


# ---- out of credit -------------------------------------------------------------------------------------------------------


def test_an_out_of_credit_402_is_not_retried_and_fails_with_one_clear_line():
    server = Server(credit(), ok("never reached"))
    client, sleeps = build(server)
    with pytest.raises(LLMError) as caught:
        client.chat(USER)
    err = caught.value

    assert len(server.requests) == 1 and sleeps == []
    assert err.status_code == 402 and err.retryable is False and err.attempts == 1
    line = str(err)
    assert line.startswith("Out of credit: llm.example.com refused the request (HTTP 402)")
    assert "for the key in AI_API_KEY and model test-model" in line and "another provider" in line
    assert "\n" not in line
    assert "can only afford 248" in err.detail, "the provider's reason is kept in .detail, not in the one-line message"


def test_an_out_of_credit_402_reported_inside_a_200_response_is_not_retried():
    server = Server(httpx.Response(200, json=CREDIT_BODY), ok("never reached"))
    client, sleeps = build(server)
    with pytest.raises(LLMError, match="Out of credit"):
        client.chat(USER)
    assert len(server.requests) == 1 and sleeps == []


@pytest.mark.parametrize(
    "response",
    [httpx.Response(402, text=""), httpx.Response(402, text="Payment Required"), httpx.Response(402, json={"error": {"message": "Insufficient balance"}})],
    ids=["empty", "plain text", "other wording"],
)
def test_any_402_that_does_not_ask_to_retry_is_treated_as_out_of_credit(response):
    server = Server(response, ok("never reached"))
    client, sleeps = build(server)
    with pytest.raises(LLMError, match="Out of credit"):
        client.chat(USER)
    assert len(server.requests) == 1 and sleeps == []


def test_in_flight_wording_without_a_retry_hint_is_not_mistaken_for_a_temporary_budget():
    body = {"error": {"message": "in-flight requests are billed to your account; your balance is 0", "code": 402}}
    server = Server(httpx.Response(402, json=body), ok("never reached"))
    client, _ = build(server)
    with pytest.raises(LLMError, match="Out of credit"):
        client.chat(USER)
    assert len(server.requests) == 1


def test_the_message_shows_the_host_and_port_but_never_credentials_in_the_base_url():
    client, _ = build(Server(credit()), base_url=f"https://user:{KEY}@llm.example.com:8443/v1")
    with pytest.raises(LLMError) as caught:
        client.chat(USER)
    assert "llm.example.com:8443 refused" in str(caught.value) and KEY not in str(caught.value) and "user" not in str(caught.value)


def test_the_key_is_redacted_from_the_message_even_when_it_ends_up_in_the_model_name():
    client, _ = build(Server(credit()), model=f"vendor/{KEY}")
    with pytest.raises(LLMError) as caught:
        client.chat(USER)
    assert KEY not in str(caught.value) and "model vendor/" in str(caught.value)


def test_the_api_key_never_appears_in_either_error_even_if_the_provider_echoes_it():
    echoing = {"error": {"message": f"key {KEY} has no credit. Retry after in-flight requests settle", "code": 402}}
    for response in (httpx.Response(402, json=echoing), httpx.Response(402, json={"error": {"message": f"key {KEY} is broke"}})):
        client, _ = build(Server(response))
        with pytest.raises(LLMError) as caught:
            client.chat(USER)
        assert KEY not in str(caught.value) and KEY not in caught.value.detail


# ---- the rest is unchanged ----------------------------------------------------------------------------------------------


def test_429_and_5xx_still_use_their_own_backoff_and_attempts():
    server = Server(httpx.Response(429, text="slow"), httpx.Response(503, text="down"), httpx.Response(503, text="down"))
    client, sleeps = build(server)
    with pytest.raises(LLMError) as caught:
        client.chat(USER)
    assert len(server.requests) == 3 and len(sleeps) == 2 and caught.value.status_code == 503


def test_other_client_errors_are_still_not_retried():
    server = Server(httpx.Response(401, text="bad key"), ok("never reached"))
    client, sleeps = build(server)
    with pytest.raises(LLMError):
        client.chat(USER)
    assert len(server.requests) == 1 and sleeps == []


# ---- what the person running the harness is told ------------------------------------------------------------------------


def test_the_advice_for_each_kind_of_402():
    out_of_credit = LLMError("Out of credit: x", status_code=402, retryable=False)
    exhausted = LLMError("in-flight budget", status_code=402, retryable=True, attempts=5)
    assert "out of credit" in llm_failure_advice(out_of_credit) and "AI_API_KEY" in llm_failure_advice(out_of_credit)
    assert "in-flight budget stayed exhausted" in llm_failure_advice(exhausted) and "max_output_tokens" in llm_failure_advice(exhausted)


def test_a_run_whose_key_is_out_of_credit_stops_at_once_and_says_so(tmp_path):
    from tests.test_orchestrator import execute

    class OutOfCredit:
        calls = 0

        def chat(self, messages, tools=None):
            OutOfCredit.calls += 1
            raise LLMError(
                "Out of credit: api.example.com refused the request (HTTP 402) for the key in AI_API_KEY and model m. Add credit.",
                status_code=402, retryable=False, attempts=1,
            )

    run = execute([], tmp_path, llm=OutOfCredit())
    assert OutOfCredit.calls == 1, "no retry loop above the client either"
    assert any(e.data["kind"] == "llm" and "Out of credit" in e.data["message"] for e in run.of("error"))
    assert "Stopped early: LLM call failed: Out of credit" in run.report and "add credit" in run.report.lower()
