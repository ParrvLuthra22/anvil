"""python -m anvil.llm.probe against a scripted provider: what it reports, what it exits with, and that it never prints the key."""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from anvil.llm.probe import main, safe_url

KEY = "sk-probe-SECRET-9f8e7d6c5b4a"
CALL = '```json\n{"tool": "add", "args": {"a": 17, "b": 25}}\n```'
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)


@pytest.fixture
def config_file(tmp_path):
    def write(**overrides) -> Path:
        config = {"model": "test-model", "base_url": "https://llm.example.com/v1", "temperature": 0, **overrides}
        config = {k: v for k, v in config.items() if v is not ...}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    return write


def message(content=None, tool_calls=None, usage=None, **extra):
    body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content, **extra}, "finish_reason": "stop"}]}
    if tool_calls:
        body["choices"][0]["message"]["tool_calls"] = tool_calls
    body["usage"] = usage or USAGE
    return httpx.Response(200, json=body)


def native_add(a=17, b=25, name="add"):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps({"a": a, "b": b})}}


class Provider:
    """A scripted endpoint. Answers turn 1 with a tool call and turn 2 with the result, natively or as text."""

    def __init__(self, *, native=True, text=True, think=False, usage=None, reject=None, first=None, second=None):
        self.native, self.text, self.think, self.usage = native, text, think, usage
        self.reject, self.first, self.second = reject or {}, first, second
        self.bodies: list[dict] = []

    def wrap(self, content):
        return f"<think>Let me work out what to do.</think>\n{content}" if self.think else content

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        for field, response in self.reject.items():
            if field in body:
                return response
        extra = {"reasoning_content": "First I need the add tool."} if self.think else {}
        last = body["messages"][-1]
        if "tools" in body:
            if not self.native:
                return httpx.Response(400, json={"error": {"message": "tools are not supported by this model"}})
            if last["role"] == "tool":
                return self.second or message(self.wrap("The result is 42."), usage=self.usage, **extra)
            return self.first or message(self.wrap(""), [native_add()], usage=self.usage, **extra)
        if not self.text:
            return httpx.Response(500, json={"error": {"message": "text mode is broken here"}})
        if last["content"].startswith("[result of add]"):
            return self.second or message(self.wrap("The answer is 42."), usage=self.usage, **extra)
        return self.first or message(self.wrap(CALL), usage=self.usage, **extra)


def squash(text: str) -> str:
    """Collapse runs of spaces, so a test checks what is said, not how the columns are padded."""
    return "\n".join(" ".join(line.split()) for line in text.splitlines())


def run(config_file, provider, *args, **config):
    lines: list[str] = []
    code = main(["--config", str(config_file(**config)), "--attempts", "1", *args], transport=httpx.MockTransport(provider), out=lines.append)
    return code, "\n".join(lines)


# ---- a healthy endpoint ---------------------------------------------------------------------------------


def test_both_modes_work_and_the_report_says_so(config_file):
    provider = Provider()
    code, out = run(config_file, provider)
    assert code == 0
    assert "== tool mode: native ==" in out and "== tool mode: text ==" in out
    assert out.count("result: WORKS") == 2
    assert "turn 1  ok" in out and "add(a=17, b=25)" in out
    assert "tokens: 100 prompt + 20 completion = 120" in out
    assert " ms " in out
    assert "Recommended tool_mode: native   (text also works, as a fallback)" in out
    assert "reasoning seen: no" in out and "worked around: nothing" in out


def test_the_second_turn_proves_the_history_round_trips_in_native_mode(config_file):
    provider = Provider()
    run(config_file, provider, "--mode", "native")
    assert len(provider.bodies) == 2
    history = provider.bodies[1]["messages"]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool"]
    assert history[2]["tool_calls"][0]["id"] == history[3]["tool_call_id"] == "call_1"
    assert history[3]["content"] == "42"
    assert {t["function"]["name"] for t in provider.bodies[0]["tools"]} == {"add", "shout"}


def test_a_single_mode_can_be_chosen(config_file):
    code, out = run(config_file, Provider(), "--mode", "text")
    assert code == 0 and "tool mode: native" not in out and "Recommended tool_mode: text" in out


def test_auto_mode_reports_where_it_settled(config_file):
    code, out = run(config_file, Provider(native=False), "--mode", "auto")
    assert code == 0 and "auto mode settled on: text" in out


# ---- endpoints with problems -----------------------------------------------------------------------------


def test_an_endpoint_without_tool_support_fails_native_and_recommends_text(config_file):
    code, out = run(config_file, Provider(native=False))
    assert code == 0
    assert "HTTP 400" in out and "tools are not supported" in out
    native, text = out.split("== tool mode: text ==")
    assert "DOES NOT WORK" in native and "result: WORKS" in text
    assert "Recommended tool_mode: text" in out and "native FAILS" in squash(out)


def test_when_nothing_works_the_exit_status_says_so(config_file):
    code, out = run(config_file, lambda request: httpx.Response(401, json={"error": {"message": "invalid api key"}}))
    assert code == 1
    assert "No tool mode worked" in out and out.count("DOES NOT WORK") == 2 and "HTTP 401" in out


def test_a_wrong_tool_is_reported_and_turn_two_is_not_attempted(config_file):
    provider = Provider(first=message("", [native_add(name="shout")]))
    code, out = run(config_file, provider, "--mode", "native")
    assert code == 1 and "called shout(" in out and "instead of add" in out
    assert "turn 2" not in out and len(provider.bodies) == 1


def test_wrong_arguments_are_reported(config_file):
    code, out = run(config_file, Provider(first=message("", [native_add(a=1, b=2)])), "--mode", "native")
    assert code == 1 and "add called with the wrong arguments" in out


def test_arguments_that_are_not_numbers_do_not_crash_the_probe(config_file):
    code, out = run(config_file, Provider(first=message("", [native_add(a="seventeen", b=None)])), "--mode", "native")
    assert code == 1 and "wrong arguments" in out and "Traceback" not in out


def test_numbers_sent_as_numeric_strings_are_accepted(config_file):
    code, out = run(config_file, Provider(first=message("", [native_add(a="17", b="25")])), "--mode", "native")
    assert code == 0


def test_a_model_that_answers_in_prose_instead_of_calling_is_reported(config_file):
    code, out = run(config_file, Provider(first=message("17 plus 25 is 42.")), "--mode", "native")
    assert code == 1 and "no tool call; the model said: 17 plus 25 is 42." in out


def test_a_model_that_calls_a_tool_again_instead_of_answering_still_proves_the_round_trip(config_file):
    code, out = run(config_file, Provider(second=message("", [native_add()])), "--mode", "native")
    assert code == 0 and "the model called add again" in out


def test_a_wrong_number_in_the_final_answer_is_noted_but_the_round_trip_still_worked(config_file):
    code, out = run(config_file, Provider(second=message("It is 41.")), "--mode", "native")
    assert code == 0 and "no 42 in answer: It is 41." in out


def test_an_empty_final_reply_is_a_failure(config_file):
    code, out = run(config_file, Provider(second=message("")), "--mode", "native")
    assert code == 1 and "the reply was empty" in out


# ---- reasoning and quirks ----------------------------------------------------------------------------------


def test_reasoning_output_is_detected_counted_and_reported(config_file):
    usage = {**USAGE, "completion_tokens_details": {"reasoning_tokens": 15}}
    code, out = run(config_file, Provider(think=True, usage=usage), "--mode", "native")
    assert code == 0
    assert "reasoning seen: yes, in 2 of 2 replies" in out
    assert "(of which 15 reasoning)" in out
    assert "Reasoning appeared; it is removed" in out


def test_the_answer_shown_is_free_of_the_reasoning(config_file):
    code, out = run(config_file, Provider(think=True), "--mode", "native")
    assert "correct answer: The result is 42." in out and "<think>" not in out and "First I need" not in out


def test_a_reasoning_model_at_temperature_zero_gets_a_warning(config_file):
    _, out = run(config_file, Provider(think=True), model="deepseek-reasoner", temperature=0)
    assert "profile: deepseek-reasoning" in squash(out)
    assert "WARNING: temperature is 0 with a reasoning model" in out and "0.6" in out


def test_no_warning_when_the_temperature_is_left_to_the_profile(config_file):
    _, out = run(config_file, Provider(think=True), model="deepseek-reasoner", temperature=...)
    assert "WARNING" not in out and "temperature: 0.6" in squash(out)


def test_no_warning_for_a_plain_model(config_file):
    _, out = run(config_file, Provider())
    assert "WARNING" not in out


def test_a_400_the_client_worked_around_is_listed(config_file):
    reject = {
        "temperature": httpx.Response(
            400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0 with this model.", "param": "temperature"}}
        )
    }
    code, out = run(config_file, Provider(reject=reject), "--mode", "native")
    assert code == 0
    assert "worked around:" in out and "dropped parameter 'temperature' (HTTP 400: " in out
    assert "The endpoint rejected some request fields" in out


def test_the_header_shows_the_effective_settings(config_file):
    _, out = run(config_file, Provider(), model="qwen-plus", llm_extra_params={"top_p": 0.8})
    flat = squash(out)
    assert "model: qwen-plus" in flat and "profile: qwen" in flat
    assert "max_output: 8192" in flat and "extra params: top_p" in flat
    assert "endpoint: https://llm.example.com/v1" in flat


# ---- the key ------------------------------------------------------------------------------------------------


def test_the_key_never_appears_even_when_the_provider_echoes_it(config_file):
    echo = httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {KEY}"}})
    _, out = run(config_file, lambda request: echo)
    assert KEY not in out and "[REDACTED]" in out


def test_the_key_never_appears_in_a_success_run_either(config_file):
    _, out = run(config_file, Provider(think=True))
    assert KEY not in out and KEY[:12] not in out


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://llm.example.com/v1", "https://llm.example.com/v1"),
        ("https://user:hunter2@llm.example.com/v1", "https://llm.example.com/v1"),
        ("https://llm.example.com/v1?api_key=SECRET&x=1", "https://llm.example.com/v1"),
        ("http://localhost:8000/v1/#frag", "http://localhost:8000/v1/"),
        ("https://u:p@host:8443/a?k=v#f", "https://host:8443/a"),
    ],
)
def test_the_base_url_is_shown_without_credentials_or_query(url, expected):
    assert safe_url(url) == expected


def test_a_base_url_with_a_key_in_it_is_not_printed(config_file):
    _, out = run(config_file, Provider(), base_url="https://llm.example.com/v1?api_key=URLSECRET")
    assert "URLSECRET" not in out and "api_key" not in out


# ---- setup problems -----------------------------------------------------------------------------------------------


def test_a_missing_key_is_a_friendly_setup_error(config_file, monkeypatch):
    monkeypatch.delenv("AI_API_KEY")
    code, out = run(config_file, Provider())
    assert code == 2 and "AI_API_KEY is not set" in out and "Traceback" not in out


def test_a_missing_config_file_is_a_setup_error(tmp_path):
    lines: list[str] = []
    assert main(["--config", str(tmp_path / "nope.yaml")], out=lines.append) == 2
    assert "Cannot read the config file" in "\n".join(lines)


def test_an_invalid_config_is_a_setup_error(config_file):
    code, out = run(config_file, Provider(), base_url="ftp://nope")
    assert code == 2 and "base_url must start with http" in out


def test_a_key_with_a_space_in_it_is_a_setup_error_that_does_not_echo_it(config_file, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "sk-bad key")
    code, out = run(config_file, Provider())
    assert code == 2 and "sk-bad" not in out


# ---- as a real command ----------------------------------------------------------------------------------------------


def _probe(tmp_path, *args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "anvil.llm.probe", *args], capture_output=True, text=True, timeout=60, cwd=tmp_path,
        env={**os.environ, "AI_API_KEY": KEY, **(env or {})},
    )


def test_the_module_runs_as_a_command(tmp_path, config_file):
    help_result = _probe(tmp_path, "--help")
    assert help_result.returncode == 0 and "--mode" in help_result.stdout


def test_an_unreachable_endpoint_gives_a_clean_report_not_a_traceback(tmp_path, config_file):
    path = config_file(base_url="http://127.0.0.1:1/v1")
    result = _probe(tmp_path, "--config", str(path), "--mode", "native", "--attempts", "1")
    assert result.returncode == 1
    assert "Traceback" not in result.stdout + result.stderr and "network error" in result.stdout
    assert KEY not in result.stdout + result.stderr
