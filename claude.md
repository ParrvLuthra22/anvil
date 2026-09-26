# PROJECT: ANVIL — autonomous coding-agent harness (AI Harness Hackathon 2026)

## What we are building
A harness around a fixed, text-only foundation model that acts like a software engineer. Given a GitHub issue URL, it clones the repo, understands the issue, navigates the code, uses tools, manages context, recovers from failures, and produces a verified patch — using tokens efficiently. We are NOT building an app; we are building the system that lets the model behave like an engineer. Multi-language repos (Python, JS/TS, Go, Rust, Java) must be supported.

## Judging
Round 1: AI evaluation — the evaluator runs our repo unattended and feeds it a prescribed GitHub issue. Round 2: faculty review of ARCHITECTURE and CODE QUALITY. So: correctness and stability first, clean architecture and docs second.

## Hard requirements from the organisers (do not violate)
- Repo root has a Makefile with: `make setup` (install everything), `make run` (launch the TUI harness), `make test`, `make clean`. `make setup && make run` MUST work from a fresh clone with no manual steps.
- The API key comes ONLY from the env var `AI_API_KEY`. NEVER hard-code or commit keys/tokens/passwords anywhere (source, Makefile, .env, docs, config). Only `.env.example` with `AI_API_KEY=` empty is allowed.
- Text-only models. Model name/base URL must be defined in `config.yaml`; the organisers may prescribe a model later, so changing it must never need a code edit.
- The TUI must launch from `make run` with no extra commands. Evaluator flow: `export AI_API_KEY=...; make setup; make run;` then the issue is supplied. The harness must never crash: every run ends with a patch and a report even on failure.

## Team and ownership (each person edits ONLY their directories)
- Parrv (lead/integrator): src/anvil/llm/, src/anvil/agent/, src/anvil/context/, src/anvil/events.py, config.yaml, pyproject.toml
- Akshat: src/anvil/repo/, src/anvil/tools/, src/anvil/sandbox/, tests for those
- Sneha: Makefile, src/anvil/tui/, src/anvil/trace/, src/anvil/__main__.py, bench/, tests/mock_llm.py, README.md, docs/
Git: one branch per person (parrv/…, akshat/…, sneha/…); rebase on main and merge to main about every hour; small commits; never force-push main. Feature freeze 12:30 AM; final tag by 2:00 AM.

## Tech stack
Python 3.11, Textual (TUI), httpx (HTTP), pyyaml, pytest, rich. ripgrep (`rg`) preferred for search, falling back to grep. macOS is the dev machine (M3 8GB dev, M4 16GB final test). Docker is OPTIONAL (auto-detected); the default sandbox is a plain-subprocess + git-worktree sandbox.

## Configuration contract
config.yaml keys: model, base_url, temperature (0), max_steps_per_phase, max_total_steps, max_tokens_total, wall_clock_seconds, tool_output_char_cap, sandbox (auto|worktree|docker). Env overrides: AI_API_KEY (required, secret), AI_BASE_URL, AI_MODEL.
The LLM is OpenAI-compatible (chat completions). During development we use a free provider (Gemini via its OpenAI-compatible endpoint, Groq, or OpenRouter free models); the code must be provider-agnostic.

## Entry points (contract)
- `python -m anvil` → TUI (this is what `make run` runs). Accepts an issue URL typed in the TUI, or `--issue <url>`.
- `python -m anvil --issue <url> --headless` → no TUI, same pipeline, prints result. Also accepts `--repo <url> --issue-text "<text>"` as a fallback if the GitHub issue fetch fails.
- `python -m anvil replay <trace.jsonl>` → replay a recorded run in the TUI without any API calls.
- Every run writes `output/patch.diff`, `output/report.md`, and `output/trace.jsonl`.

## The agent cycle (phases)
INGEST → PROFILE → UNDERSTAND → LOCALIZE → REPRODUCE → PATCH → VERIFY → REVIEW → FINALIZE.
Reproduce-first (write a failing repro before patching), verify-last (re-run repro + relevant tests), self-review the diff, roll back to a checkpoint after repeated failed patches. Budgets on steps, tokens and wall-clock; a loop detector; timeouts and output caps on every shell command.

## INTERFACES (the contract — do not rename or change signatures; if you need a change, write "CONTRACT CHANGE REQUEST: …" and tell the human instead of editing it)
```python
# src/anvil/events.py
class Phase(str, Enum):
    INGEST="ingest"; PROFILE="profile"; UNDERSTAND="understand"; LOCALIZE="localize"
    REPRODUCE="reproduce"; PATCH="patch"; VERIFY="verify"; REVIEW="review"; FINALIZE="finalize"

@dataclass
class AgentEvent:
    ts: float
    type: str            # "phase" | "message" | "tool_call" | "tool_result" | "llm_usage" | "error" | "done"
    phase: Phase | None
    data: dict           # phase: {"name"}; message: {"role","text"}; tool_call: {"tool","args"};
                         # tool_result: {"tool","ok","output_preview"}; llm_usage: {"prompt_tokens","completion_tokens","total_tokens","cost_estimate"};
                         # error: {"kind","message"}; done: {"resolved_confidence","patch_path","report_path","steps","tokens","seconds"}

class EventBus:
    def emit(self, event: AgentEvent) -> None: ...
    def subscribe(self) -> "asyncio.Queue[AgentEvent]": ...

# src/anvil/sandbox/base.py
@dataclass
class ExecResult:
    exit_code: int; stdout: str; stderr: str; timed_out: bool; duration: float

class Sandbox(Protocol):
    root: Path                                   # repo working directory
    def exec(self, cmd: str, timeout: int = 120) -> ExecResult: ...
    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str: ...  # 1-indexed inclusive line range
    def write_file(self, path: str, content: str) -> None: ...
    def diff(self) -> str: ...                   # unified diff of all changes vs baseline
    def checkpoint(self, label: str) -> str: ... # returns a ref id
    def rollback(self, ref: str) -> None: ...
    def close(self) -> None: ...

# src/anvil/tools/base.py
@dataclass
class ToolResult:
    ok: bool; output: str; meta: dict = field(default_factory=dict)

class Tool(Protocol):
    name: str
    description: str
    parameters: dict                             # JSON Schema for args
    def run(self, args: dict, sandbox: Sandbox) -> ToolResult: ...
# tools/registry.py:  class ToolRegistry: register(tool); get(name); schemas() -> list[dict]
# Required tools: list_dir, grep, read_file(path,start,end), edit_file(path,old,new) [exact str_replace; on mismatch return the closest matching lines],
#                 run_cmd(cmd,timeout), run_tests(target=None), git_diff()

# src/anvil/repo/ingest.py
@dataclass
class IssueRef:
    owner: str; repo: str; number: int; url: str; title: str = ""; body: str = ""; comments: list[str] = field(default_factory=list)
def parse_issue_url(url: str) -> IssueRef: ...   # no network
def fetch_issue(ref: IssueRef) -> IssueRef: ...  # GitHub REST API, unauthenticated, handle 403/404/rate-limits
def clone_repo(ref: IssueRef, dest: Path) -> Path: ...  # shallow clone, returns repo root

# src/anvil/repo/profile.py
@dataclass
class RepoProfile:
    languages: list[str]; primary_language: str; install_cmd: str | None; test_cmd: str | None; test_framework: str | None; notes: str = ""
def profile_repo(root: Path) -> RepoProfile: ...
def repo_map(root: Path, max_chars: int = 6000) -> str: ...  # compact tree + top-level symbols

# src/anvil/trace/recorder.py
class TraceRecorder:
    def __init__(self, path: Path): ...
    def record(self, event: AgentEvent) -> None: ...   # append one JSON line, flush
    @staticmethod
    def load(path: Path) -> Iterator[AgentEvent]: ...

# src/anvil/llm/client.py   (Parrv owns; others only use it via the mock)
@dataclass
class LLMResponse:
    text: str; tool_calls: list[dict]; usage: dict
class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...

# tests/mock_llm.py (Sneha owns)
class MockLLM(LLMClient):  # replays a scripted list of LLMResponse objects in order
    def __init__(self, script: list[LLMResponse]): ...
```

## Working rules for you, the AI agent
1. Work ONLY inside the directories your human owns. Do not edit others' files or the interfaces above.
2. Every module gets pytest tests that run offline (no network, no real LLM) using fakes/mocks.
3. Keep code typed, small and documented (docstrings on public functions); prefer the standard library; no heavy frameworks.
4. Never write secrets. Never commit .env. Read the key only via os.environ["AI_API_KEY"].
5. Every shell command has a timeout and an output cap. Fail soft: return a ToolResult(ok=False, …) instead of raising.
6. Work in small steps, run the tests after each step, and commit after each passing step with a clear message.
7. When done with a step, tell the human what you built, how to run its tests, and anything that deviates from the contract.