"""Detect a repository's languages, install/test commands and produce a compact map of it."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RepoProfile:
    """What the harness needs to know to build and test a repo."""

    languages: list[str]
    primary_language: str
    install_cmd: str | None
    test_cmd: str | None
    test_framework: str | None
    notes: str = ""


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def _has(root: Path, *filenames: str) -> Path | None:
    """Return the path of the first matching file at *root*, or None."""
    for name in filenames:
        p = root / name
        if p.exists():
            return p
    return None


def _read_text(path: Path) -> str:
    """Read text from *path* silently returning '' on any error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _detect_python(root: Path) -> tuple[str | None, str | None, str | None]:
    """Return (install_cmd, test_cmd, test_framework) for Python repos."""
    pyproject = root / "pyproject.toml"
    setup_py = root / "setup.py"
    setup_cfg = root / "setup.cfg"
    req = _has(root, "requirements.txt", "requirements-dev.txt", "requirements_dev.txt")
    req_test = _has(root, "requirements-test.txt", "requirements_test.txt")

    has_pyproject = pyproject.exists()
    has_setup = setup_py.exists() or setup_cfg.exists()

    if not (has_pyproject or has_setup or req):
        return None, None, None

    # Install command
    if has_pyproject:
        text = _read_text(pyproject)
        # Check for [dev] or [test] extras
        if re.search(r'\[(?:project\.optional-dependencies|tool\.poetry\.dev-dependencies)\]', text):
            install_cmd = "pip install -e '.[dev]'"
        else:
            install_cmd = "pip install -e ."
    elif has_setup:
        install_cmd = "pip install -e ."
    elif req:
        install_cmd = f"pip install -r {req.name}"
    else:
        install_cmd = None

    if req_test:
        install_cmd = (install_cmd or "") + f" && pip install -r {req_test.name}"

    # Test framework — prefer pytest, fallback to unittest
    framework = "pytest"
    test_cmd = "pytest"
    if has_pyproject:
        text = _read_text(pyproject)
        if "pytest" in text:
            framework, test_cmd = "pytest", "pytest"
        elif "unittest" in text:
            framework, test_cmd = "unittest", "python -m unittest discover"
    elif req:
        text = _read_text(req)
        if "pytest" not in text:
            framework, test_cmd = "unittest", "python -m unittest discover"

    return install_cmd, test_cmd, framework


def _detect_js(root: Path) -> tuple[str | None, str | None, str | None] | None:
    """Return (install_cmd, test_cmd, test_framework) for JS/TS repos, or None."""
    pkg_json = root / "package.json"
    if not pkg_json.exists():
        return None

    try:
        pkg = json.loads(_read_text(pkg_json))
    except json.JSONDecodeError:
        return None

    # Determine package manager from lockfile
    if (root / "yarn.lock").exists():
        pm = "yarn"
        install_cmd = "yarn install"
    elif (root / "pnpm-lock.yaml").exists():
        pm = "pnpm"
        install_cmd = "pnpm install"
    else:
        pm = "npm"
        install_cmd = "npm install"

    # Test command from scripts.test
    scripts = pkg.get("scripts", {})
    raw_test = scripts.get("test", "")
    if raw_test and raw_test != "echo \"Error: no test specified\" && exit 1":
        test_cmd = f"{pm} test" if pm != "npm" else "npm test"
    else:
        test_cmd = f"{pm} test"

    # Detect framework from devDependencies / dependencies
    all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    if "jest" in all_deps:
        framework = "jest"
    elif "vitest" in all_deps:
        framework = "vitest"
    elif "mocha" in all_deps:
        framework = "mocha"
    else:
        framework = None

    return install_cmd, test_cmd, framework


def _detect_go(root: Path) -> tuple[str | None, str | None, str | None] | None:
    """Return (install_cmd, test_cmd, test_framework) for Go repos, or None."""
    if not (root / "go.mod").exists():
        return None
    return "go mod download", "go test ./...", "go test"


def _detect_rust(root: Path) -> tuple[str | None, str | None, str | None] | None:
    """Return (install_cmd, test_cmd, test_framework) for Rust repos, or None."""
    if not (root / "Cargo.toml").exists():
        return None
    return "cargo build", "cargo test", "cargo test"


def _detect_java(root: Path) -> tuple[str | None, str | None, str | None] | None:
    """Return (install_cmd, test_cmd, test_framework) for Java repos, or None."""
    if (root / "pom.xml").exists():
        return "mvn -q package -DskipTests", "mvn -q test", "junit"
    gradle = _has(root, "build.gradle", "build.gradle.kts")
    if gradle:
        wrapper = "./gradlew" if (root / "gradlew").exists() else "gradle"
        return f"{wrapper} build -x test", f"{wrapper} test", "junit"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_repo(root: Path) -> RepoProfile:
    """Inspect *root* and infer languages, install command and test command.

    Detection order: Python → JS/TS → Go → Rust → Java.
    Multiple languages can be detected; the primary is the first detected.
    """
    languages: list[str] = []
    install_cmd: str | None = None
    test_cmd: str | None = None
    test_framework: str | None = None
    notes_parts: list[str] = []

    # --- Python ---
    py_install, py_test, py_fw = _detect_python(root)
    if py_install or py_test:
        languages.append("python")
        if install_cmd is None:
            install_cmd = py_install
            test_cmd = py_test
            test_framework = py_fw

    # --- JavaScript / TypeScript ---
    js_result = _detect_js(root)
    if js_result is not None:
        js_install, js_test, js_fw = js_result
        lang = "typescript" if list(root.rglob("*.ts")) else "javascript"
        if lang not in languages:
            languages.append(lang)
        if install_cmd is None:
            install_cmd = js_install
            test_cmd = js_test
            test_framework = js_fw

    # --- Go ---
    go_result = _detect_go(root)
    if go_result is not None:
        if "go" not in languages:
            languages.append("go")
        if install_cmd is None:
            install_cmd, test_cmd, test_framework = go_result

    # --- Rust ---
    rust_result = _detect_rust(root)
    if rust_result is not None:
        if "rust" not in languages:
            languages.append("rust")
        if install_cmd is None:
            install_cmd, test_cmd, test_framework = rust_result

    # --- Java ---
    java_result = _detect_java(root)
    if java_result is not None:
        if "java" not in languages:
            languages.append("java")
        if install_cmd is None:
            install_cmd, test_cmd, test_framework = java_result

    if not languages:
        languages = ["unknown"]
        notes_parts.append("No recognised language manifest found.")

    primary = languages[0]
    return RepoProfile(
        languages=languages,
        primary_language=primary,
        install_cmd=install_cmd,
        test_cmd=test_cmd,
        test_framework=test_framework,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Repo map
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    ".git", "node_modules", "vendor", "venv", ".venv", "dist", "build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "target",
    ".gradle", ".idea", ".vscode",
}

# Priority source directories — shown first in the map
_PRIORITY_DIRS = {"src", "lib", "pkg", "cmd", "app", "core", "internal"}

# Symbol extraction regexes per extension
_SYMBOL_RE: dict[str, re.Pattern] = {
    ".py":   re.compile(r"^(?:def|class|async def)\s+(\w+)", re.MULTILINE),
    ".ts":   re.compile(r"^(?:export\s+)?(?:function|class|const|interface|type)\s+(\w+)", re.MULTILINE),
    ".js":   re.compile(r"^(?:export\s+)?(?:function|class|const)\s+(\w+)", re.MULTILINE),
    ".go":   re.compile(r"^(?:func|type|var|const)\s+(\w+)", re.MULTILINE),
    ".rs":   re.compile(r"^(?:pub\s+)?(?:fn|struct|enum|trait|impl|type)\s+(\w+)", re.MULTILINE),
    ".java": re.compile(r"^(?:public\s+|private\s+|protected\s+)?(?:class|interface|enum)\s+(\w+)", re.MULTILINE),
}

_MAX_SYMBOLS_PER_FILE = 8
_MAX_FILE_SIZE_BYTES = 256 * 1024  # skip large generated files


def _extract_symbols(path: Path) -> list[str]:
    """Return top-level symbol names from *path* using language-specific regex."""
    pattern = _SYMBOL_RE.get(path.suffix)
    if pattern is None:
        return []
    try:
        if path.stat().st_size > _MAX_FILE_SIZE_BYTES:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        return pattern.findall(text)[:_MAX_SYMBOLS_PER_FILE]
    except OSError:
        return []


def _walk_tree(
    root: Path,
    current: Path,
    lines: list[str],
    indent: int,
    char_budget: list[int],
) -> None:
    """Recursively walk *current* and append formatted lines to *lines*."""
    try:
        entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return

    for entry in entries:
        if char_budget[0] <= 0:
            return

        if entry.is_dir():
            if entry.name in _SKIP_DIRS:
                continue
            line = "  " * indent + f"{entry.name}/"
            lines.append(line)
            char_budget[0] -= len(line) + 1
            _walk_tree(root, entry, lines, indent + 1, char_budget)

        elif entry.is_file():
            symbols = _extract_symbols(entry)
            sym_str = f"  [{', '.join(symbols)}]" if symbols else ""
            line = "  " * indent + entry.name + sym_str
            lines.append(line)
            char_budget[0] -= len(line) + 1


def repo_map(root: Path, max_chars: int = 6000) -> str:
    """Return a compact file tree with top-level symbols, truncated to *max_chars*.

    Priority directories (src/, lib/, pkg/, cmd/, ...) are rendered first so the
    most relevant code appears before the budget is exhausted.
    """
    lines: list[str] = [f"# repo: {root.name}", ""]
    char_budget = [max_chars - len(f"# repo: {root.name}\n\n")]

    try:
        top_entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return f"# repo: {root.name}\n(permission denied reading root)"

    # Priority dirs first
    for entry in top_entries:
        if entry.is_dir() and entry.name in _PRIORITY_DIRS and entry.name not in _SKIP_DIRS:
            if char_budget[0] <= 0:
                break
            line = f"{entry.name}/"
            lines.append(line)
            char_budget[0] -= len(line) + 1
            _walk_tree(root, entry, lines, 1, char_budget)

    # Then everything else
    for entry in top_entries:
        if char_budget[0] <= 0:
            break
        name = entry.name
        if entry.is_dir():
            if name in _SKIP_DIRS or name in _PRIORITY_DIRS:
                continue
            line = f"{name}/"
            lines.append(line)
            char_budget[0] -= len(line) + 1
            _walk_tree(root, entry, lines, 1, char_budget)
        else:
            symbols = _extract_symbols(entry)
            sym_str = f"  [{', '.join(symbols)}]" if symbols else ""
            line = name + sym_str
            lines.append(line)
            char_budget[0] -= len(line) + 1

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars - 20] + "\n... [truncated]"
    return result
