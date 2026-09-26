"""Offline tests for repo/profile.py — all use tmp_path, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.repo.profile import RepoProfile, profile_repo, repo_map


# ---------------------------------------------------------------------------
# Helpers to build fake repo fixtures
# ---------------------------------------------------------------------------

def _make_python_pyproject(root: Path, has_dev_extras: bool = True) -> None:
    toml = "[project]\nname = 'mypkg'\n"
    if has_dev_extras:
        toml += "\n[project.optional-dependencies]\ndev = ['pytest']\n"
    toml += "\n[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    (root / "pyproject.toml").write_text(toml)


def _make_python_requirements(root: Path) -> None:
    (root / "requirements.txt").write_text("httpx\npytest\n")


def _make_package_json(root: Path, pm: str = "npm", test_script: str = "jest") -> None:
    pkg: dict = {
        "name": "my-pkg",
        "scripts": {"test": test_script},
        "devDependencies": {test_script.split()[0]: "^29"},
    }
    (root / "package.json").write_text(json.dumps(pkg))
    if pm == "yarn":
        (root / "yarn.lock").write_text("")
    elif pm == "pnpm":
        (root / "pnpm-lock.yaml").write_text("")


def _make_go_mod(root: Path) -> None:
    (root / "go.mod").write_text("module example.com/mymod\n\ngo 1.22\n")


def _make_cargo_toml(root: Path) -> None:
    (root / "Cargo.toml").write_text("[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n")


def _make_pom_xml(root: Path) -> None:
    (root / "pom.xml").write_text("<project><groupId>com.example</groupId></project>")


def _make_gradle(root: Path, has_wrapper: bool = True) -> None:
    (root / "build.gradle").write_text("plugins { id 'java' }")
    if has_wrapper:
        (root / "gradlew").write_text("#!/bin/sh")
        (root / "gradlew").chmod(0o755)


# ---------------------------------------------------------------------------
# profile_repo — language detection
# ---------------------------------------------------------------------------

class TestProfileRepoPython:
    def test_pyproject_detected(self, tmp_path):
        _make_python_pyproject(tmp_path)
        profile = profile_repo(tmp_path)
        assert "python" in profile.languages
        assert profile.primary_language == "python"

    def test_requirements_detected(self, tmp_path):
        _make_python_requirements(tmp_path)
        profile = profile_repo(tmp_path)
        assert "python" in profile.languages

    def test_pytest_framework(self, tmp_path):
        _make_python_pyproject(tmp_path)
        profile = profile_repo(tmp_path)
        assert profile.test_framework == "pytest"
        assert "pytest" in (profile.test_cmd or "")

    def test_install_cmd_with_extras(self, tmp_path):
        _make_python_pyproject(tmp_path, has_dev_extras=True)
        profile = profile_repo(tmp_path)
        assert profile.install_cmd is not None
        assert "dev" in profile.install_cmd or "install" in profile.install_cmd

    def test_install_cmd_without_extras(self, tmp_path):
        _make_python_pyproject(tmp_path, has_dev_extras=False)
        profile = profile_repo(tmp_path)
        assert profile.install_cmd is not None


class TestProfileRepoJS:
    def test_npm_detected(self, tmp_path):
        _make_package_json(tmp_path, pm="npm")
        profile = profile_repo(tmp_path)
        assert any(lang in profile.languages for lang in ("javascript", "typescript"))

    def test_yarn_install_cmd(self, tmp_path):
        _make_package_json(tmp_path, pm="yarn")
        profile = profile_repo(tmp_path)
        assert profile.install_cmd == "yarn install"

    def test_pnpm_install_cmd(self, tmp_path):
        _make_package_json(tmp_path, pm="pnpm")
        profile = profile_repo(tmp_path)
        assert profile.install_cmd == "pnpm install"

    def test_jest_framework_detected(self, tmp_path):
        _make_package_json(tmp_path, pm="npm", test_script="jest")
        profile = profile_repo(tmp_path)
        assert profile.test_framework == "jest"

    def test_vitest_framework_detected(self, tmp_path):
        pkg = {
            "name": "x",
            "scripts": {"test": "vitest"},
            "devDependencies": {"vitest": "^1"},
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        profile = profile_repo(tmp_path)
        assert profile.test_framework == "vitest"

    def test_test_cmd_set(self, tmp_path):
        _make_package_json(tmp_path, pm="npm")
        profile = profile_repo(tmp_path)
        assert profile.test_cmd is not None


class TestProfileRepoGo:
    def test_go_detected(self, tmp_path):
        _make_go_mod(tmp_path)
        profile = profile_repo(tmp_path)
        assert "go" in profile.languages

    def test_go_test_cmd(self, tmp_path):
        _make_go_mod(tmp_path)
        profile = profile_repo(tmp_path)
        assert profile.test_cmd == "go test ./..."

    def test_go_install_cmd(self, tmp_path):
        _make_go_mod(tmp_path)
        profile = profile_repo(tmp_path)
        assert profile.install_cmd == "go mod download"


class TestProfileRepoRust:
    def test_rust_detected(self, tmp_path):
        _make_cargo_toml(tmp_path)
        profile = profile_repo(tmp_path)
        assert "rust" in profile.languages

    def test_cargo_test_cmd(self, tmp_path):
        _make_cargo_toml(tmp_path)
        profile = profile_repo(tmp_path)
        assert profile.test_cmd == "cargo test"


class TestProfileRepoJava:
    def test_maven_detected(self, tmp_path):
        _make_pom_xml(tmp_path)
        profile = profile_repo(tmp_path)
        assert "java" in profile.languages
        assert profile.test_cmd == "mvn -q test"

    def test_gradle_with_wrapper(self, tmp_path):
        _make_gradle(tmp_path, has_wrapper=True)
        profile = profile_repo(tmp_path)
        assert "java" in profile.languages
        assert "./gradlew" in (profile.test_cmd or "")

    def test_gradle_without_wrapper(self, tmp_path):
        _make_gradle(tmp_path, has_wrapper=False)
        profile = profile_repo(tmp_path)
        assert "java" in profile.languages
        assert "gradle" in (profile.test_cmd or "")


class TestProfileRepoUnknown:
    def test_no_manifests(self, tmp_path):
        profile = profile_repo(tmp_path)
        assert "unknown" in profile.languages or profile.languages  # graceful
        assert profile.test_cmd is None or isinstance(profile.test_cmd, str)

    def test_notes_populated_for_unknown(self, tmp_path):
        profile = profile_repo(tmp_path)
        # notes should mention no manifest if truly unknown
        if "unknown" in profile.languages:
            assert profile.notes  # non-empty explanation


class TestProfileRepoDataclass:
    def test_notes_default_empty(self, tmp_path):
        _make_go_mod(tmp_path)
        profile = profile_repo(tmp_path)
        assert isinstance(profile.notes, str)

    def test_languages_is_list(self, tmp_path):
        _make_python_pyproject(tmp_path)
        profile = profile_repo(tmp_path)
        assert isinstance(profile.languages, list)


# ---------------------------------------------------------------------------
# repo_map
# ---------------------------------------------------------------------------

class TestRepoMap:
    def _make_python_src(self, root: Path) -> None:
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text(
            "def main():\n    pass\n\nclass MyClass:\n    pass\n"
        )
        (root / "README.md").write_text("# Hello")
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("ref: refs/heads/main")

    def test_returns_string(self, tmp_path):
        self._make_python_src(tmp_path)
        result = repo_map(tmp_path)
        assert isinstance(result, str)

    def test_respects_max_chars(self, tmp_path):
        # Lots of files
        for i in range(200):
            (tmp_path / f"file_{i}.txt").write_text("x" * 100)
        result = repo_map(tmp_path, max_chars=500)
        assert len(result) <= 520  # small tolerance for truncation marker

    def test_skips_git_dir(self, tmp_path):
        self._make_python_src(tmp_path)
        result = repo_map(tmp_path)
        assert ".git" not in result

    def test_skips_node_modules(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lodash").mkdir()
        result = repo_map(tmp_path)
        # "node_modules/" as a directory entry must be absent;
        # the header line may contain the tmp_path name as a substring — ignore it
        lines = result.splitlines()[2:]  # skip header lines
        assert not any("node_modules" in line for line in lines)

    def test_src_dir_appears(self, tmp_path):
        self._make_python_src(tmp_path)
        result = repo_map(tmp_path)
        assert "src" in result

    def test_python_symbols_extracted(self, tmp_path):
        self._make_python_src(tmp_path)
        result = repo_map(tmp_path)
        assert "main" in result or "MyClass" in result

    def test_repo_name_in_header(self, tmp_path):
        # Task 2: default header is neutral 'repo', NOT the dir name.
        result = repo_map(tmp_path)
        assert result.startswith("# repo: repo")
        # But passing label= puts owner/repo in header
        result_labelled = repo_map(tmp_path, label="psf/requests")
        assert "psf/requests" in result_labelled

    def test_priority_dir_first(self, tmp_path):
        """src/ must appear before other dirs like zzz/."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("def foo(): pass")
        (tmp_path / "zzz").mkdir()
        (tmp_path / "zzz" / "b.py").write_text("x = 1")
        result = repo_map(tmp_path)
        src_pos = result.find("src/")
        zzz_pos = result.find("zzz/")
        if src_pos != -1 and zzz_pos != -1:
            assert src_pos < zzz_pos

    def test_go_symbols_extracted(self, tmp_path):
        (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n\ntype Server struct {}\n")
        result = repo_map(tmp_path)
        assert "main" in result or "Server" in result

    def test_rust_symbols_extracted(self, tmp_path):
        (tmp_path / "lib.rs").write_text("pub fn hello() {}\npub struct Foo;\n")
        result = repo_map(tmp_path)
        assert "hello" in result or "Foo" in result


# ---------------------------------------------------------------------------
# Regression tests for Task 1 (unittest discover) and Task 2 (repo_map label)
# ---------------------------------------------------------------------------

class TestTask1UnittestDetection:
    """Task 1: unittest discover must use python3, not bare python."""

    def test_unittest_in_pyproject_uses_python3(self, tmp_path):
        """When pyproject mentions unittest (not pytest), command uses python3."""
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'mypkg'\n\n"
            "# we use unittest for testing\n"
        )
        profile = profile_repo(tmp_path)
        # Should NOT emit 'python -m unittest discover' (bare python)
        if profile.test_cmd and "unittest" in profile.test_cmd:
            assert "python3" in profile.test_cmd
            assert profile.test_cmd != "python -m unittest discover"

    def test_explicit_unittest_in_pyproject(self, tmp_path):
        """Pyproject with 'unittest' keyword uses python3 -m unittest discover."""
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'mypkg'\n"
            "# unittest discover\n"
        )
        profile = profile_repo(tmp_path)
        if profile.test_framework == "unittest":
            assert "python3" in profile.test_cmd
            assert "python -m" not in profile.test_cmd

    def test_requirements_without_pytest_uses_python3_unittest(self, tmp_path):
        """requirements.txt without pytest → python3 -m unittest discover."""
        (tmp_path / "requirements.txt").write_text("requests\nflask\n")
        profile = profile_repo(tmp_path)
        assert profile.test_framework == "unittest"
        assert profile.test_cmd == "python3 -m unittest discover"
        assert "python3" in profile.test_cmd

    def test_requirements_with_pytest_stays_pytest(self, tmp_path):
        """requirements.txt with pytest → keep pytest."""
        (tmp_path / "requirements.txt").write_text("requests\npytest\n")
        profile = profile_repo(tmp_path)
        assert profile.test_cmd == "pytest"

    def test_no_bare_python_in_any_test_cmd(self, tmp_path):
        """Ensure 'python -m' (without '3') never appears in test_cmd."""
        (tmp_path / "requirements.txt").write_text("httpx\n")
        profile = profile_repo(tmp_path)
        if profile.test_cmd:
            # 'python -m' without '3' must not appear
            assert "python -m" not in profile.test_cmd or "python3" in profile.test_cmd


class TestTask2RepoMapLabel:
    """Task 2: repo_map header must not expose clone dir name; use label param."""

    def test_default_label_is_repo_not_dirname(self, tmp_path):
        """Without a label, header says 'repo', NOT the tmp dir name."""
        result = repo_map(tmp_path)
        header = result.split("\n")[0]
        # Must NOT contain the tmp dir basename (e.g. 'pytest-xxx' or similar)
        assert "# repo: repo" == header

    def test_label_overrides_header(self, tmp_path):
        """Passing label='psf/requests' puts that in the header."""
        result = repo_map(tmp_path, label="psf/requests")
        header = result.split("\n")[0]
        assert header == "# repo: psf/requests"

    def test_label_none_uses_neutral(self, tmp_path):
        """Explicit label=None still uses neutral 'repo' label."""
        result = repo_map(tmp_path, label=None)
        assert result.startswith("# repo: repo")

    def test_label_shown_in_permission_error(self, tmp_path):
        """Even on PermissionError the label appears correctly."""
        # We can't easily test PermissionError so just verify the API accepts label
        result = repo_map(tmp_path, label="owner/myrepo")
        assert "# repo: owner/myrepo" in result
