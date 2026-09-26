import pytest

from anvil.tools.related import RelatedTestsTool
from anvil.sandbox.worktree import WorktreeSandbox
from anvil.repo.profile import RepoProfile

class TestRelatedTestsTool:
    def test_run_success_python(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, check=True)
        sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
        
        # Create some source files and test files
        sandbox.write_file("src/foo.py", "def foo(): pass\n")
        sandbox.write_file("src/bar.py", "def bar(): pass\n")
        sandbox.write_file("tests/test_foo.py", "import foo\ndef test_foo(): pass\n")
        sandbox.write_file("tests/test_other.py", "import bar\ndef test_other(): pass\n")

        # Commit so files exist for sandbox.exec
        sandbox.exec("git add .")
        sandbox.exec("git config user.email 'test@example.com'")
        sandbox.exec("git config user.name 'Test'")
        sandbox.exec("git commit -m 'initial'")

        tool = RelatedTestsTool()
        result = tool.run({"paths": ["src/foo.py", "src/bar.py"]}, sandbox)

        assert result.ok
        assert "tests/test_foo.py" in result.output
        assert "tests/test_other.py" in result.output

    def test_run_success_go(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, check=True)
        sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
        sandbox.write_file("pkg/server/server.go", "package server\n")
        sandbox.write_file("pkg/server/server_test.go", "package server\n")

        sandbox.exec("git add .")
        sandbox.exec("git config user.email 'test@example.com'")
        sandbox.exec("git config user.name 'Test'")
        sandbox.exec("git commit -m 'go files'")

        tool = RelatedTestsTool()
        result = tool.run({"paths": ["pkg/server/server.go"]}, sandbox)

        assert result.ok
        assert "./pkg/server" in result.output

    def test_no_paths_provided(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, check=True)
        sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
        tool = RelatedTestsTool()
        result = tool.run({}, sandbox)
        assert not result.ok
        assert "'paths' must be a non-empty list" in result.output

    def test_no_related_tests(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, check=True)
        sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
        sandbox.write_file("src/isolated.py", "x = 1\n")
        sandbox.exec("git add .")
        sandbox.exec("git config user.email 'test@example.com'")
        sandbox.exec("git config user.name 'Test'")
        sandbox.exec("git commit -m 'iso'")

        tool = RelatedTestsTool()
        result = tool.run({"paths": ["src/isolated.py"]}, sandbox)
        assert result.ok
        assert "No related tests found" in result.output
