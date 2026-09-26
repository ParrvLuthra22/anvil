import pytest

from anvil.tools.outline import OutlineTool
from anvil.sandbox.worktree import WorktreeSandbox
import subprocess

def test_outline_python(tmp_path):
    src = """\
class Foo:
    def bar(self):
        pass

def baz():
    pass
"""
    (tmp_path / "test.py").write_text(src)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "test.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
    tool = OutlineTool()
    
    res = tool.run({"path": "test.py"}, sandbox)
    assert res.ok
    assert "1: class Foo" in res.output
    assert "2: def bar(...)" in res.output
    assert "5: def baz(...)" in res.output

def test_outline_regex_fallback(tmp_path):
    src = """\
function foo() {
    return 1;
}
class Bar {
    method() {}
}
"""
    (tmp_path / "test.js").write_text(src)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "test.js"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
    tool = OutlineTool()
    
    res = tool.run({"path": "test.js"}, sandbox)
    assert res.ok
    assert "1: function foo() {" in res.output
    assert "4: class Bar {" in res.output

def test_outline_no_file(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True)
    sandbox = WorktreeSandbox(tmp_path, tmp_path / "wt")
    tool = OutlineTool()
    res = tool.run({"path": "missing.py"}, sandbox)
    assert not res.ok
