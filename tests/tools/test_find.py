import pytest

from anvil.tools.find import FindSymbolTool, FindReferencesTool
from anvil.sandbox.worktree import WorktreeSandbox
from anvil.sandbox.base import Sandbox

def test_find_symbol(tmp_path):
    src = """\
class Foo:
    def bar(self):
        pass

def baz():
    pass
"""
    (tmp_path / "test.py").write_text(src)
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "test.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    sb = WorktreeSandbox(tmp_path, tmp_path / "wt")
    tool = FindSymbolTool()
    
    res = tool.run({"name": "baz"}, sb)
    assert res.ok
    assert "test.py:5:def baz():" in res.output

def test_find_references(tmp_path):
    src = """\
def baz():
    pass

baz()
baz()
"""
    (tmp_path / "test.py").write_text(src)
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "test.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    sb = WorktreeSandbox(tmp_path, tmp_path / "wt")
    tool = FindReferencesTool()
    
    res = tool.run({"name": "baz"}, sb)
    assert res.ok
    assert "test.py:1:def baz():" in res.output
    assert "test.py:4:baz()" in res.output
