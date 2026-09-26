"""Sandbox package: factory function to pick the right sandbox backend."""

from __future__ import annotations

from pathlib import Path


def make_sandbox(config: dict, repo_root: Path, profile=None):
    """Create and return the appropriate sandbox for *repo_root*.

    Selection logic (matches ``config.yaml`` ``sandbox:`` key):

    * ``"worktree"`` → :class:`~anvil.sandbox.worktree.WorktreeSandbox` always.
    * ``"docker"``   → :class:`~anvil.sandbox.docker.DockerSandbox` always
      (caller must ensure Docker is available).
    * ``"auto"``     → Docker if ``docker info`` succeeds; else WorktreeSandbox.

    Args:
        config:    Dict with at least ``sandbox`` (str) and optionally
                   ``tool_output_char_cap`` (int) and ``docker_images`` (dict).
        repo_root: The cloned repository directory returned by ``clone_repo``.
        profile:   Optional :class:`~anvil.repo.profile.RepoProfile` — used to
                   pick the Docker image for the primary language.

    Returns:
        An object implementing :class:`~anvil.sandbox.base.Sandbox`.
    """
    from anvil.sandbox.worktree import WorktreeSandbox
    from anvil.sandbox.docker import DockerSandbox, _docker_available

    backend: str = config.get("sandbox", "auto")
    char_cap: int = config.get("tool_output_char_cap", 8000)
    image_overrides: dict = config.get("docker_images", {})

    # Determine work directory (sibling to repo_root)
    work_dir = repo_root.parent / f"{repo_root.name}_sandbox"

    primary_language = (profile.primary_language if profile else "python")

    if backend == "worktree":
        return WorktreeSandbox(repo_root=repo_root, work_dir=work_dir, char_cap=char_cap)

    if backend == "docker":
        return DockerSandbox(
            repo_root=repo_root,
            work_dir=work_dir,
            primary_language=primary_language,
            image_overrides=image_overrides,
            char_cap=char_cap,
        )

    # "auto"
    if _docker_available():
        return DockerSandbox(
            repo_root=repo_root,
            work_dir=work_dir,
            primary_language=primary_language,
            image_overrides=image_overrides,
            char_cap=char_cap,
        )

    return WorktreeSandbox(repo_root=repo_root, work_dir=work_dir, char_cap=char_cap)
