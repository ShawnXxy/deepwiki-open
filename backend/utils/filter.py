"""
Centralized file filtering utilities.

Provides .gitignore parsing and helpers used by document processing
and wiki generation to avoid hardcoded filter lists.
"""
import os
import re
import logging
from typing import Optional

import pathspec

logger = logging.getLogger(__name__)


def sanitize_branch_for_path(
    branch: Optional[str],
    default: str = 'main',
) -> str:
    """Sanitize a branch name for use in filesystem paths and blob prefixes.

    Replaces characters that are problematic in paths (dots, slashes,
    backslashes, colons, spaces, etc.) with dashes. Result is safe for
    local paths, blob keys, and storage filenames.

    This is the single source of truth for branch sanitisation across
    the project; all path/filename construction sites that include the
    branch must funnel through this helper so that:

    * paths produced by different components (repo folder, vectors
      blob, codemap blob, wikicache blob, AI Search data source) stay
      consistent for the same branch, and
    * branch refs containing ``/`` (e.g. ``rel/latest``) cannot leak a
      separator into a path and create unintended sub-folders.

    Args:
        branch: Raw branch name (e.g., ``mysql_8.4``, ``rel/latest``).
            ``None`` / empty / whitespace-only inputs return ``default``.
        default: Fallback value used when ``branch`` is empty after
            stripping/sanitising. Defaults to ``'main'``.

    Returns:
        Sanitised string safe for paths
        (e.g., ``mysql_8-4``, ``rel-latest``).
    """
    if not branch or not str(branch).strip():
        return default
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '-', branch.strip())
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    return sanitized or default


def load_gitignore(repo_path: str) -> Optional[pathspec.PathSpec]:
    """Load and compile the root .gitignore from a cloned repository.

    Args:
        repo_path: Path to the cloned repository root.

    Returns:
        Compiled PathSpec for matching, or None if no .gitignore found.
    """
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if not os.path.isfile(gitignore_path):
        logger.info(f"No .gitignore found at {repo_path}")
        return None

    try:
        with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        pattern_count = len([line for line in lines if line.strip() and not line.startswith("#")])
        logger.info(f"Loaded .gitignore with {pattern_count} patterns from {repo_path}")
        return spec
    except Exception as e:
        logger.warning(f"Failed to parse .gitignore at {repo_path}: {e}")
        return None


def is_gitignored(spec: Optional[pathspec.PathSpec], relative_path: str) -> bool:
    """Check if a file path matches the .gitignore spec.

    Args:
        spec: Compiled PathSpec from load_gitignore(), or None.
        relative_path: File path relative to repo root.

    Returns:
        True if the file should be ignored.
    """
    if spec is None:
        return False
    normalized = relative_path.replace("\\", "/")
    return spec.match_file(normalized)
