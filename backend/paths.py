"""
Storage layout for DeepWiki.

Defines where all persistent data lives on disk:
    ~/.adalflow/wikicache/       — Generated wiki JSON cache files
    ~/.adalflow/repos/           — Cloned git repositories
    ~/.adalflow/vectors/         — Embedding vector JSON chunks
    ~/.adalflow/embedding_cache/ — Embedding API response cache

Uses ~/.adalflow consistently on all platforms (Windows, Linux, macOS)
to ensure Docker volume mounting works correctly.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Use ~/.adalflow consistently on all platforms
# This ensures Docker volume mounting works correctly
# (Docker mounts ~/.adalflow:/root/.adalflow)
_ADALFLOW_ROOT = os.path.join(os.path.expanduser("~"), ".adalflow")


def get_adalflow_root_path() -> str:
    """
    Get the root path for adalflow data storage.

    Uses ~/.adalflow consistently on all platforms (Windows, Linux, macOS).
    This differs from adalflow's default which uses %APPDATA%/adalflow on Windows.

    We use this consistent path to ensure Docker volume mounting works
    correctly across all environments.

    Returns:
        str: The absolute path to ~/.adalflow
    """
    # Ensure directory exists
    if not os.path.exists(_ADALFLOW_ROOT):
        os.makedirs(_ADALFLOW_ROOT, exist_ok=True)
        logger.debug(f"Created adalflow root directory: {_ADALFLOW_ROOT}")

    return _ADALFLOW_ROOT


def get_wikicache_path() -> str:
    """Get the path for wiki cache storage."""
    path = os.path.join(get_adalflow_root_path(), "wikicache")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_repos_path() -> str:
    """Get the path for cloned repositories."""
    path = os.path.join(get_adalflow_root_path(), "repos")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_vectors_path() -> str:
    """Get the path for vector embeddings storage."""
    path = os.path.join(get_adalflow_root_path(), "vectors")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_codemap_path() -> str:
    """Get the path for codemap graph storage."""
    path = os.path.join(get_adalflow_root_path(), "codemap")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_embedding_cache_path() -> str:
    """Get the path for embedding cache."""
    path = os.path.join(get_adalflow_root_path(), "embedding_cache")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
