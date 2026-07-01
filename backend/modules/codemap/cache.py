"""
CodeMap cache operations.

Provides functions for reading, saving, and deleting codemap data.
Stores in ~/.adalflow/codemap/ (local) or codemap/ prefix (blob).
"""

import json
import logging
import os
from typing import Optional

from backend.clients.blob_client import (
    get_blob_storage_client,
    is_blob_storage_configured,
)
from backend.modules.codemap.models import CodeMapData
from backend.paths import get_codemap_path
from backend.utils.filter import sanitize_branch_for_path

logger = logging.getLogger(__name__)

CODEMAP_CACHE_DIR = get_codemap_path()
CODEMAP_BLOB_PREFIX = "codemap"


def get_codemap_filename(
    owner: str,
    repo: str,
    repo_type: str,
    branch: Optional[str] = None,
) -> str:
    """Generate the filename for a codemap cache file.

    Branch names may contain ``/`` (e.g. ``rel/latest``); we sanitize them
    so the filename stays a single component and does not leak into a
    sub-folder under the blob prefix.
    """
    branch_suffix = f"_{sanitize_branch_for_path(branch, default='default')}"
    return f"codemap_{repo_type}_{owner}_{repo}{branch_suffix}.json"


def get_codemap_cache_path(
    owner: str,
    repo: str,
    repo_type: str,
    branch: Optional[str] = None,
) -> str:
    """Generate the local file path for a codemap cache."""
    filename = get_codemap_filename(owner, repo, repo_type, branch)
    return os.path.join(CODEMAP_CACHE_DIR, filename)


def get_codemap_blob_path(
    owner: str,
    repo: str,
    repo_type: str,
    branch: Optional[str] = None,
) -> str:
    """Generate the blob storage path for a codemap cache."""
    filename = get_codemap_filename(owner, repo, repo_type, branch)
    return f"{CODEMAP_BLOB_PREFIX}/{filename}"


async def read_codemap_cache(
    owner: str,
    repo: str,
    repo_type: str,
    branch: Optional[str] = None,
) -> Optional[CodeMapData]:
    """Read codemap data from storage (blob or local disk)."""
    if is_blob_storage_configured():
        blob_path = get_codemap_blob_path(
            owner, repo, repo_type, branch,
        )
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                logger.error(
                    "Blob storage configured but client unavailable"
                )
                return None
            if blob_client.exists(blob_path):
                content = blob_client.download_text(blob_path)
                if content:
                    data = json.loads(content)
                    logger.info(
                        f"Read codemap from blob: {blob_path}"
                    )
                    return CodeMapData(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to read codemap from blob: {e}")
            return None

    # Local storage
    cache_path = get_codemap_cache_path(
        owner, repo, repo_type, branch,
    )
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Read codemap from: {cache_path}")
            return CodeMapData(**data)
        except Exception as e:
            logger.error(
                f"Error reading codemap from {cache_path}: {e}"
            )
    return None


def save_codemap_cache(
    data: CodeMapData,
    owner: str,
    repo: str,
    repo_type: str,
    branch: Optional[str] = None,
) -> bool:
    """Save codemap data to storage (blob or local disk).

    Memory note: this used to peak at ``3x`` JSON size on the blob path
    (``model_dump()`` dict + ``json.dumps()`` string + SDK upload buffer).
    Pydantic v2's ``model_dump_json()`` skips the intermediate dict, so the
    blob path now peaks at ``2x`` size. The local path streams via
    ``json.dump(..., f)`` and is unchanged.
    """
    if is_blob_storage_configured():
        blob_path = get_codemap_blob_path(
            owner, repo, repo_type, branch,
        )
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                logger.error(
                    "Blob storage configured but client unavailable"
                )
                return False
            # Single-step Pydantic -> bytes; skips the intermediate dict copy
            # that ``data.model_dump()`` would create. For 200 MB codemaps
            # this saves ~200 MB at peak.
            content_bytes = data.model_dump_json().encode('utf-8')
            del data  # safe: data already serialised
            if blob_client.upload_bytes(blob_path, content_bytes):
                logger.info(
                    f"Codemap saved to blob: {blob_path} "
                    f"({len(content_bytes)} bytes)"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save codemap to blob: {e}")
            return False

    # Local storage -- already streaming via json.dump(payload, f, ...)
    payload = data.model_dump()
    del data  # Free Pydantic object before building JSON string

    cache_path = get_codemap_cache_path(
        owner, repo, repo_type, branch,
    )
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, separators=(',', ':'))
        del payload
        logger.info(f"Codemap saved to: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save codemap to {cache_path}: {e}")
        return False


async def delete_codemap_cache(
    owner: str,
    repo: str,
    repo_type: str,
    branch: Optional[str] = None,
) -> bool:
    """Delete a codemap cache file."""
    if is_blob_storage_configured():
        blob_path = get_codemap_blob_path(
            owner, repo, repo_type, branch,
        )
        try:
            blob_client = get_blob_storage_client()
            if blob_client and blob_client.exists(blob_path):
                blob_client.delete(blob_path)
                logger.info(f"Deleted codemap from blob: {blob_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete codemap from blob: {e}")
        return False

    cache_path = get_codemap_cache_path(
        owner, repo, repo_type, branch,
    )
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Deleted codemap: {cache_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete codemap: {e}")
    return False
