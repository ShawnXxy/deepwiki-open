"""
Wiki cache operations.

Provides functions for reading, saving, and deleting wiki cache data.
"""

import os
import json
import logging
from typing import Optional

from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.types import WikiCacheIdentifier
from backend.modules.wiki.models import WikiCacheData, WikiCacheRequest

logger = logging.getLogger(__name__)

# Use consistent ~/.adalflow path for Docker volume mounting compatibility
# See backend/utils/paths.py for rationale
from backend.utils.paths import get_wikicache_path


WIKI_CACHE_DIR = get_wikicache_path()
WIKI_CACHE_BLOB_PREFIX = "wikicache"


def get_wiki_cache_filename(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True,
    branch: Optional[str] = None
) -> str:
    """Generates the filename for a given wiki cache (new format with branch suffix).
    
    DEPRECATED: Use WikiCacheIdentifier.get_cache_filename() instead.
    """
    cache_id = WikiCacheIdentifier(
        owner=owner,
        repo=repo,
        repo_type=repo_type,
        language=language,
        comprehensive=comprehensive,
        branch=branch
    )
    return cache_id.get_cache_filename()


def get_wiki_cache_filename_legacy(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True
) -> str:
    """Generates the legacy filename for wiki cache (without branch suffix).
    
    DEPRECATED: Use WikiCacheIdentifier.get_cache_filename_legacy() instead.
    """
    cache_id = WikiCacheIdentifier(
        owner=owner,
        repo=repo,
        repo_type=repo_type,
        language=language,
        comprehensive=comprehensive,
        branch=None
    )
    return cache_id.get_cache_filename_legacy()


def get_wiki_cache_path(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True,
    branch: Optional[str] = None
) -> str:
    """Generates the local file path for a given wiki cache."""
    filename = get_wiki_cache_filename(owner, repo, repo_type, language, comprehensive, branch)
    return os.path.join(WIKI_CACHE_DIR, filename)


def get_wiki_cache_path_legacy(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True
) -> str:
    """Generates the legacy local file path for wiki cache (without branch suffix)."""
    filename = get_wiki_cache_filename_legacy(owner, repo, repo_type, language, comprehensive)
    return os.path.join(WIKI_CACHE_DIR, filename)


def get_wiki_cache_blob_path(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True,
    branch: Optional[str] = None
) -> str:
    """Generates the blob path for a given wiki cache."""
    filename = get_wiki_cache_filename(owner, repo, repo_type, language, comprehensive, branch)
    return f"{WIKI_CACHE_BLOB_PREFIX}/{filename}"


def get_wiki_cache_blob_path_legacy(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True
) -> str:
    """Generates the legacy blob path for wiki cache (without branch suffix)."""
    filename = get_wiki_cache_filename_legacy(owner, repo, repo_type, language, comprehensive)
    return f"{WIKI_CACHE_BLOB_PREFIX}/{filename}"


async def read_wiki_cache(
    owner: str,
    repo: str,
    repo_type: str,
    language: str,
    comprehensive: bool = True,
    branch: Optional[str] = None
) -> Optional[WikiCacheData]:
    """
    Reads wiki cache data from storage with backward compatibility.
    
    First tries the new format with branch suffix, then falls back to legacy format.
    """
    # Try Azure Blob Storage when configured
    if is_blob_storage_configured():
        blob_path = get_wiki_cache_blob_path(owner, repo, repo_type, language, comprehensive, branch)
        blob_path_legacy = get_wiki_cache_blob_path_legacy(owner, repo, repo_type, language, comprehensive)
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                error_msg = "Azure Blob Storage is configured but failed to create client."
                logger.error(error_msg)
                raise ConnectionError(error_msg)
            
            # Try new format first (with branch suffix)
            if blob_client.exists(blob_path):
                logger.info(f"Reading wiki cache from Azure Blob Storage: {blob_path}")
                content = blob_client.download_text(blob_path)
                if content:
                    data = json.loads(content)
                    return WikiCacheData(**data)
            
            # Fall back to legacy format
            if blob_client.exists(blob_path_legacy):
                logger.info(f"Reading wiki cache from legacy blob path: {blob_path_legacy}")
                content = blob_client.download_text(blob_path_legacy)
                if content:
                    data = json.loads(content)
                    return WikiCacheData(**data)
            
            logger.info(f"Wiki cache not found in blob storage")
            return None
        except ConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to read wiki cache from Azure Blob Storage: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    # Local storage mode
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language, comprehensive, branch)
    cache_path_legacy = get_wiki_cache_path_legacy(owner, repo, repo_type, language, comprehensive)
    
    # Try new format first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Read wiki cache from: {cache_path}")
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
    
    # Fall back to legacy format
    if os.path.exists(cache_path_legacy):
        try:
            with open(cache_path_legacy, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Read wiki cache from legacy path: {cache_path_legacy}")
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from legacy path {cache_path_legacy}: {e}")
            return None
    
    return None


async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """
    Saves wiki cache data to storage.
    
    Supports partial/checkpoint saves when is_partial=True.
    """
    payload = WikiCacheData(
        wiki_structure=data.wiki_structure,
        generated_pages=data.generated_pages,
        repo=data.repo,
        provider=data.provider,
        model=data.model,
        comprehensive=data.comprehensive,
        is_partial=data.is_partial
    )
    
    # Log size of data to be cached
    try:
        payload_json = payload.model_dump_json()
        payload_size = len(payload_json.encode('utf-8'))
        partial_status = "PARTIAL" if data.is_partial else "COMPLETE"
        pages_count = len(data.generated_pages)
        total_pages = len(data.wiki_structure.pages) if data.wiki_structure.pages else 0
        logger.info(f"Payload prepared for caching. Size: {payload_size} bytes. Status: {partial_status}. Pages: {pages_count}/{total_pages}")
    except Exception as ser_e:
        logger.warning(f"Could not serialize payload for size logging: {ser_e}")
    
    # Try Azure Blob Storage when configured
    if is_blob_storage_configured():
        blob_path = get_wiki_cache_blob_path(
            data.repo.owner, data.repo.repo, data.repo.type,
            data.language, data.comprehensive, data.repo.branch
        )
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                error_msg = "Azure Blob Storage is configured but failed to create client."
                logger.error(error_msg)
                raise ConnectionError(error_msg)
            
            logger.info(f"Saving wiki cache to Azure Blob Storage: {blob_path}")
            content = json.dumps(payload.model_dump(), indent=2)
            if blob_client.upload_text(blob_path, content):
                logger.info(f"Wiki cache successfully saved to blob: {blob_path}")
                return True
            else:
                error_msg = f"Failed to save wiki cache to Azure Blob Storage: {blob_path}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
        except ConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to save wiki cache to Azure Blob Storage: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    # Local storage mode
    cache_path = get_wiki_cache_path(
        data.repo.owner, data.repo.repo, data.repo.type,
        data.language, data.comprehensive, data.repo.branch
    )
    logger.info(f"Attempting to save wiki cache locally. Path: {cache_path}")
    try:
        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False
