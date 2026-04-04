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
from backend.paths import get_wikicache_path


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
        is_partial=data.is_partial,
        commit_hash=data.commit_hash,
        indexed_at=data.indexed_at,
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
        import tempfile
        logger.info(f"Writing cache file to: {cache_path}")
        cache_dir = os.path.dirname(cache_path)
        # Write to temp file first, then atomic rename to prevent corruption
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload.model_dump(), f, indent=2)
            # Atomic rename (same filesystem guaranteed by mkstemp in same dir)
            os.replace(tmp_path, cache_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False


# Cache filename pattern: deepwiki_cache_{type}_{owner}_{repo}_{lang}_{mode}[_{branch}].json
import re
_CACHE_PATTERN = re.compile(
    r'^deepwiki_cache_(\w+)_(.+?)_([^_]+)_([a-z]+(?:-[a-z]+)*)_'
    r'(comprehensive|concise)(?:_(.+))?\.json$'
)


async def list_wiki_caches() -> list:
    """List all processed wiki projects from storage (blob or local).

    Returns a list of dicts with project metadata extracted from filenames.
    Uses blob last_modified or file mtime for submittedAt.
    """
    projects = []

    if is_blob_storage_configured():
        blob_client = get_blob_storage_client()
        if blob_client:
            try:
                blobs = blob_client.list_blobs_with_metadata(
                    prefix=f"{WIKI_CACHE_BLOB_PREFIX}/deepwiki_cache_"
                )
                for blob_info in blobs:
                    name = blob_info['name']
                    filename = name.split('/')[-1] if '/' in name else name
                    mtime = blob_info.get('last_modified', 0)
                    _parse_cache_filename(filename, projects, mtime)
            except Exception as e:
                logger.error(f"Error listing wiki caches from blob: {e}")
    else:
        if os.path.exists(WIKI_CACHE_DIR):
            for name in os.listdir(WIKI_CACHE_DIR):
                if name.startswith('deepwiki_cache_') and name.endswith('.json'):
                    mtime = 0
                    try:
                        stat = os.stat(os.path.join(WIKI_CACHE_DIR, name))
                        mtime = int(stat.st_mtime * 1000)
                    except OSError:
                        pass
                    _parse_cache_filename(name, projects, mtime)

    projects.sort(key=lambda p: p.get('submittedAt', 0), reverse=True)
    return projects


def _parse_cache_filename(
    filename: str, projects: list, mtime_ms: int = 0
) -> None:
    """Parse a cache filename and append project metadata to the list."""
    match = _CACHE_PATTERN.match(filename)
    if not match:
        return
    repo_type, owner, repo, language, mode, branch = match.groups()
    projects.append({
        'id': filename.replace('.json', ''),
        'owner': owner,
        'repo': repo,
        'name': f"{owner}/{repo}",
        'repo_type': repo_type,
        'submittedAt': mtime_ms,
        'language': language,
        'comprehensive': mode == 'comprehensive',
        'branch': branch or None,
    })
