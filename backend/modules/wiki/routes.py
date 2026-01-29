"""
Wiki-related API routes.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE
from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.modules.wiki.models import (
    WikiCacheData,
    WikiCacheRequest,
    WikiExportRequest,
    ProcessedProjectEntry,
)
from backend.modules.wiki.cache import (
    read_wiki_cache,
    save_wiki_cache,
    get_wiki_cache_blob_path,
    get_wiki_cache_path,
    WIKI_CACHE_DIR,
    WIKI_CACHE_BLOB_PREFIX,
)
from backend.modules.wiki.export import generate_markdown_export, generate_json_export

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type"),
    language: str = Query(..., description="Language of the wiki content"),
    comprehensive: bool = Query(True, description="Whether this is comprehensive or concise wiki"),
    branch: Optional[str] = Query(None, description="Branch name")
):
    """Retrieves cached wiki data for a repository."""
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}, comprehensive: {comprehensive}, branch: {branch}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language, comprehensive, branch)
    if cached_data:
        return cached_data
    else:
        logger.info(f"Wiki cache not found for {owner}/{repo}")
        return None


@router.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """Stores generated wiki data to the server-side cache."""
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")


@router.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type"),
    language: str = Query(..., description="Language of the wiki content"),
    comprehensive: bool = Query(True, description="Whether this is comprehensive or concise wiki"),
    branch: Optional[str] = Query(None, description="Branch name"),
    authorization_code: Optional[str] = Query(None, description="Authorization code")
):
    """Deletes a specific wiki cache."""
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if not authorization_code or WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Attempting to delete wiki cache for {owner}/{repo}")

    try:
        # Try Azure Blob Storage when configured
        if is_blob_storage_configured():
            blob_path = get_wiki_cache_blob_path(owner, repo, repo_type, language, comprehensive, branch)
            try:
                blob_client = get_blob_storage_client()
                if not blob_client:
                    error_msg = "Azure Blob Storage is configured but failed to create client."
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                
                if blob_client.exists(blob_path):
                    if blob_client.delete(blob_path):
                        logger.info(f"Successfully deleted wiki cache from blob: {blob_path}")
                        mode = "comprehensive" if comprehensive else "concise"
                        return {"message": f"Wiki cache for {owner}/{repo} ({language}, {mode}) deleted successfully"}
                    else:
                        error_msg = f"Failed to delete wiki cache from blob: {blob_path}"
                        logger.error(error_msg)
                        raise ConnectionError(error_msg)
                else:
                    logger.warning(f"Wiki cache not found in blob storage: {blob_path}")
                    raise HTTPException(status_code=404, detail="Wiki cache not found")
            except ConnectionError:
                raise
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Failed to delete wiki cache from Azure Blob Storage: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e
        
        # Local storage mode
        cache_path = get_wiki_cache_path(owner, repo, repo_type, language, comprehensive, branch)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info(f"Successfully deleted wiki cache: {cache_path}")
                mode = "comprehensive" if comprehensive else "concise"
                return {"message": f"Wiki cache for {owner}/{repo} ({language}, {mode}) deleted successfully"}
            except Exception as e:
                logger.error(f"Error deleting wiki cache {cache_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
        else:
            logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
            raise HTTPException(status_code=404, detail="Wiki cache not found")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """Export wiki content as Markdown or JSON."""
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """Lists all processed projects found in the wiki cache."""
    project_entries: List[ProcessedProjectEntry] = []

    def parse_cache_filename(filename: str, last_modified_ms: int) -> Optional[ProcessedProjectEntry]:
        """Parse a cache filename into a ProcessedProjectEntry."""
        base_filename = os.path.basename(filename)
        if not (base_filename.startswith("deepwiki_cache_") and base_filename.endswith(".json")):
            return None
        
        parts = base_filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')
        
        # New format with branch: repo_type_owner_repo_language_mode_branch (6+ parts)
        if len(parts) >= 6 and parts[-2] in ("comprehensive", "concise"):
            repo_type = parts[0]
            owner = parts[1]
            branch = parts[-1]
            mode = parts[-2]
            language = parts[-3]
            repo = "_".join(parts[2:-3])
            is_comprehensive = (mode == "comprehensive")
            return ProcessedProjectEntry(
                id=base_filename,
                owner=owner,
                repo=repo,
                name=f"{owner}/{repo}",
                repo_type=repo_type,
                submittedAt=last_modified_ms,
                language=language,
                comprehensive=is_comprehensive,
                branch=branch
            )
        
        # Legacy format without branch: repo_type_owner_repo_language_mode (5+ parts)
        if len(parts) >= 5 and parts[-1] in ("comprehensive", "concise"):
            repo_type = parts[0]
            owner = parts[1]
            mode = parts[-1]
            language = parts[-2]
            repo = "_".join(parts[2:-2])
            is_comprehensive = (mode == "comprehensive")
            return ProcessedProjectEntry(
                id=base_filename,
                owner=owner,
                repo=repo,
                name=f"{owner}/{repo}",
                repo_type=repo_type,
                submittedAt=last_modified_ms,
                language=language,
                comprehensive=is_comprehensive,
                branch="default"
            )
        return None

    try:
        # Try Azure Blob Storage when configured
        if is_blob_storage_configured():
            try:
                blob_client = get_blob_storage_client()
                if not blob_client:
                    error_msg = "Azure Blob Storage is configured but failed to create client."
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                
                logger.debug(f"Scanning for project cache files in Azure Blob Storage: {WIKI_CACHE_BLOB_PREFIX}/")
                blobs = blob_client.list_blobs_with_metadata(prefix=f"{WIKI_CACHE_BLOB_PREFIX}/")
                
                for blob_info in blobs:
                    entry = parse_cache_filename(blob_info["name"], blob_info["last_modified"])
                    if entry:
                        project_entries.append(entry)
                    else:
                        logger.warning(f"Could not parse project details from blob: {blob_info['name']}")
                
                project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
                logger.debug(f"Found {len(project_entries)} processed project entries from Azure Blob Storage.")
                return project_entries
            except ConnectionError:
                raise
            except Exception as e:
                error_msg = f"Failed to list processed projects from Azure Blob Storage: {e}"
                logger.error(error_msg, exc_info=True)
                raise ConnectionError(error_msg) from e
        
        # Local storage mode
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files locally in: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR)

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path)
                    last_modified_ms = int(stats.st_mtime * 1000)
                    entry = parse_cache_filename(filename, last_modified_ms)
                    if entry:
                        project_entries.append(entry)
                    else:
                        logger.warning(f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing processed projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects from server cache.")
