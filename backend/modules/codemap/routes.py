"""
FastAPI routes for codemap operations.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.modules.codemap.cache import read_codemap_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["codemap"])


@router.get("/api/codemap")
async def get_codemap(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query("azuredevops", description="Repository type"),
    branch: Optional[str] = Query(None, description="Branch name"),
):
    """Read codemap from cache (blob or local disk)."""
    data = await read_codemap_cache(
        owner=owner,
        repo=repo,
        repo_type=repo_type,
        branch=branch,
    )
    if data:
        return JSONResponse(content=data.model_dump())
    return JSONResponse(
        content={"error": "Codemap not found"},
        status_code=404,
    )
