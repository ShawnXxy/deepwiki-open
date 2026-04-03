"""
FastAPI routes for codetrace operations.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.modules.codetrace.models import CodeTraceRequest
from backend.modules.codetrace.service import generate_code_trace

logger = logging.getLogger(__name__)

router = APIRouter(tags=["codetrace"])


@router.post("/api/codetrace")
async def create_code_trace(request: CodeTraceRequest):
    """Generate a code trace for a question about a repository."""
    import asyncio
    from backend.modules.embedder.retriever import RAG

    try:
        # Prepare RAG retriever
        rag = RAG(provider='azure')
        await asyncio.to_thread(
            rag.prepare_retriever,
            repo_url_or_path=request.repo_url,
            type=request.type or 'azuredevops',
            access_token=request.token,
            branch=request.branch,
        )

        # Extract repo name from URL
        from urllib.parse import unquote, urlparse
        url = unquote(request.repo_url.rstrip('/'))
        parts = [p for p in urlparse(url).path.split('/') if p]
        repo_name = parts[-1] if parts else 'repo'

        # Generate trace
        result = await asyncio.to_thread(
            generate_code_trace,
            question=request.question,
            rag=rag,
            repo_url=request.repo_url,
            repo_type=request.type or 'azuredevops',
            repo_name=repo_name,
            language=request.language or 'en',
        )

        return JSONResponse(content=result.model_dump())

    except Exception as e:
        logger.error(f"[CodeTrace] Error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )
