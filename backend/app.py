"""
FastAPI Backend for DeepWiki Ask/Chat.

Provides only the endpoints needed for the Ask/Chat feature:
- /ws/chat — WebSocket streaming chat (primary)
- /chat/completions/stream — HTTP fallback for cloud environments
- /models/config — Available model info from infra.json
- /filters/config — Default file exclusion patterns
- /health — Deployment health check

Wiki viewing does NOT require this backend. The Next.js frontend
reads cached wiki JSON directly from ~/.adalflow/wikicache/.
"""

import logging
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.logger import setup_logging
from backend.config import (
    get_azure_openai_config,
    get_file_filters_config,
)
from backend.modules.chat.http_handler import chat_completions_stream
from backend.modules.chat.ws_handler import handle_websocket_chat
from backend.modules.wiki.models import (
    Model, Provider, ModelConfig,
)
from backend.modules.codemap.routes import router as codemap_router
from backend.modules.codetrace.routes import router as codetrace_router

setup_logging()
logger = logging.getLogger(__name__)

# Suppress noisy third-party warnings
logging.getLogger("adalflow.tracing").setLevel(logging.ERROR)

app = FastAPI(
    title="DeepWiki Chat API",
    description="Backend for Ask/Chat feature (WebSocket + HTTP streaming)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chat Endpoints ---
app.add_api_route(
    "/chat/completions/stream",
    chat_completions_stream,
    methods=["POST"],
)
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# --- CodeMap Endpoints ---
app.include_router(codemap_router)

# --- CodeTrace Endpoints ---
app.include_router(codetrace_router)


# --- Configuration Endpoints (used by Ask UI) ---

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config_endpoint():
    """Return available model providers from infra.json."""
    try:
        azure_config = get_azure_openai_config(task='chat')
        deployment = azure_config.get("deployment", "o4-mini")
        return ModelConfig(
            providers=[
                Provider(
                    id="azure",
                    name="Azure OpenAI",
                    supportsCustomModel=False,
                    models=[Model(id=deployment, name=deployment)],
                )
            ],
            defaultProvider="azure",
        )
    except Exception as e:
        logger.error(f"Error creating model config: {e}")
        return ModelConfig(
            providers=[
                Provider(
                    id="azure",
                    name="Azure OpenAI",
                    supportsCustomModel=False,
                    models=[Model(id="o4-mini", name="o4-mini")],
                )
            ],
            defaultProvider="azure",
        )


@app.get("/filters/config")
async def get_filters_config():
    """Return default file filters from excluded.json."""
    return get_file_filters_config()


@app.get("/api/wiki_cache")
async def get_wiki_cache(
    owner: str,
    repo: str,
    repo_type: str = "azuredevops",
    language: str = "en",
    comprehensive: bool = True,
    branch: str = None,
):
    """Read wiki cache from storage (blob or local disk)."""
    from backend.modules.wiki.cache import read_wiki_cache
    from fastapi.responses import JSONResponse

    data = await read_wiki_cache(
        owner=owner, repo=repo, repo_type=repo_type,
        language=language, comprehensive=comprehensive, branch=branch,
    )
    if data:
        return JSONResponse(content=data.model_dump())
    return JSONResponse(content={"error": "Wiki cache not found"}, status_code=404)


@app.get("/api/processed_projects")
async def list_processed_projects():
    """List all processed wiki projects from storage."""
    from backend.modules.wiki.cache import list_wiki_caches
    projects = await list_wiki_caches()
    return projects


@app.get("/health")
async def health_check():
    """Health check for deployment monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-chat",
    }


@app.get("/health/openai")
async def health_openai():
    """Test Azure OpenAI connectivity by listing deployments.

    Uses the raw OpenAI sync client directly — avoids adalflow
    parameter conversion that may not match newer model APIs.
    """
    import asyncio
    from backend.config import get_azure_ai_client, get_azure_deployment_name

    try:
        azure_config = get_azure_openai_config(task='chat')
        deployment = azure_config.get("deployment", "o4-mini")
        model = get_azure_ai_client(task='chat')
        deployment_name = get_azure_deployment_name(task='chat')

        def _sync_ping():
            # Get the underlying OpenAI sync client
            sync_client = model.sync_client
            # Simple completions call with minimal overhead
            resp = sync_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=5,
            )
            return resp

        await asyncio.wait_for(
            asyncio.to_thread(_sync_ping),
            timeout=30,
        )

        return {
            "status": "connected",
            "model": deployment_name,
            "timestamp": datetime.now().isoformat(),
        }
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "message": "Azure OpenAI request timed out (30s)",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        msg = str(e)
        if len(msg) > 300:
            msg = msg[:300] + "..."
        return {
            "status": "error",
            "message": msg,
            "timestamp": datetime.now().isoformat(),
        }
