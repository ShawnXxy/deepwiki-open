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

setup_logging()
logger = logging.getLogger(__name__)

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


# --- Configuration Endpoints (used by Ask UI) ---

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config_endpoint():
    """Return available model providers from infra.json."""
    try:
        azure_config = get_azure_openai_config()
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
    """Return default file filters from repo.json."""
    return get_file_filters_config()


@app.get("/health")
async def health_check():
    """Health check for deployment monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-chat",
    }
