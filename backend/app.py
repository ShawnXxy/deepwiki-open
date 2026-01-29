"""
FastAPI Application for DeepWiki Backend.

This module initializes the FastAPI application and registers all routes.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
from backend.tools.logger import setup_logging, log_frontend_message
from backend.config import (
    configs,
    WIKI_AUTH_MODE,
    WIKI_AUTH_CODE,
    get_azure_openai_config,
    get_file_filters_config,
)

# Import module routes
from backend.modules.wiki.routes import router as wiki_router
from backend.modules.repository.routes import router as repo_router
from backend.modules.chat.http_handler import chat_completions_stream
from backend.modules.chat.ws_handler import handle_websocket_chat

# Import wiki models for API
from backend.modules.wiki.models import (
    Model,
    Provider,
    ModelConfig,
    AuthorizationConfig,
    FrontendLogRequest,
    FrontendLogBatchRequest,
)

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Module Routers ---
app.include_router(wiki_router, tags=["wiki"])
app.include_router(repo_router, tags=["repository"])

# --- Add Chat Endpoints ---
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])
app.add_websocket_route("/ws/chat", handle_websocket_chat)


# --- Configuration Endpoints ---

@app.get("/lang/config")
async def get_lang_config():
    """Get language configuration."""
    return configs["lang_config"]


@app.get("/filters/config")
async def get_filters_config():
    """Get default file filters configuration from repo.json."""
    return get_file_filters_config()


@app.get("/auth/status")
async def get_auth_status():
    """Check if authentication is required for the wiki."""
    return {"auth_required": WIKI_AUTH_MODE}


@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """Check authorization code."""
    return {"success": WIKI_AUTH_CODE == request.code}


@app.get("/models/config", response_model=ModelConfig)
async def get_model_config_endpoint():
    """
    Get available model providers and their models.

    Returns the configuration of Azure OpenAI provider with deployment info from infra.json.
    """
    try:
        logger.debug("Fetching model configurations")

        # Get deployment name from infra.json
        azure_config = get_azure_openai_config()
        deployment = azure_config.get("deployment", "o4-mini")

        # Return Azure-only configuration
        return ModelConfig(
            providers=[
                Provider(
                    id="azure",
                    name="Azure OpenAI",
                    supportsCustomModel=False,
                    models=[Model(id=deployment, name=deployment)]
                )
            ],
            defaultProvider="azure"
        )

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return Azure default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="azure",
                    name="Azure OpenAI",
                    supportsCustomModel=False,
                    models=[Model(id="o4-mini", name="o4-mini")]
                )
            ],
            defaultProvider="azure"
        )


# --- Health and Logging Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }


@app.post("/log")
async def log_frontend(request: FrontendLogRequest):
    """Receive and store frontend log messages."""
    try:
        log_frontend_message(request.level, request.message, request.context)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Failed to log frontend message: {e}")
        raise HTTPException(status_code=500, detail="Failed to log message")


@app.post("/log/batch")
async def log_frontend_batch(request: FrontendLogBatchRequest):
    """Receive and store multiple frontend log messages in a batch."""
    try:
        for log_entry in request.logs:
            log_frontend_message(log_entry.level, log_entry.message, log_entry.context)
        return {"status": "logged", "count": len(request.logs)}
    except Exception as e:
        logger.error(f"Failed to log frontend batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to log messages")


@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints."""
    # Collect routes dynamically from the FastAPI app
    endpoints: Dict[str, Any] = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    # Sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }
