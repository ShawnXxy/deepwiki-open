import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from backend/.env
_backend_dir = Path(__file__).resolve().parent
load_dotenv(_backend_dir / '.env')

from backend.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply watchfiles monkey patch BEFORE uvicorn import
is_development = os.environ.get("NODE_ENV") != "production"
if is_development:
    import watchfiles
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "logs")
    
    original_watch = watchfiles.watch
    def patched_watch(*args, **kwargs):
        # Only watch the api directory but exclude logs subdirectory
        # Instead of watching the entire api directory, watch specific subdirectories
        api_subdirs = []
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and item != "logs":
                api_subdirs.append(item_path)
            elif os.path.isfile(item_path) and item.endswith(".py"):
                api_subdirs.append(item_path)
        
        return original_watch(*api_subdirs, **kwargs)
    watchfiles.watch = patched_watch

import uvicorn

# Import Azure configuration functions
from backend.config import is_azure_openai_configured

# Check for Azure OpenAI configuration (only log in worker process, not reloader parent)
use_azure_openai = is_azure_openai_configured()

# Only log startup messages in the actual worker process, not the reloader parent
# The reloader parent has __name__ == "__main__", worker has __name__ == "backend.main"
if __name__ != "__main__":
    if use_azure_openai:
        logger.info("Azure OpenAI configuration detected. Using Azure OpenAI for text generation and embeddings.")
    else:
        logger.error("Azure OpenAI is not configured. Please set the required environment variables.")
        logger.error("Required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_VERSION")

# Import the app at module level for uvicorn to find it
from backend.app import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    # Use FASTAPI_PORT to avoid conflict with Azure's PORT/WEBSITES_PORT
    port = int(os.environ.get("FASTAPI_PORT", os.environ.get("PORT", 8001)))

    logger.info(f"Starting Streaming API on port {port}")

    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        reload_excludes=["**/logs/*", "**/__pycache__/*", "**/*.pyc"] if is_development else None,
    )
