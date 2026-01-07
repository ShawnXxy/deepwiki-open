import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio

# Configure logging
from backend.tools.logger import setup_logging, log_frontend_message
from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.types import WikiCacheIdentifier

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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Pydantic Models ---
class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp
    language: str # Extracted from filename
    comprehensive: bool = True  # Whether this is a comprehensive wiki
    branch: Optional[str] = None  # Branch name (None for legacy caches without branch)

class RepoInfo(BaseModel):
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    branch: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None


class WikiSection(BaseModel):
    """
    Model for the wiki sections.
    """
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[str]] = None


class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  #compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    comprehensive: bool = True  # Whether this is a comprehensive wiki (default True for backwards compatibility)
    is_partial: bool = False  # Whether this is a partial/checkpoint cache (incomplete generation)

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    repo: RepoInfo
    language: str
    comprehensive: bool = True  # Whether this is a comprehensive wiki
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str
    is_partial: bool = False  # Whether this is a partial/checkpoint save (incomplete generation)

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

class AuthorizationConfig(BaseModel):
    code: str = Field(..., description="Authorization code")

from backend.config import configs, WIKI_AUTH_MODE, WIKI_AUTH_CODE, get_azure_openai_config, get_file_filters_config

@app.get("/lang/config")
async def get_lang_config():
    return configs["lang_config"]

@app.get("/filters/config")
async def get_filters_config():
    """
    Get default file filters configuration from repo.json.
    Returns excluded directories and files lists.
    """
    return get_file_filters_config()

@app.get("/auth/status")
async def get_auth_status():
    """
    Check if authentication is required for the wiki.
    """
    return {"auth_required": WIKI_AUTH_MODE}

@app.post("/auth/validate")
async def validate_auth_code(request: AuthorizationConfig):
    """
    Check authorization code.
    """
    return {"success": WIKI_AUTH_CODE == request.code}

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of Azure OpenAI provider
    with deployment info from infra.json.

    Returns:
        ModelConfig: A configuration object containing Azure provider and model
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

@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
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

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path):
            # Exclude hidden dirs/files and virtual envs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != '.venv']
            for file in files:
                if file.startswith('.') or file == '__init__.py' or file == '.DS_Store':
                    continue
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file)
                # Find README.md (case-insensitive)
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

# Import the simplified chat implementation
from backend.simple_chat import chat_completions_stream
from backend.websocket_wiki import handle_websocket_chat

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoint
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
WIKI_CACHE_BLOB_PREFIX = "wikicache"  # Blob path prefix
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_filename(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True, branch: Optional[str] = None) -> str:
    """Generates the filename for a given wiki cache (new format with branch suffix).
    
    DEPRECATED: Use WikiCacheIdentifier.get_cache_filename() instead.
    
    Args:
        owner: Repository owner
        repo: Repository name
        repo_type: Repository type (github, gitlab, etc.)
        language: Language code
        comprehensive: Whether this is comprehensive or concise wiki
        branch: Branch name (optional, defaults to 'default' if not specified)
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


def get_wiki_cache_filename_legacy(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True) -> str:
    """Generates the legacy filename for wiki cache (without branch suffix).
    
    DEPRECATED: Use WikiCacheIdentifier.get_cache_filename_legacy() instead.
    
    Used for backward compatibility with existing caches created before branch support.
    
    Args:
        owner: Repository owner
        repo: Repository name
        repo_type: Repository type (github, gitlab, etc.)
        language: Language code
        comprehensive: Whether this is comprehensive or concise wiki
    """
    cache_id = WikiCacheIdentifier(
        owner=owner,
        repo=repo,
        repo_type=repo_type,
        language=language,
        comprehensive=comprehensive,
        branch=None  # Legacy doesn't include branch
    )
    return cache_id.get_cache_filename_legacy()


def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True, branch: Optional[str] = None) -> str:
    """Generates the local file path for a given wiki cache."""
    filename = get_wiki_cache_filename(owner, repo, repo_type, language, comprehensive, branch)
    return os.path.join(WIKI_CACHE_DIR, filename)


def get_wiki_cache_path_legacy(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True) -> str:
    """Generates the legacy local file path for wiki cache (without branch suffix)."""
    filename = get_wiki_cache_filename_legacy(owner, repo, repo_type, language, comprehensive)
    return os.path.join(WIKI_CACHE_DIR, filename)


def get_wiki_cache_blob_path(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True, branch: Optional[str] = None) -> str:
    """Generates the blob path for a given wiki cache."""
    filename = get_wiki_cache_filename(owner, repo, repo_type, language, comprehensive, branch)
    return f"{WIKI_CACHE_BLOB_PREFIX}/{filename}"


def get_wiki_cache_blob_path_legacy(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True) -> str:
    """Generates the legacy blob path for wiki cache (without branch suffix)."""
    filename = get_wiki_cache_filename_legacy(owner, repo, repo_type, language, comprehensive)
    return f"{WIKI_CACHE_BLOB_PREFIX}/{filename}"

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str, comprehensive: bool = True, branch: Optional[str] = None) -> Optional[WikiCacheData]:
    """
    Reads wiki cache data from storage with backward compatibility.
    
    First tries the new format with branch suffix, then falls back to legacy format
    (without branch suffix) for backward compatibility with existing caches.
    
    When Azure Blob Storage is configured:
        - Uses blob storage exclusively
        - Raises ConnectionError on failure (no fallback)
    When Azure Blob Storage is NOT configured:
        - Uses local storage
    
    Args:
        owner: Repository owner
        repo: Repository name
        repo_type: Repository type
        language: Language code
        comprehensive: Whether this is comprehensive or concise wiki
        branch: Branch name (optional)
    """
    # Try Azure Blob Storage when configured
    if is_blob_storage_configured():
        blob_path = get_wiki_cache_blob_path(owner, repo, repo_type, language, comprehensive, branch)
        blob_path_legacy = get_wiki_cache_blob_path_legacy(owner, repo, repo_type, language, comprehensive)
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
                logger.error(error_msg)
                raise ConnectionError(error_msg)
            
            # Try new format first (with branch suffix)
            if blob_client.exists(blob_path):
                logger.info(f"Reading wiki cache from Azure Blob Storage: {blob_path}")
                content = blob_client.download_text(blob_path)
                if content:
                    data = json.loads(content)
                    return WikiCacheData(**data)
            
            # Fall back to legacy format (without branch suffix) for backward compatibility
            if blob_client.exists(blob_path_legacy):
                logger.info(f"Reading wiki cache from legacy blob path: {blob_path_legacy}")
                content = blob_client.download_text(blob_path_legacy)
                if content:
                    data = json.loads(content)
                    return WikiCacheData(**data)
            
            logger.info(f"Wiki cache not found in blob storage (tried: {blob_path} and {blob_path_legacy})")
            return None
        except ConnectionError:
            raise
        except Exception as e:
            error_msg = f"Failed to read wiki cache from Azure Blob Storage: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    # Local storage mode (blob not configured)
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language, comprehensive, branch)
    cache_path_legacy = get_wiki_cache_path_legacy(owner, repo, repo_type, language, comprehensive)
    
    # Try new format first (with branch suffix)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Read wiki cache from: {cache_path}")
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
    
    # Fall back to legacy format (without branch suffix) for backward compatibility
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
    
    When Azure Blob Storage is configured:
        - Uses blob storage exclusively
        - Raises ConnectionError on failure (no fallback)
    When Azure Blob Storage is NOT configured:
        - Uses local storage
        
    Supports partial/checkpoint saves when is_partial=True, allowing resumption
    after interruption.
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
        blob_path = get_wiki_cache_blob_path(data.repo.owner, data.repo.repo, data.repo.type, data.language, data.comprehensive, data.repo.branch)
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
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
    
    # Local storage mode (blob not configured)
    cache_path = get_wiki_cache_path(data.repo.owner, data.repo.repo, data.repo.type, data.language, data.comprehensive, data.repo.branch)
    logger.info(f"Attempting to save wiki cache locally. Path: {cache_path}")
    try:
        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    comprehensive: bool = Query(True, description="Whether this is comprehensive or concise wiki"),
    branch: Optional[str] = Query(None, description="Branch name (optional, defaults to 'default')")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        language = configs["lang_config"]["default"]

    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}, comprehensive: {comprehensive}, branch: {branch}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language, comprehensive, branch)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        # Or, raise HTTPException(status_code=404, detail="Wiki cache not found") if preferred
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}, comprehensive: {comprehensive}, branch: {branch}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]

    if not supported_langs.__contains__(request_data.language):
        request_data.language = configs["lang_config"]["default"]

    logger.info(f"Attempting to save wiki cache for {request_data.repo.owner}/{request_data.repo.repo} ({request_data.repo.type}), lang: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content"),
    comprehensive: bool = Query(True, description="Whether this is comprehensive or concise wiki"),
    branch: Optional[str] = Query(None, description="Branch name (optional, defaults to 'default')"),
    authorization_code: Optional[str] = Query(None, description="Authorization code")
):
    """
    Deletes a specific wiki cache.
    Uses Azure Blob Storage when configured, raises error on failure.
    """
    # Language validation
    supported_langs = configs["lang_config"]["supported_languages"]
    if not supported_langs.__contains__(language):
        raise HTTPException(status_code=400, detail="Language is not supported")

    if WIKI_AUTH_MODE:
        logger.info("check the authorization code")
        if not authorization_code or WIKI_AUTH_CODE != authorization_code:
            raise HTTPException(status_code=401, detail="Authorization code is invalid")

    logger.info(f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type}), lang: {language}, comprehensive: {comprehensive}, branch: {branch}")

    try:
        # Try Azure Blob Storage when configured
        if is_blob_storage_configured():
            blob_path = get_wiki_cache_blob_path(owner, repo, repo_type, language, comprehensive, branch)
            try:
                blob_client = get_blob_storage_client()
                if not blob_client:
                    error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
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
        
        # Local storage mode (blob not configured)
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

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }


class FrontendLogRequest(BaseModel):
    """Model for frontend log messages"""
    level: str = Field(..., description="Log level: debug, info, warn, error")
    message: str = Field(..., description="Log message")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context data")


class FrontendLogBatchRequest(BaseModel):
    """Model for batch frontend log messages"""
    logs: List[FrontendLogRequest] = Field(..., description="List of log entries")


@app.post("/log")
async def log_frontend(request: FrontendLogRequest):
    """
    Receive and store frontend log messages.
    Logs are written to frontend-yymmdd.log with daily rotation.
    """
    try:
        log_frontend_message(request.level, request.message, request.context)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Failed to log frontend message: {e}")
        raise HTTPException(status_code=500, detail="Failed to log message")


@app.post("/log/batch")
async def log_frontend_batch(request: FrontendLogBatchRequest):
    """
    Receive and store multiple frontend log messages in a batch.
    More efficient for high-volume logging.
    """
    try:
        for log_entry in request.logs:
            log_frontend_message(log_entry.level, log_entry.message, log_entry.context)
        return {"status": "logged", "count": len(request.logs)}
    except Exception as e:
        logger.error(f"Failed to log frontend batch: {e}")
        raise HTTPException(status_code=500, detail="Failed to log messages")

@app.get("/")
async def root():
    """Root endpoint to check if the API is running and list available endpoints dynamically."""
    # Collect routes dynamically from the FastAPI app
    endpoints = {}
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

    # Optionally, sort endpoints for readability
    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }

# --- Processed Projects Endpoint --- (New Endpoint)
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache.
    Uses Azure Blob Storage when configured, raises error on failure.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json
    """
    project_entries: List[ProcessedProjectEntry] = []

    def parse_cache_filename(filename: str, last_modified_ms: int) -> Optional[ProcessedProjectEntry]:
        """Parse a cache filename into a ProcessedProjectEntry.
        
        Supports two filename formats:
        - New: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}_{branch}.json
        - Legacy: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}.json (no branch)
        where mode = comprehensive/concise
        """
        # Extract just the filename if it includes a path prefix
        base_filename = os.path.basename(filename)
        if not (base_filename.startswith("deepwiki_cache_") and base_filename.endswith(".json")):
            return None
        
        parts = base_filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')
        
        # New format with branch: repo_type_owner_repo_language_mode_branch (6+ parts)
        # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en_comprehensive_main.json
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
        # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en_comprehensive.json
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
                branch="default"  # Legacy format has no branch, use "default"
            )
        return None

    try:
        # Try Azure Blob Storage when configured
        if is_blob_storage_configured():
            try:
                blob_client = get_blob_storage_client()
                if not blob_client:
                    error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
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
        
        # Local storage mode (blob not configured)
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
