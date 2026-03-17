"""
Pydantic models for wiki operations.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

from backend.modules.repository.models import RepoInfo


class WikiPage(BaseModel):
    """Model for a wiki page."""
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str  # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]


class WikiSection(BaseModel):
    """Model for the wiki sections."""
    id: str
    title: str
    pages: List[str]
    subsections: Optional[List[Any]] = None

    @field_validator('subsections', mode='before')
    @classmethod
    def accept_string_or_section(cls, v):
        """Accept both string IDs (frontend) and WikiSection dicts."""
        if v is None:
            return None
        result = []
        for item in v:
            if isinstance(item, str):
                # Legacy format: subsection ID as string — skip
                # (frontend uses flat section refs, not nested objects)
                continue
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append(item)
        return result if result else None


# Resolve forward reference for self-referencing model
WikiSection.model_rebuild()


class WikiStructureModel(BaseModel):
    """Model for the overall wiki structure."""
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: Optional[List[WikiSection]] = None
    rootSections: Optional[List[str]] = None


class WikiCacheData(BaseModel):
    """Model for the data to be stored in the wiki cache."""
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  # compatible for old cache
    repo: Optional[RepoInfo] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    comprehensive: bool = True  # Whether this is a comprehensive wiki
    is_partial: bool = False  # Whether this is a partial/checkpoint cache
    commit_hash: Optional[str] = None  # HEAD commit used during indexing
    indexed_at: Optional[str] = None   # ISO timestamp of wiki generation


class WikiCacheRequest(BaseModel):
    """Model for the request body when saving wiki cache."""
    repo: RepoInfo
    language: str
    comprehensive: bool = True
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    provider: str
    model: str
    is_partial: bool = False
    commit_hash: Optional[str] = None
    indexed_at: Optional[str] = None


class WikiExportRequest(BaseModel):
    """Model for requesting a wiki export."""
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format")


class ProcessedProjectEntry(BaseModel):
    """Model for a processed project entry."""
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str
    submittedAt: int  # Timestamp
    language: str
    comprehensive: bool = True
    branch: Optional[str] = None


# --- Model Configuration Models ---
class Model(BaseModel):
    """Model for LLM model configuration."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")


class Provider(BaseModel):
    """Model for LLM provider configuration."""
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether supports custom models")


class ModelConfig(BaseModel):
    """Model for the entire model configuration."""
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")


class AuthorizationConfig(BaseModel):
    """Model for authorization."""
    code: str = Field(..., description="Authorization code")


class FrontendLogRequest(BaseModel):
    """Model for frontend log messages."""
    level: str = Field(..., description="Log level: debug, info, warn, error")
    message: str = Field(..., description="Log message")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context data")


class FrontendLogBatchRequest(BaseModel):
    """Model for batch frontend log messages."""
    logs: List[FrontendLogRequest] = Field(..., description="List of log entries")
