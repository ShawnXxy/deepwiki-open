"""
Type classes for DeepWiki configuration JSON files.
These classes provide strong typing and validation for configuration data.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


# ============================================================================
# Infra Configuration (infra.json)
# ============================================================================

class ManagedIdentityConfig(BaseModel):
    """Managed identity configuration for Azure authentication."""
    name: str
    client_id: str


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration for text generation."""
    endpoint: str
    api_version: str = "2024-12-01-preview"
    deployment: str
    temperature: float = 1.0


class AzureOpenAIEmbeddingConfig(BaseModel):
    """Azure OpenAI configuration for embeddings."""
    endpoint: str
    api_version: str = "2024-12-01-preview"
    deployment: str = "text-embedding-3-large"
    dimensions: int = 3072


class AzureBlobStorageConfig(BaseModel):
    """Azure Blob Storage configuration."""
    enabled: bool = False
    account_name: str
    container_name: str


class AzureApplicationInsightsConfig(BaseModel):
    """Azure Application Insights configuration."""
    enabled: bool = False
    name: str
    connection_string: str


class InfraConfig(BaseModel):
    """Main infrastructure configuration from infra.json."""
    managed_identity: ManagedIdentityConfig
    azure_openai: AzureOpenAIConfig
    azure_openai_embedding: AzureOpenAIEmbeddingConfig
    azure_blob_storage: AzureBlobStorageConfig
    azure_application_insights: AzureApplicationInsightsConfig


# ============================================================================
# Embedder Configuration (embedder.json)
# ============================================================================

class ModelKwargs(BaseModel):
    """Model keyword arguments."""
    encoding_format: Optional[str] = None
    model: Optional[str] = None
    dimensions: Optional[int] = None
    temperature: Optional[float] = None

    class Config:
        extra = "allow"  # Allow additional fields


class EmbedderModelConfig(BaseModel):
    """Embedder model configuration."""
    client_class: str = "AzureAIClient"
    batch_size: int = 10
    model_kwargs: ModelKwargs = Field(default_factory=ModelKwargs)
    initialize_kwargs: Optional[Dict[str, Any]] = None


class RetrieverConfig(BaseModel):
    """Retriever configuration for RAG."""
    top_k: int = 20
    top_k_wiki: int = 40


class TextSplitterConfig(BaseModel):
    """Text splitter configuration for chunking. All values from embedder.json."""
    split_by: str
    chunk_size: int
    chunk_overlap: int
    separators: Optional[Dict[str, str]] = None


class EmbedderConfig(BaseModel):
    """Embedder pipeline configuration from embedder.json."""
    embedder: EmbedderModelConfig
    retriever: RetrieverConfig
    text_splitter: TextSplitterConfig


# ============================================================================
# Generator Configuration (generator.json)
# ============================================================================

class GeneratorConfig(BaseModel):
    """Generator configuration for text generation from generator.json."""
    client_class: str = "AzureAIClient"
    model_kwargs: ModelKwargs = Field(default_factory=ModelKwargs)
    initialize_kwargs: Optional[Dict[str, Any]] = None


# ============================================================================
# Repository Configuration (repo.json)
# ============================================================================

class FileFiltersConfig(BaseModel):
    """File filtering configuration from repo.json."""
    excluded_dirs: List[str] = Field(default_factory=list)
    excluded_files: List[str] = Field(default_factory=list)


class RepositoryConfig(BaseModel):
    """Repository size and constraints configuration from repo.json."""
    max_size_mb: int = 50000


# ============================================================================
# Language Configuration (lang.json)
# ============================================================================

class LanguageConfig(BaseModel):
    """Language configuration from lang.json."""
    supported_languages: Dict[str, str]
    default: str = "en"

    @validator('default')
    def validate_default_language(cls, v, values):
        """Ensure default language is in supported languages."""
        if 'supported_languages' in values and v not in values['supported_languages']:
            raise ValueError(f"Default language '{v}' not in supported languages")
        return v


# ============================================================================
# Helper Functions
# ============================================================================
# Conversion utilities have been moved to converter.py
# Import from_dict, to_dict, to_dict_flatten from backend.types.converter
