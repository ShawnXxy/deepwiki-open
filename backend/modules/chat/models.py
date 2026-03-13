"""
Pydantic models for chat requests and responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Model for a chat message."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatCompletionRequest(BaseModel):
    """Model for requesting a chat completion."""
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(
        None, description="Optional path to a file in the repository"
    )
    token: Optional[str] = Field(
        None, description="Personal access token for private repositories"
    )
    type: Optional[str] = Field(
        "github", description="Type of repository (e.g., 'github', 'gitlab')"
    )
    branch: Optional[str] = Field(
        None, description="Specific branch to clone/process"
    )

    # Model parameters (provider is ignored, always uses Azure)
    provider: str = Field(
        "azure", description="Model provider (always Azure OpenAI)"
    )
    model: Optional[str] = Field(
        None, description="Model name for Azure OpenAI deployment"
    )

    language: Optional[str] = Field(
        "en", description="Language for content generation"
    )
    excluded_dirs: Optional[str] = Field(
        None, description="Comma-separated list of directories to exclude"
    )
    excluded_files: Optional[str] = Field(
        None, description="Comma-separated list of file patterns to exclude"
    )
    included_dirs: Optional[str] = Field(
        None, description="Comma-separated list of directories to include exclusively"
    )
    included_files: Optional[str] = Field(
        None, description="Comma-separated list of file patterns to include"
    )
    force_reprocess: Optional[bool] = Field(
        False, description="If True, ignore existing pkl/vectors and create fresh"
    )

    # Wiki structure generation fields
    # When wiki_structure_request=True, backend uses promptstore templates
    wiki_structure_request: Optional[bool] = Field(
        False, description="If True, this is a wiki structure generation request"
    )
    file_tree: Optional[str] = Field(
        None, description="Repository file tree for wiki structure generation"
    )
    readme: Optional[str] = Field(
        None, description="README content for wiki structure generation"
    )
    comprehensive: Optional[bool] = Field(
        True, description="If True, use comprehensive wiki prompt with sections"
    )

    # Wiki page generation fields
    # When wiki_page_request=True, backend uses file-path-aware retrieval
    # and constructs the page prompt from promptstore templates
    wiki_page_request: Optional[bool] = Field(
        False, description="If True, generate wiki page with "
        "file-path-aware retrieval"
    )
    page_id: Optional[str] = Field(
        None, description="Numbered ID of the wiki page (e.g., 2.1)"
    )
    page_title: Optional[str] = Field(
        None, description="Title of the wiki page to generate"
    )
    page_file_paths: Optional[List[str]] = Field(
        None, description="Relevant source file paths for the page"
    )
    page_related_pages: Optional[List[str]] = Field(
        None, description="IDs of related wiki pages"
    )
