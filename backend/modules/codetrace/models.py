"""
Pydantic models for codetrace operations.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CodeReference(BaseModel):
    """A reference to a specific code location."""
    ref_id: str = ""
    file_path: str
    start_line: int
    end_line: int
    snippet: str = ""
    annotation: str = ""


class CodeTraceSection(BaseModel):
    """A numbered section in the code trace."""
    id: str
    title: str
    motivation: str = ""
    details: str = ""
    code_refs: List[CodeReference] = []
    connections: List[str] = []


class SourceChunk(BaseModel):
    """A chunk of source code from a file."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str = ""


class CodeTraceResult(BaseModel):
    """Complete code trace response."""
    query: str
    title: str
    sections: List[CodeTraceSection] = []
    source_files: List[str] = []
    source_contents: dict = {}  # file_path -> List[SourceChunk dict]
    generated_at: Optional[str] = None


class CodeTraceRequest(BaseModel):
    """Request model for code trace generation."""
    repo_url: str = Field(..., description="Repository URL")
    question: str = Field(..., description="User question")
    type: Optional[str] = Field(
        "azuredevops", description="Repository type"
    )
    branch: Optional[str] = Field(
        None, description="Branch name"
    )
    token: Optional[str] = Field(
        None, description="Access token for private repos"
    )
    language: Optional[str] = Field(
        "en", description="Response language"
    )
