"""
Pydantic models for repository operations.
"""

from typing import Optional
from pydantic import BaseModel


class RepoInfo(BaseModel):
    """Repository information model."""
    owner: str
    repo: str
    type: str
    token: Optional[str] = None
    branch: Optional[str] = None
    localPath: Optional[str] = None
    repoUrl: Optional[str] = None
