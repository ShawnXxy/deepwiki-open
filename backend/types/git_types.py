"""
Git source types for DeepWiki.
These classes provide strong typing and validation for git repository sources.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from urllib.parse import urlparse


# Supported repository types
RepoType = Literal["github", "gitlab", "bitbucket", "azuredevops"]


class GitCredentials(BaseModel):
    """Git authentication credentials."""
    access_token: Optional[str] = Field(None, description="Personal access token for private repositories")
    
    @validator('access_token')
    def validate_token(cls, v):
        """Ensure token is either None or non-empty string."""
        if v is not None and not v.strip():
            raise ValueError("Access token cannot be empty string")
        return v


class GitReference(BaseModel):
    """Git branch or reference information."""
    branch: Optional[str] = Field(None, description="Specific branch name (e.g., 'main', 'develop')")
    
    @validator('branch')
    def validate_branch(cls, v):
        """Ensure branch name is valid if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Branch name cannot be empty string")
            # Branch names cannot contain certain characters
            invalid_chars = ['..', '~', '^', ':', '?', '*', '[', '\\', ' ']
            for char in invalid_chars:
                if char in v:
                    raise ValueError(f"Branch name cannot contain '{char}'")
        return v


class GitRepository(BaseModel):
    """Git repository information."""
    url: str = Field(..., description="Repository URL (HTTPS format)")
    repo_type: RepoType = Field("github", description="Type of repository hosting service")
    owner: Optional[str] = Field(None, description="Repository owner/organization")
    repo: Optional[str] = Field(None, description="Repository name")
    local_path: Optional[str] = Field(None, description="Local path where repository is cloned")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate repository URL format."""
        if not v:
            raise ValueError("Repository URL is required")
        
        # Basic URL validation
        try:
            parsed = urlparse(v)
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("Repository URL must use http or https protocol")
            if not parsed.netloc:
                raise ValueError("Repository URL must have a valid domain")
        except Exception as e:
            raise ValueError(f"Invalid repository URL: {e}")
        
        return v
    
    @validator('owner', 'repo')
    def validate_name_parts(cls, v):
        """Validate owner and repo names."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Repository names cannot contain certain characters
            invalid_chars = ['..', ' ', '\\', '?', '*', '[', ']', ':', '<', '>', '|']
            for char in invalid_chars:
                if char in v:
                    raise ValueError(f"Name cannot contain '{char}'")
        return v
    
    def get_full_name(self) -> Optional[str]:
        """Get full repository name as 'owner/repo'."""
        if self.owner and self.repo:
            return f"{self.owner}/{self.repo}"
        return None


class GitSource(BaseModel):
    """Complete git source specification with repository, credentials, and reference."""
    repository: GitRepository
    credentials: Optional[GitCredentials] = Field(default_factory=GitCredentials)
    reference: Optional[GitReference] = Field(default_factory=GitReference)
    file_path: Optional[str] = Field(None, description="Optional specific file path in repository")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Ensure file path doesn't start with /."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Remove leading slash for consistency
            if v.startswith('/'):
                v = v[1:]
        return v
    
    def to_dict_for_download(self) -> dict:
        """
        Convert to dictionary format compatible with download_repo function.
        
        Returns:
            dict: Dictionary with keys: repo_url, type, access_token, branch
        """
        return {
            "repo_url": self.repository.url,
            "type": self.repository.repo_type,
            "access_token": self.credentials.access_token if self.credentials else None,
            "branch": self.reference.branch if self.reference else None
        }


class WikiCacheIdentifier(BaseModel):
    """Identifier for wiki cache files."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    repo_type: RepoType = Field(..., description="Repository type")
    language: str = Field("en", description="Content language code")
    comprehensive: bool = Field(True, description="Whether this is comprehensive or concise wiki")
    branch: Optional[str] = Field(None, description="Branch name (None for legacy or 'default' for default branch)")
    
    @validator('language')
    def validate_language(cls, v):
        """Ensure language code is lowercase."""
        return v.lower() if v else "en"
    
    def get_cache_filename(self) -> str:
        """
        Generate cache filename following the naming convention.
        
        Format: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}_{branch}.json
        """
        mode = "comprehensive" if self.comprehensive else "concise"
        branch_suffix = self.branch if self.branch else "default"
        return f"deepwiki_cache_{self.repo_type}_{self.owner}_{self.repo}_{self.language}_{mode}_{branch_suffix}.json"
    
    def get_cache_filename_legacy(self) -> str:
        """
        Generate legacy cache filename (without branch).
        
        Format: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}.json
        """
        mode = "comprehensive" if self.comprehensive else "concise"
        return f"deepwiki_cache_{self.repo_type}_{self.owner}_{self.repo}_{self.language}_{mode}.json"


# ============================================================================
# Helper Functions
# ============================================================================

def parse_github_url(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract owner and repo from a GitHub URL.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Tuple of (owner, repo) or (None, None) if parsing fails
    """
    try:
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1].replace('.git', '')
            return owner, repo
    except Exception:
        pass
    
    return None, None


def create_git_source_from_params(
    repo_url: str,
    repo_type: str = "github",
    token: Optional[str] = None,
    branch: Optional[str] = None,
    file_path: Optional[str] = None,
    local_path: Optional[str] = None
) -> GitSource:
    """
    Create a GitSource object from individual parameters.
    
    Args:
        repo_url: Repository URL
        repo_type: Type of repository (github, gitlab, etc.)
        token: Optional access token
        branch: Optional branch name
        file_path: Optional file path
        local_path: Optional local clone path
        
    Returns:
        GitSource object
    """
    # Try to extract owner/repo from URL
    owner, repo = parse_github_url(repo_url)
    
    repository = GitRepository(
        url=repo_url,
        repo_type=repo_type,
        owner=owner,
        repo=repo,
        local_path=local_path
    )
    
    credentials = GitCredentials(access_token=token) if token else None
    reference = GitReference(branch=branch) if branch else None
    
    return GitSource(
        repository=repository,
        credentials=credentials,
        reference=reference,
        file_path=file_path
    )
