"""
Repository Module

This module provides Git operations and file content retrieval including:
- Repository cloning and updating
- Branch detection
- File content retrieval from GitHub, GitLab, Bitbucket, Azure DevOps

Exports:
    - RepoInfo: Repository info model
    - download_repo: Clone/update repository
    - detect_default_branch: Detect default branch
    - get_file_content: Get file content from remote
"""

from backend.modules.repository.models import RepoInfo
from backend.modules.repository.git_ops import (
    download_repo,
    detect_default_branch,
    download_github_repo,
)
from backend.modules.repository.file_content import (
    get_file_content,
    get_github_file_content,
    get_gitlab_file_content,
    get_bitbucket_file_content,
    get_azuredevops_file_content,
)

# Alias for backward compatibility
get_ado_file_content = get_azuredevops_file_content

__all__ = [
    "RepoInfo",
    "download_repo",
    "detect_default_branch",
    "download_github_repo",
    "get_file_content",
    "get_github_file_content",
    "get_gitlab_file_content",
    "get_bitbucket_file_content",
    "get_azuredevops_file_content",
    "get_ado_file_content",
]
