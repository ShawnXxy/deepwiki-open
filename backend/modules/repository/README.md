# Repository Module

Git operations for code repositories.

## Responsibility

Handles all interactions with source code repositories:
- Clone repositories (Azure DevOps, GitHub, GitLab, Bitbucket)
- Pull latest changes
- Get HEAD commit hash
- Read file content from cloned repos

## Files

| File | Purpose |
|------|---------|
| `git_ops.py` | `download_repo()`, `get_head_commit_hash()` — clone/pull via git |
| `file_content.py` | `get_file_content()` — read specific file from cloned repo |
| `models.py` | `RepoInfo`, `RepoType` — Pydantic models for repo metadata |

## Dependencies

- **Invokes:** Nothing (foundation layer — zero module dependencies)
- **Invoked by:** `embedder/indexer` (clone for embedding), `chat/` (file content for Q&A)
