# Repository Module

Git operations and file content retrieval for code repositories.

## Responsibility

Foundation layer that handles all interactions with source code repositories:
- Clone repositories from Azure DevOps, GitHub, GitLab, and Bitbucket
- Pull latest changes (with force update support)
- Detect default branch
- Get HEAD commit hash (for citation URLs)
- Read specific file contents from remote hosting APIs

This module has **zero dependencies** on other backend modules — it is the base layer that all other modules build upon.

## Files

| File | Purpose |
|------|---------|
| `git_ops.py` | `download_repo()`, `get_head_commit_hash()`, `detect_default_branch()` — clone/pull via git CLI |
| `file_content.py` | `get_file_content()` — read files from remote hosting APIs (GitHub, GitLab, etc.) |
| `models.py` | `RepoInfo` — Pydantic model for repository metadata |

## How It Works

### Git Operations (`git_ops.py`)

**`download_repo(repo_url, local_path, type, access_token, branch, force_update)`**

Clones a repository to local disk with full error handling:

```
download_repo()
  ├─ If local_path exists and force_update=False → skip (already cloned)
  ├─ If local_path exists and force_update=True  → git pull (update)
  └─ Otherwise → git clone (fresh)
```

**Authentication:** Embeds PAT into the clone URL:
- Azure DevOps: `https://{pat}@dev.azure.com/org/proj/_git/repo`
- GitHub: `https://{pat}@github.com/owner/repo.git`
- Special characters in PATs are URL-encoded (`quote(pat, safe='')`)

**Branch resolution** with fallback chain:
1. Try specified branch (`--branch {branch}`)
2. If fails → try `main`
3. If fails → try `master`
4. If fails → clone without `--branch` (use default)

**Retry logic:** 3 attempts with 2-second delay for transient network errors.

**Shallow cloning:** Uses `--depth 1` for fast initial clone. When re-pulling, detects shallow repos and handles them correctly.

**`get_head_commit_hash(local_path)`**

Returns the full SHA commit hash from `git rev-parse HEAD`. Used by the processor to create commit-pinned source URLs in wiki citations.

**`detect_default_branch(local_path)`**

Queries `git symbolic-ref refs/remotes/origin/HEAD` to determine the default branch name (e.g., `main` vs `master`).

### File Content Retrieval (`file_content.py`)

`get_file_content(repo_url, file_path, type, access_token)` dispatches to platform-specific API calls:

| Platform | Function | API |
|----------|----------|-----|
| GitHub | `get_github_file_content()` | `GET /repos/{owner}/{repo}/contents/{path}` — supports GitHub Enterprise via `/api/v3` prefix |
| GitLab | `get_gitlab_file_content()` | `GET /api/v4/projects/{id}/repository/files/{path}/raw` — supports cloud + self-hosted |
| Bitbucket | `get_bitbucket_file_content()` | `GET /2.0/repositories/{owner}/{repo}/src/{branch}/{path}` |
| Azure DevOps | `get_azuredevops_file_content()` | `GET /{org}/{proj}/_apis/git/repositories/{repo}/items?path={path}` — requires PAT as Basic auth |

All functions return raw file content as a string (UTF-8), or raise on failure.

### Data Model (`models.py`)

```python
class RepoInfo(BaseModel):
    owner: str          # Organization or user
    repo: str           # Repository name
    type: str           # "github", "gitlab", "bitbucket", "azuredevops"
    token: str = ""     # Access token (optional)
    branch: str = ""    # Branch name
    localPath: str = "" # Local clone path
    repoUrl: str = ""   # Full repository URL
```

Used throughout the system to pass repository context between modules.

## Usage

```python
from backend.modules.repository import download_repo, get_file_content
from backend.modules.repository.git_ops import get_head_commit_hash

# Clone a repo
download_repo(
    repo_url="https://dev.azure.com/org/proj/_git/repo",
    local_path="/tmp/repos/org_repo",
    type="azuredevops",
    access_token="pat-token",
    branch="main",
    force_update=True,
)

# Get commit hash for citations
commit_hash = get_head_commit_hash("/tmp/repos/org_repo")

# Read a specific file from remote
content = get_file_content(
    repo_url="https://github.com/owner/repo",
    file_path="src/main.py",
    type="github",
    access_token="ghp_token",
)
```

> The package's `__init__.py` re-exports the most common helpers
> (`download_repo`, `detect_default_branch`, `download_github_repo`,
> `get_file_content`, and the per-platform `get_*_file_content` functions,
> plus an alias `get_ado_file_content`). Lower-level helpers like
> `get_head_commit_hash` live in `git_ops.py` and must be imported
> from the submodule.

## Dependencies

- **Invokes:** Nothing (foundation layer — zero module dependencies)
- **Invoked by:** `embedder/indexer` (clone for embedding), `chat/ws_handler` (file content for Q&A), `processor/code_processor` (clone for wiki generation)
