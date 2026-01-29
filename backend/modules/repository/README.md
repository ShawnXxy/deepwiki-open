# Repository Module

The repository module handles Git operations and file content retrieval from Azure DevOps repositories.

## Module Structure

```
modules/repository/
├── __init__.py       # Module exports
├── models.py         # Pydantic models for repository info
├── git_ops.py        # Git clone, pull, branch detection
├── file_content.py   # Remote file content retrieval
└── routes.py         # API routes for local repo operations
```

## Components

### models.py - Data Models

**RepoInfo**: Repository information model:
```python
class RepoInfo(BaseModel):
    owner: str           # Repository owner
    repo: str            # Repository name
    type: str            # github, gitlab, bitbucket, azuredevops
    token: Optional[str] # Access token for private repos
    branch: Optional[str]
    localPath: Optional[str]
    repoUrl: Optional[str]
```

### git_ops.py - Git Operations

**download_repo()**: Clone or update a repository:
```python
def download_repo(
    repo_url: str,
    local_path: str,
    type: str = "github",
    access_token: str = None,
    branch: str = None,
    force_update: bool = False
) -> str:
    # 1. Check if repo already exists
    # 2. If exists and force_update: git pull
    # 3. Otherwise: git clone with auth
```

Authentication URL format for Azure DevOps:
```
https://{token}@dev.azure.com/org/project/_git/repo
```

**detect_default_branch()**: Detect the default branch:
```python
def detect_default_branch(local_path: str) -> str:
    # 1. Try git symbolic-ref refs/remotes/origin/HEAD
    # 2. Fallback to checking main/master
    # 3. Default to 'main'
```

### file_content.py - Remote File Retrieval

**get_azuredevops_file_content(repo_url, file_path, access_token)**:
- Uses Azure DevOps Items API
- Parses organization, project, repo from URL
- Returns file content as string

**get_file_content()**: Unified dispatcher:
```python
def get_file_content(repo_url, file_path, repo_type, access_token=None) -> str:
    if repo_type == "azuredevops":
        return get_azuredevops_file_content(...)
    # ...
```

### routes.py - API Endpoints

**GET /local_repo/structure**: Get file tree and README for local repository:
```python
@router.get("/local_repo/structure")
async def get_local_repo_structure(path: str):
    # Returns: {"file_tree": "...", "readme": "..."}
```

Used by frontend for local repository wiki generation.

## Workflow

### Repository Cloning Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Repo URL   │────►│  download_repo │────►│  Auth URL       │
│  + Token    │     │  ()            │     │  Construction   │
└─────────────┘     └────────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  git clone      │
                                           │  --depth 1      │
                                           │  --single-branch│
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  ~/.adalflow/   │
                                           │  repos/         │
                                           │  {owner}_{repo} │
                                           └─────────────────┘
```

### File Content Retrieval Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  File Path  │────►│  get_file_     │────►│  Azure DevOps   │
│  + Repo URL │     │  content()     │     │  /items?path=   │
└─────────────┘     └────────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  Return Content │
                                           └─────────────────┘
```

## Local Storage

Cloned repositories are stored at:
```
~/.adalflow/repos/
├── owner1_repo1/
├── owner2_repo2/
└── ...
```

## Usage Examples

### Clone a Repository

```python
from backend.modules.repository import download_repo

result = download_repo(
    repo_url="https://dev.azure.com/org/project/_git/repo",
    local_path="~/.adalflow/repos/org_repo",
    type="azuredevops",
    access_token="your-pat-token",
    branch="main"
)
```

### Get Remote File Content

```python
from backend.modules.repository import get_file_content

content = get_file_content(
    repo_url="https://dev.azure.com/org/project/_git/repo",
    file_path="src/main.py",
    repo_type="azuredevops",
    access_token="your-pat-token"
)
```

### Get Local Repo Structure

```bash
curl "http://localhost:8001/local_repo/structure?path=/path/to/repo"
```

Response:
```json
{
    "file_tree": "src/main.py\nsrc/utils.py\nREADME.md",
    "readme": "# My Project\n\nDescription..."
}
```

## Dependencies

- `subprocess`: Git command execution
- `requests`: HTTP requests to Git hosting APIs
- `urllib.parse`: URL parsing and construction
