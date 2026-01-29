# Wiki Module

The wiki module manages wiki cache storage, retrieval, and export functionality for generated repository documentation.

## Module Structure

```
modules/wiki/
├── __init__.py       # Module exports
├── models.py         # Pydantic models for wiki structures
├── cache.py          # Wiki cache read/write/delete operations
├── export.py         # Export to Markdown/JSON formats
└── routes.py         # FastAPI routes for wiki operations
```

## Components

### models.py - Data Models

**WikiPage**: Individual wiki page with:
- `id`: Unique page identifier
- `title`: Page title
- `content`: Markdown content
- `filePaths`: Related source files
- `importance`: Priority (high/medium/low)
- `relatedPages`: Links to other pages

**WikiSection**: Grouping of pages:
- `id`: Section identifier
- `title`: Section title
- `pages`: List of page IDs in this section
- `subsections`: Optional nested sections

**WikiStructureModel**: Complete wiki structure:
- `id`, `title`, `description`
- `pages`: All WikiPage objects
- `sections`: All WikiSection objects
- `rootSections`: Top-level section IDs

**WikiCacheData**: Stored cache format:
- `wiki_structure`: The WikiStructureModel
- `generated_pages`: Dict of page ID → WikiPage
- `repo`: Repository info
- `comprehensive`: Full vs concise mode
- `is_partial`: For checkpoint saves

**ProcessedProjectEntry**: List entry for processed repos:
- `owner`, `repo`, `name`
- `repo_type`, `language`
- `comprehensive`, `branch`
- `submittedAt`: Timestamp

### cache.py - Cache Operations

Handles reading and writing wiki cache to local storage or Azure Blob:

```python
WIKI_CACHE_DIR = ~/.adalflow/wikicache/
WIKI_CACHE_BLOB_PREFIX = "wikicache"

def get_wiki_cache_filename(owner, repo, repo_type, language, comprehensive, branch):
    # Returns: deepwiki_cache_{repo_type}_{owner}_{repo}_{lang}_{mode}_{branch}.json
    
async def read_wiki_cache(...) -> Optional[WikiCacheData]:
    # 1. Check blob storage if configured
    # 2. Fall back to local storage
    # 3. Try new format first, then legacy (without branch)
    
async def save_wiki_cache(request_data) -> bool:
    # Saves to blob or local based on configuration
```

**Backward Compatibility**: Automatically handles legacy cache files (without branch suffix) for smooth migration.

### export.py - Export Utilities

Generates downloadable wiki exports:

```python
def generate_markdown_export(repo_url, pages) -> str:
    # Creates single Markdown document with:
    # - Metadata header
    # - Table of contents
    # - All pages with content
    
def generate_json_export(repo_url, pages) -> str:
    # Creates JSON with:
    # - Metadata (repository, timestamp, page_count)
    # - Full pages array
```

### routes.py - API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/wiki_cache` | GET | Retrieve cached wiki |
| `/api/wiki_cache` | POST | Store generated wiki |
| `/api/wiki_cache` | DELETE | Delete cached wiki |
| `/api/export/wiki` | POST | Export wiki to Markdown/JSON |
| `/api/processed_projects` | GET | List all processed repos |

Key features:
- Language validation against `lang_config`
- Authorization code for delete operations
- Pagination for processed projects list

## Workflow

### Wiki Cache Storage Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│   Frontend  │────►│  POST /api/    │────►│  save_wiki_     │
│  Generated  │     │  wiki_cache    │     │  cache()        │
│  Wiki Data  │     └────────────────┘     └────────┬────────┘
└─────────────┘                                     │
                                                    ▼
                                   ┌────────────────────────────┐
                                   │  Storage Decision:         │
                                   │  - Blob if configured      │
                                   │  - Local ~/.adalflow/      │
                                   │    wikicache/ otherwise    │
                                   └────────────────────────────┘
```

### Wiki Retrieval Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│   Frontend  │────►│  GET /api/     │────►│  read_wiki_     │
│  Request    │     │  wiki_cache    │     │  cache()        │
└─────────────┘     └────────────────┘     └────────┬────────┘
                                                    │
                           ┌────────────────────────┴────────────────────────┐
                           │                                                 │
                           ▼                                                 ▼
                  ┌─────────────────┐                              ┌─────────────────┐
                  │  Try new format │                              │  Try legacy     │
                  │  (with branch)  │──────────── NOT FOUND ──────►│  (no branch)    │
                  └────────┬────────┘                              └────────┬────────┘
                           │                                                 │
                           │ FOUND                                           │ FOUND
                           ▼                                                 ▼
                  ┌─────────────────────────────────────────────────────────────────┐
                  │                     Return WikiCacheData                        │
                  └─────────────────────────────────────────────────────────────────┘
```

## Cache File Naming

Format: `deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}_{branch}.json`

Example:
- `deepwiki_cache_azuredevops_org_project_en_comprehensive_main.json`

## Storage Locations

| Environment | Location |
|-------------|----------|
| Local | `~/.adalflow/wikicache/` |
| Docker | `/root/.adalflow/wikicache/` (volume mount) |
| Azure | Blob container: `wikicache/` prefix |

## Usage Example

```python
from backend.modules.wiki import read_wiki_cache, save_wiki_cache
from backend.modules.wiki.models import WikiCacheRequest

# Read cached wiki
cache = await read_wiki_cache(
    owner="SupportTechOps",
    repo="OrcasCodeWiki",
    repo_type="azuredevops",
    language="en",
    comprehensive=True,
    branch="main"
)

if cache:
    print(f"Found wiki with {len(cache.generated_pages)} pages")
```

## Dependencies

- `backend.clients.blob_client`: Azure Blob operations
- `backend.types`: WikiCacheIdentifier type
- `backend.utils.paths`: Consistent path management
