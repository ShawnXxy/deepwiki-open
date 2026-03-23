# Wiki Module

Wiki cache management, data models, and export for DeepWiki.

## Responsibility

Manages the lifecycle of generated wiki data:
- **Read/write** wiki cache JSON files (local disk or Azure Blob)
- **Data models** for pages, sections, structure, and cache metadata
- **Export** wiki content as Markdown or JSON downloads
- **XML repair** for truncated LLM responses (content filter recovery)
- **Cache naming** conventions with branch support + legacy fallback

## Files

| File | Purpose |
|------|---------|
| `models.py` | `WikiPage`, `WikiSection`, `WikiStructureModel`, `WikiCacheData` — all data models |
| `cache.py` | `read_wiki_cache()`, `save_wiki_cache()` — read/write wiki JSON cache |
| `export.py` | `generate_markdown_export()`, `generate_json_export()` — downloadable exports |
| `xml_repair.py` | `repair_wiki_structure_xml()`, `close_open_tags()` — truncation recovery |

## How It Works

### Data Models (`models.py`)

The wiki data hierarchy:

```
WikiCacheData                              # Top-level cache file
├── wiki_structure: WikiStructureModel     # Table of contents
│   ├── title, description                 # Wiki metadata
│   ├── pages: List[WikiPage]              # All page stubs (id, title, filePaths)
│   ├── sections: List[WikiSection]        # Hierarchical sections
│   │   └── subsections: List[WikiSection] # Recursive nesting
│   └── rootSections: List[str]            # Top-level section IDs
├── generated_pages: Dict[str, WikiPage]   # Actual page content (keyed by id)
│   └── WikiPage
│       ├── id: str                        # "1", "1.1", "2.3"
│       ├── title: str                     # Page heading
│       ├── content: str                   # Markdown with Mermaid diagrams
│       ├── filePaths: List[str]           # Relevant source files
│       ├── importance: str                # "high", "medium", "low"
│       └── relatedPages: List[str]        # Cross-reference IDs
├── repo: RepoInfo                         # Repository metadata
├── commit_hash: str                       # HEAD commit at generation time
├── indexed_at: str                        # ISO timestamp
├── provider: str                          # "azure"
├── model: str                             # Deployment name (e.g., "gpt-4o")
├── comprehensive: bool                    # True=15-25 pages, False=4-6
└── is_partial: bool                       # True if checkpoint save
```

Additional models:
- `WikiCacheRequest` — Request body for saving (sent by processor)
- `WikiExportRequest` — Export request (markdown/json format)
- `ProcessedProjectEntry` — Entry for the project listing page
- `Model`, `Provider`, `ModelConfig` — LLM provider configuration models
- `FrontendLogRequest/Batch` — Frontend log shipping models

### Cache Operations (`cache.py`)

**File naming convention:**

```
deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}_{branch}.json
```

Where `mode` is `comprehensive` or `concise`, and `branch` defaults to `default` if not specified.

Example: `deepwiki_cache_azuredevops_myorg_myrepo_en_comprehensive_main.json`

**Legacy fallback:** If no branch-suffixed file exists, falls back to `deepwiki_cache_{repo_type}_{owner}_{repo}_{language}_{mode}.json`.

**`read_wiki_cache(owner, repo, type, language, comprehensive, branch)`**

```
1. Build filename with branch suffix
2. Try loading from Azure Blob (if configured)
   └─ Fallback to legacy filename (no branch)
3. Try loading from local ~/.adalflow/wikicache/
   └─ Fallback to legacy filename
4. Return WikiCacheData or None
```

**`save_wiki_cache(request: WikiCacheRequest)`**

```
1. Build WikiCacheData from request
2. Serialize to JSON (using Pydantic .model_dump())
3. If blob configured → save to Azure Blob and return
4. Otherwise → save to local disk: ~/.adalflow/wikicache/{filename}.json
5. Supports is_partial=True for checkpoint saves during generation
```

### Export (`export.py`)

Two export formats for wiki content download:

**Markdown export** (`generate_markdown_export()`):
```markdown
# Wiki Title

## Table of Contents
- [Page 1](#page-1)
- [Page 2](#page-2)

---
# Page 1
[page content]

### Related Pages
- Page 2
- Page 3
---
```

**JSON export** (`generate_json_export()`):
```json
{
  "title": "Wiki Title",
  "pages": [
    {"id": "1", "title": "...", "content": "...", "filePaths": [...]}
  ],
  "metadata": {"exportedAt": "...", "repoUrl": "..."}
}
```

### XML Repair (`xml_repair.py`)

Azure OpenAI content filters can truncate LLM responses mid-XML. This module recovers partial wiki structures.

**`close_open_tags(xml_text)`** — Appends missing closing tags by tracking open/close tag stack. Handles self-closing tags and attributes.

**`repair_wiki_structure_xml(xml_text)`** — Three recovery strategies:

1. **Truncate after last `</page>`** — Discard incomplete page, close remaining section/wiki tags
2. **Extract stub pages from `<page_ref>`** — If no complete pages, create minimal page stubs from section references
3. **Minimal fallback** — Extract title/description only, create a single "Overview" page

```
LLM output (truncated):
  <wiki_structure>
    <title>My Wiki</title>
    <sections>
      <section id="1"><title>Intro</title>
        <pages><page_ref>1</page_ref></pages>
      </section>
    </sections>
    <pages>
      <page id="1"><title>Overview</title>
        <file_path>README.md</fi    ← TRUNCATED HERE
        
repair_wiki_structure_xml() →
  <wiki_structure>
    <title>My Wiki</title>
    <sections>...</sections>
    <pages>
      <page id="1"><title>Overview</title></page>
    </pages>
  </wiki_structure>
```

## Usage

```python
# Read cached wiki
from backend.modules.wiki import read_wiki_cache
cache = await read_wiki_cache(
    owner="myorg", repo="myrepo", type="azuredevops",
    language="en", comprehensive=True, branch="main",
)

# Save wiki cache (from processor)
from backend.modules.wiki import save_wiki_cache, WikiCacheRequest
await save_wiki_cache(WikiCacheRequest(
    repo=repo_info, language="en", comprehensive=True,
    wiki_structure=structure, generated_pages=pages,
    provider="azure", model="gpt-4o",
    commit_hash="abc123", indexed_at="2026-03-21T12:00:00Z",
))

# Export
from backend.modules.wiki import generate_markdown_export
markdown = generate_markdown_export(pages)
```

## Dependencies

- **Invokes:** `clients/blob_client` (Azure Blob storage), `repository/models` (RepoInfo)
- **Invoked by:** `processor/` (save cache after generation), Next.js API routes (`/api/wiki_cache`, `/api/wiki/projects`)
