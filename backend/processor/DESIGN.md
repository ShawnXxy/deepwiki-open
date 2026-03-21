# Processor Module — Design Document

> **Module:** `backend/processor/`
> **Purpose:** Standalone CLI pipeline that generates AI wikis from code repositories.
> **Last updated:** 2026-03-21

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Execution Modes](#execution-modes)
4. [Pipeline Stages](#pipeline-stages)
5. [File Inventory](#file-inventory)
6. [Data Flow](#data-flow)
7. [Authentication Chain](#authentication-chain)
8. [Storage Layout](#storage-layout)
9. [Configuration](#configuration)
10. [LLM Interaction](#llm-interaction)
11. [Error Handling & Retry](#error-handling--retry)
12. [Output Format](#output-format)
13. [Docker Packaging](#docker-packaging)
14. [Cloud Mode (Azure ML)](#cloud-mode-azure-ml)
15. [Zero-Downtime Reprocessing](#zero-downtime-reprocessing)
16. [Design Decisions](#design-decisions)

---

## Overview

The processor is a **server-less, batch CLI** that transforms a code repository into a structured, AI-generated wiki. It operates independently of the FastAPI chat server and the Next.js frontend — its only output is a JSON cache file that the frontend reads at render time.

```
code_processor.py  ──▶  wiki_generator.py  ──▶  ~/.adalflow/wikicache/*.json
                                                         │
                                              Next.js reads via API route
```

**Key properties:**
- No running server required — runs as a one-shot CLI invocation
- Produces self-contained JSON — frontend needs zero backend connectivity
- Supports three execution modes: `local`, `docker`, `cloud`
- Fully stateless — all state is persisted to disk or Azure Blob

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    code_processor.py (CLI)                    │
│  Parses args ─▶ Resolves auth ─▶ Orchestrates pipeline       │
└────────┬─────────────────────────────────────────────────────┘
         │
         │  1. Clone/pull
         ▼
┌─────────────────────┐     ┌─────────────────────────────────┐
│  repository/        │     │  embedder/                       │
│  ├─ git_ops.py      │     │  ├─ indexer.py   (DatabaseMgr)  │
│  └─ file_content.py │     │  ├─ document.py  (read + split) │
└─────────────────────┘     │  ├─ code_splitter.py (boundaries)│
         │                  │  ├─ retriever.py (RAG + FAISS)   │
         │  2. Embed        │  └─ tokenizer.py (token limits)  │
         ▼                  └──────────────────────────────────┘
┌─────────────────────┐                │
│  clients/           │                │  3. Generate wiki
│  ├─ azureai_client  │◄───────────────┤
│  ├─ embedding_client│                ▼
│  ├─ vector_storage  │     ┌──────────────────────────────────┐
│  └─ storage         │     │  wiki_generator.py               │
└─────────────────────┘     │  ├─ build_file_tree()            │
                            │  ├─ LLM: wiki structure (XML)    │
                            │  ├─ LLM: each page (Markdown)    │
                            │  └─ Assemble WikiCacheData       │
                            └──────────┬───────────────────────┘
                                       │  4. Save
                                       ▼
                            ┌──────────────────────────────────┐
                            │  wiki/cache.py                   │
                            │  └─ save_wiki_cache()            │
                            │     → ~/.adalflow/wikicache/     │
                            └──────────────────────────────────┘

 (Cloud mode only)
         │  5. Push vectors
         ▼
┌─────────────────────┐
│  cloud_setup.py     │
│  ├─ AI Search index │
│  └─ AML pipeline    │
└─────────────────────┘
```

---

## Execution Modes

| Mode | Entry | Auth | Storage | Scheduling |
|------|-------|------|---------|------------|
| **local** | `python -m backend.processor.code_processor --repo=URL --branch=main --mode=local` | PAT from `.env`, or `az login` / `DefaultAzureCredential` | Local disk (`~/.adalflow/`) | Manual |
| **docker** | Same CLI with `--mode=docker` | PAT from `backend/.env` (required) | Mounted volume (`~/.adalflow:/root/.adalflow`) | Manual |
| **cloud** | Same CLI with `--mode=cloud` | MSI (`DefaultAzureCredential`) | Azure Blob + AI Search | AML recurrence schedule (default: every 480h) |

### Mode Dispatch

```python
main()
  ├─ --mode=local  → run_code_processor()        # Direct execution
  ├─ --mode=docker → _run_docker_mode()           # Build + docker run
  └─ --mode=cloud  → _run_cloud_mode()            # Setup resources + run + push
```

### Config File Support

CLI arguments can be provided via a JSON config file:

```bash
python -m backend.processor.code_processor --config=run.json --branch=dev
```

```json
{
  "repo": "https://dev.azure.com/org/proj/_git/repo",
  "branch": "main",
  "mode": "local",
  "language": "en"
}
```

CLI args override config file values. `--branch=dev` overrides `"branch": "main"` above.

---

## Pipeline Stages

### Stage 1: Clone Repository

```
code_processor.py → git_ops.download_repo()
```

- Clones (or pulls if `force_update=True`) the repo to `~/.adalflow/repos/{owner}_{repo}/`
- Extracts HEAD commit hash via `git rev-parse HEAD` for citation URLs
- Supports Azure DevOps repos with PAT-based HTTPS auth

### Stage 2: Embed Documents (RAG Preparation)

```
code_processor.py → RAG.prepare_retriever() → DatabaseManager.prepare_database()
```

**Sub-steps:**

1. **Read all documents** (`document.py::read_all_documents()`)
   - Walks the repo directory recursively
   - Applies `FileFilter` rules from `repo.json` (exclusion mode by default)
   - Reads code files (`.py`, `.ts`, `.js`, etc.) and docs (`.md`, `.txt`, `.json`)
   - Respects `max_file_size_mb` limit (10 MB default)

2. **Code-aware splitting** (`code_splitter.py::split_and_enrich_documents()`)
   - Detects logical boundaries (functions, classes, blocks) using regex patterns
   - Splits at boundaries instead of arbitrary token positions
   - Enriches chunk text with structural prefix: `[File: path/to/file.py | Language: python | Section: function/class/imports]`
   - Falls back to token-based splitting for non-code files

3. **Embedding** (`document.py::transform_documents_and_save_as_json()`)
   - Calls Azure OpenAI `text-embedding-3-large` (3072 dimensions) via `SafeEmbedder`
   - Processes in batches (configured in `embedder.json`)
   - Uses embedding cache (`~/.adalflow/embedding_cache/`) for deduplication
   - Saves each chunk as individual JSON file in `~/.adalflow/vectors/{owner}_{repo}_{branch}/`

4. **FAISS index construction** (`retriever.py::RAG.prepare_retriever()`)
   - Validates embedding dimensions (filters mismatched sizes)
   - Builds FAISS index for semantic search
   - Configurable `top_k` from `embedder.json`

### Stage 3: Generate Wiki

```
code_processor.py → wiki_generator.generate_wiki()
```

**Sub-steps:**

1. **Build file tree** — Walks repo directory, produces indented tree string (max 6 levels)
2. **Read README** — Finds and reads `README.md` (truncated at 15,000 chars)
3. **Generate structure via LLM** — Sends file tree + README to Azure OpenAI with `WIKI_STRUCTURE_PROMPT`
   - Returns XML: `<wiki_structure>` with `<title>`, `<sections>`, `<page>` elements
   - Content filter retry: if response is too short, retries with directory-only tree and sanitized README
4. **Parse XML structure** — Extracts pages (id, title, filePaths, importance, relatedPages) and sections hierarchy
5. **Generate each page via LLM** — For each page:
   - Retrieves context via `RAG.call_with_file_filter()` (file-priority + semantic)
   - Formats context with commit-pinned source URLs
   - Builds page prompt with `WIKI_PAGE_CONTENT_PROMPT`
   - LLM generates Markdown with Mermaid diagrams and `[[deepwiki://id]]` cross-links
6. **Assemble `WikiCacheData`** — Combines structure + all generated pages into single object

### Stage 4: Save Wiki Cache

```
code_processor.py → wiki/cache.save_wiki_cache()
```

- Serializes `WikiCacheData` to JSON
- Saves to `~/.adalflow/wikicache/deepwiki_cache_{owner}_{repo}_{branch}_{lang}_{mode}.json`
- In blob mode: also uploads to Azure Blob container

### Stage 5: Push to AI Search (Cloud Mode Only)

```
_run_cloud_mode() → _push_vectors_to_search()
```

- Loads all vector JSON files from local storage
- Pushes to Azure AI Search index via `search_client.push_documents()`
- Enables cloud-based semantic search for the Ask/Chat feature

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | — | Package marker |
| `code_processor.py` | ~300 | CLI entry point, arg parsing, mode dispatch, auth resolution, pipeline orchestration |
| `wiki_generator.py` | ~340 | LLM-based structure + content generation, XML parsing, WikiCacheData assembly |
| `cloud_setup.py` | ~210 | Azure AI Search index management, AML pipeline creation/scheduling |

### Key Functions

**code_processor.py:**
| Function | Purpose |
|----------|---------|
| `main()` | CLI entry: loads `.env`, sets up logging, dispatches to mode handler |
| `_parse_args()` | Argument parsing with `--config` file support |
| `_extract_owner_repo()` | Extracts org + repo from ADO URLs (handles `dev.azure.com` and `visualstudio.com`) |
| `run_code_processor()` | Main pipeline: clone → embed → generate → save |
| `_run_docker_mode()` | Builds `Dockerfile.processor` image, runs with volume mount |
| `_run_cloud_mode()` | Sets up cloud resources, runs processor, pushes vectors |

**wiki_generator.py:**
| Function | Purpose |
|----------|---------|
| `generate_wiki()` | Full wiki generation orchestrator |
| `build_file_tree()` | Recursively walks repo to build tree string |
| `read_readme()` | Reads and truncates README (15K char limit) |
| `_call_llm()` | Direct (non-streaming) Azure OpenAI call |
| `_parse_structure_xml()` | Parses LLM XML output into pages + sections |

**cloud_setup.py:**
| Function | Purpose |
|----------|---------|
| `setup_cloud_resources()` | Creates AI Search index + AML scheduled pipeline |
| `teardown_cloud_resources()` | Deletes index + pipeline |
| `_ensure_compute()` | Creates/verifies AML compute cluster |
| `_create_or_update_pipeline()` | Creates AML command job with recurrence schedule |

---

## Data Flow

```
                    ┌─────────────┐
                    │  ADO Repo   │
                    │  (remote)   │
                    └──────┬──────┘
                           │ git clone / pull
                           ▼
                    ┌─────────────┐
                    │  Local Repo │  ~/.adalflow/repos/{owner}_{repo}/
                    └──────┬──────┘
                           │ read_all_documents()
                           ▼
                    ┌─────────────┐
                    │  Documents  │  List[Document] with metadata
                    └──────┬──────┘
                           │ split_and_enrich_documents()
                           ▼
                    ┌─────────────┐
                    │  Chunks     │  Code-aware boundary splits
                    └──────┬──────┘
                           │ Azure OpenAI Embeddings API
                           ▼
                    ┌─────────────┐
                    │  Vectors    │  ~/.adalflow/vectors/{owner}_{repo}_{branch}/
                    │  (JSON)     │  Individual JSON per chunk
                    └──────┬──────┘
                           │ FAISS index build
                           ▼
                    ┌─────────────┐
                    │  Retriever  │  In-memory FAISS index
                    └──────┬──────┘
                           │ LLM calls (structure + pages)
                           ▼
                    ┌─────────────┐
                    │  Wiki Cache │  ~/.adalflow/wikicache/*.json
                    │  (JSON)     │  WikiCacheData (structure + pages)
                    └──────┬──────┘
                           │ (Next.js reads)
                           ▼
                    ┌─────────────┐
                    │  Frontend   │  /api/wiki_cache → fs.readFile
                    └─────────────┘
```

---

## Authentication Chain

The processor resolves authentication in priority order:

```
1. PAT from environment (REPO_ACCESS_TOKEN, ADO_PAT, AZURE_DEVOPS_PAT)
   └─ Used directly as git credential
   └─ Required in Docker mode

2. Azure Identity (DefaultAzureCredential)
   ├─ MSI with explicit client_id  →  Azure Container Apps (cloud mode)
   ├─ Azure CLI token              →  Local development (az login)
   ├─ VS Code credential           →  Local development
   └─ Environment credential       →  CI/CD pipelines
   Token scope: "499b84ac-1321-427f-aa17-267ca6975798/.default" (Azure DevOps)

3. Failure
   └─ Exit with error message listing options
```

**Azure OpenAI authentication** is separate — handled by `AzureAIClient`:
- API key from `AZURE_OPENAI_API_KEY` env var (Docker/local)
- MSI token via `DefaultAzureCredential` (cloud)
- Configured in `infra.json` (endpoint, deployment name, API version)

---

## Storage Layout

All persistent data lives under `~/.adalflow/`:

```
~/.adalflow/
├── repos/                              # Cloned repositories
│   └── {owner}_{repo}/                 # e.g., msdata_orcasql-myfile/
│       └── (full git working tree)
│
├── vectors/                            # Embedding vectors (JSON chunks)
│   └── {owner}_{repo}_{branch}/        # e.g., msdata_orcasql-myfile_main/
│       ├── src/backend/main_001.json   # Chunk 1 of main.py
│       ├── src/backend/main_002.json   # Chunk 2 of main.py
│       ├── src/utils/helper_001.json
│       └── README_001.json
│
├── databases/                          # Legacy pkl format (backward compatible)
│   └── {owner}_{repo}_{branch}.pkl
│
├── wikicache/                          # Generated wiki JSON
│   └── deepwiki_cache_{owner}_{repo}_{branch}_{lang}_{mode}.json
│
└── embedding_cache/                    # API response deduplication cache
    └── (hash-keyed cache files)
```

### Vector JSON Format

Each chunk file in `vectors/` contains:

```json
{
  "file_path": "src/backend/main.py",
  "chunk_index": 0,
  "total_chunks": 5,
  "text": "[File: src/backend/main.py | Language: python | Section: function]\ndef process_data():\n    ...",
  "vector": [0.123, 0.456, ...],
  "meta_data": {
    "file_path": "src/backend/main.py",
    "type": "py",
    "url": "https://dev.azure.com/org/proj/_git/repo?path=/src/backend/main.py&version=GBmain",
    "raw_content": "..."
  }
}
```

---

## Configuration

All configuration files live in `backend/config/`:

| File | Purpose | Key Settings |
|------|---------|--------------|
| `infra.json` | Azure service endpoints | `azure_openai.endpoint`, `azure_openai.deployment`, `azure_openai.embedding_deployment`, `azure_ml.*`, `azure_search.*` |
| `repo.json` | File filtering rules | `excluded_dirs` (`.git`, `node_modules`, etc.), `excluded_files` (`*.pyc`, `*.log`, etc.) |
| `embedder.json` | Embedding parameters | `batch_size`, `chunk_size`, `overlap`, `top_k` |
| `lang.json` | Supported wiki languages | Language codes → display names (`en`, `ja`, `zh`, `es`, `kr`, etc.) |

### Type-Safe Configuration

All configs are loaded into Pydantic models (`backend/types/config_types.py`):

```python
InfraConfig         # Azure endpoints, MSI client ID, service configs
EmbedderConfig      # Embedding batch/chunk/overlap settings
FileFiltersConfig   # Exclusion dirs and patterns
LanguageConfig      # Supported languages
```

Accessed via cached functions in `backend/config.py`:
- `get_infra_config()` → `InfraConfig`
- `get_model_config(provider, model)` → dict with `model_kwargs`
- `get_azure_ai_client()` → singleton `AzureAIClient`
- `get_azure_deployment_name()` → str
- `is_search_configured()` → bool
- `get_aml_config()` → `AzureMLConfig | None`

---

## LLM Interaction

### Models Used

| Purpose | Model | Config Source |
|---------|-------|--------------|
| Wiki structure generation | Azure OpenAI Chat (e.g., `gpt-4o`) | `infra.json → azure_openai.deployment` |
| Wiki page generation | Same deployment | Same |
| Document embeddings | Azure OpenAI Embeddings (`text-embedding-3-large`) | `infra.json → azure_openai.embedding_deployment` |

### Prompt Templates

Located in `backend/promptstore/`:

| Template | Input Placeholders | Output |
|----------|--------------------|--------|
| `wiki_structure.py::WIKI_STRUCTURE_PROMPT` | `{file_tree}`, `{readme}`, `{owner}`, `{repo}`, `{language_name}` | XML: `<wiki_structure>` with pages + sections |
| `wiki_structure.py::WIKI_STRUCTURE_CONCISE_PROMPT` | Same as above | Smaller XML (4-6 pages) |
| `wiki_page.py::WIKI_PAGE_CONTENT_PROMPT` | `{page_title}`, `{file_paths_list}`, `{context_text}`, `{page_catalog}`, `{language_name}` | Markdown with Mermaid diagrams |

### Content Safety Handling

If the initial structure generation returns insufficient content (< 200 chars or missing `<wiki_structure>` tag), the processor retries with:
1. Directory-only file tree (strips filenames)
2. Sanitized README placeholder

This handles Azure OpenAI content filter rejections for repos with flagged file/variable names.

### RAG Retrieval Strategy

For each wiki page, the retriever uses a **file-priority + semantic** merge:

1. **File-filtered chunks:** All chunks from the page's declared `filePaths` (exhaustive)
2. **Semantic search:** FAISS top-k query using the page title
3. **Merge:** File chunks first (priority), then semantic chunks (deduplicated)

This ensures pages reference their declared source files while also discovering related context.

---

## Error Handling & Retry

### LLM Call Errors

`AzureAIClient` handles:
- **Rate limiting (429):** Parses `Retry-After` header, exponential backoff
- **Content filter (400):** Logged as warning, returns partial content
- **Transient errors (5xx):** Retried with backoff
- **API key `\r`:** Stripped on initialization (Windows `.env` line endings)

### Credential Protection

`_mask_secrets()` in `azureai_client.py` replaces sensitive patterns in exception messages:
- Bearer tokens → `Bearer ***`
- API keys (32+ hex chars) → first 6 chars + `***`
- Raw byte strings → `<binary data>`

### Pipeline Failures

- Clone failure → exit with error (no partial state)
- Embedding failure → raises `ValueError` (no valid embeddings)
- Structure generation failure → attempts content-filter retry, then raises
- Page generation failure → logs error, stores error message as page content
- Cache save failure → logs warning, prints failure message

---

## Output Format

### WikiCacheData (JSON)

```json
{
  "wiki_structure": {
    "id": "wiki",
    "title": "Repository Wiki Title",
    "description": "High-level description",
    "pages": [
      {
        "id": "1",
        "title": "Overview",
        "content": "",
        "filePaths": ["README.md"],
        "importance": "high",
        "relatedPages": ["1.1", "2"]
      }
    ],
    "sections": [
      {
        "id": "1",
        "title": "Getting Started",
        "pages": ["1", "1.1", "1.2"],
        "subsections": null
      }
    ],
    "rootSections": ["1", "2", "3"]
  },
  "generated_pages": {
    "1": {
      "id": "1",
      "title": "Overview",
      "content": "# Overview\n\nThis page covers...",
      "filePaths": ["README.md"],
      "importance": "high",
      "relatedPages": ["1.1", "2"]
    }
  },
  "repo": {
    "owner": "myorg",
    "repo": "myrepo",
    "type": "azuredevops",
    "branch": "main",
    "repoUrl": "https://dev.azure.com/myorg/proj/_git/myrepo"
  },
  "provider": "azure",
  "model": "gpt-4o",
  "comprehensive": true,
  "is_partial": false,
  "commit_hash": "abc123def456...",
  "indexed_at": "2026-03-21T12:00:00+00:00"
}
```

### Page Content Features

Generated Markdown pages include:
- **Relevant source files block:** `<details>` with commit-pinned links
- **Mermaid diagrams:** Architecture, data flow, state machines
- **Cross-page links:** `[[deepwiki://2.1]]` syntax for internal navigation
- **Code examples:** Summarized, not verbatim (content safety)

---

## Docker Packaging

### Dockerfile.processor

A lightweight image (~200 MB) containing only Python + Git:

```dockerfile
FROM python:3.11-slim
# Install git + Python deps via poetry
# No Next.js, no nginx, no FastAPI
ENTRYPOINT ["python", "-m", "backend.processor.code_processor"]
```

### Docker Run Flow

```
_run_docker_mode(args)
  1. Build image from Dockerfile.processor
  2. Mount ~/.adalflow → /root/.adalflow
  3. Pass --env-file backend/.env
  4. Pass AZURE_* env vars from host
  5. Run with --mode=local inside container
  6. Wiki cache written to mounted volume
```

The container always runs as `--mode=local` internally — Docker mode is just a packaging wrapper.

---

## Cloud Mode (Azure ML)

### Resource Setup (Idempotent)

`setup_cloud_resources()` creates:

1. **AI Search Index** — One per `{owner}_{repo}_{branch}`
   - Schema matches vector JSON format (text + vector + metadata)
   - Created via `search_client.create_or_update_index()`

2. **AML Compute Cluster** — Shared across all repos
   - Default size: `STANDARD_D2_V2`
   - Auto-scales 0→4 instances
   - 600s idle timeout

3. **AML Scheduled Pipeline** — One per repo+branch
   - Runs `python -m backend.processor.code_processor --repo=URL --branch=B --mode=local`
   - Recurrence: every 480 hours (20 days) by default
   - Pipeline name: `deepwiki-{owner}-{repo}-{branch}` (max 128 chars)

### Cloud Flow

```
_run_cloud_mode()
  1. setup_cloud_resources()     # Create/update index + pipeline
  2. run_code_processor()        # Same as local mode
  3. _push_vectors_to_search()   # Load from vectors/, push to AI Search
```

### Teardown

`teardown_cloud_resources()` removes:
- AI Search index (via `delete_index()`)
- AML schedule (disable → delete)

---

## Zero-Downtime Reprocessing

When `force_reprocess=True`, the processor re-embeds and regenerates the wiki for a repository. The naive approach (delete all vectors → re-embed) creates a downtime window where the chat/RAG system has no embeddings available.

### Problem

```
DELETE all vectors  ─── 0s ───  Start embedding  ─── 5-30 min ───  Done
                    ▲                                              ▲
              DOWNTIME STARTS                              DOWNTIME ENDS
```

During the gap, any Ask/Chat query returns empty context.

### Solution: Incremental Overwrite + Deferred Orphan Cleanup

```
Snapshot old files  ─── 0s ───  Embed (overwrite in-place)  ─── 0s ───  Delete orphans
                                                                        ▲
                                                               ~instant (few files)
```

**No downtime.** Old vector files remain readable throughout embedding. New files overwrite in-place as they're processed. Only orphaned files (from deleted/renamed source files) are cleaned up after completion.

### Implementation

Three phases in `DatabaseManager.prepare_db_index()`:

1. **Snapshot** — `vector_storage.list_files()` captures the set of existing vector file paths
2. **Embed & Overwrite** — `transform_documents_and_save_as_json()` saves new chunks, naturally overwriting files for unchanged paths
3. **Orphan Cleanup** — After embedding completes, compute `orphans = old_files - new_files` and delete them via `vector_storage.delete_files()`

```python
# Phase 1: Snapshot
old_files = vector_storage.list_files(repo_name, branch)    # {"src/old_001.json", ...}

# Phase 2: Embed (overwrites existing, adds new)
transform_documents_and_save_as_json(documents, repo_name, branch)

# Phase 3: Cleanup
new_files = vector_storage.list_files(repo_name, branch)    # {"src/new_001.json", ...}
orphans = old_files - new_files                              # Files that no longer exist
vector_storage.delete_files(repo_name, branch, orphans)      # Remove stale files
```

### What Counts as an Orphan?

| Scenario | Old File | New File | Orphan? |
|----------|----------|----------|---------|
| File unchanged, same chunks | `main_001.json` | `main_001.json` | No (overwritten) |
| File changed, fewer chunks | `main_003.json` | — | Yes |
| File renamed | `old_name_001.json` | `new_name_001.json` | `old_name_001.json` is orphan |
| File deleted from repo | `removed_001.json` | — | Yes |
| New file added | — | `added_001.json` | No (new) |

### Legacy Pkl Handling

Legacy `.pkl` databases are **always deleted upfront** during reprocessing, since they are never served live (the system already migrated to JSON vectors). Only JSON vector files use the zero-downtime approach.

### VectorStorage API

Two new methods support this flow:

- `list_files(repo_name, branch) → set[str]` — Returns relative paths of all JSON chunk files
- `delete_files(repo_name, branch, rel_paths) → int` — Deletes specific files by relative path, cleans empty parent directories

Both methods work transparently with local and Azure Blob storage backends.

---

## Design Decisions

### Why CLI Instead of Server?

Wiki generation is a **batch operation** (5-30 minutes per repo). Running it as a long-lived server would:
- Waste compute between runs
- Complicate scaling (one repo per process)
- Couple wiki generation with the chat service

The CLI model enables: cron scheduling, Docker batch jobs, AML pipelines, and CI/CD integration.

### Why XML for Structure?

LLMs produce structured output more reliably in XML than JSON:
- XML is more forgiving of minor formatting errors
- `xml.etree` parsing is built-in (no dependencies)
- XML attributes (`id="1"`) are cleaner than JSON key juggling
- Repair is straightforward (`close_open_tags()` handles truncation)

### Why File-Priority Retrieval?

Pure semantic search often misses files that are structurally important but textually distant from the page title. By exhaustively including all declared `filePaths` chunks first, pages always reference their intended source files.

### Why Code-Aware Splitting?

Arbitrary token-based splits frequently cut mid-function, producing chunks that:
- Lose semantic coherence (half a function definition)
- Embed poorly (context is split across chunks)
- Miss structural boundaries (class headers separated from methods)

Boundary-aware splitting keeps logical units intact, improving both embedding quality and retrieval relevance.

### Why JSON Vectors Instead of Pickle?

Legacy pickle databases (`*.pkl`) had several problems:
- Must load entire database into memory at once
- Not human-readable or debuggable
- Pickle security concerns (arbitrary code execution on load)
- No incremental update support

JSON vectors (one file per chunk):
- Memory-efficient — load only what's needed
- Inspectable — debug individual chunks
- Incremental — add/remove files without full rebuild
- Safe — no deserialization risks

Legacy pkl format is still supported for backward compatibility.

### Why `~/.adalflow/` on All Platforms?

Docker volume mounting requires a consistent path between host and container:
- Host: `~/.adalflow` (Windows: `C:\Users\{user}\.adalflow`)
- Container: `/root/.adalflow`
- Mount: `-v ~/.adalflow:/root/.adalflow`

Using the OS-specific default (`%APPDATA%` on Windows) would break this mapping.
