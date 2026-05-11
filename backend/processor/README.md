# Processor Module

Standalone CLI pipeline that generates AI wikis from code repositories.

## Responsibility

Runs as a one-shot CLI command — no server required. Transforms a code repository into a structured, AI-generated wiki saved as JSON.

**Pipeline:** Clone repo → Build codemap → Embed documents → Generate wiki → Save cache

<details>
The pipeline steps differ between local/Docker and cloud modes:

**Local / Docker mode** (FAISS retrieval):

Step 1 — Clone: Git clone to `~/.adalflow/repos/`. 

Step 2 — Embed (document.py + indexer.py):
- Phase A: Walk filesystem, collect file paths (~300B/file, lightweight)
- Phase B+C (fused): Read files in batches of FILE_BATCH_SIZE, split into chunks,
  embed immediately, save to disk, release. Each batch bounded at ~120 MB.
- Documents returned with vectors for FAISS construction (no disk reload).

Step 3 — Build FAISS & Generate Wiki (retriever.py + wiki_generator.py):
- Build FAISS in-memory index from returned documents
- Strip .vector from docs (saves ~12KB/chunk)
- Build file-path index for O(1) lookup
  
  For each page (~20-25): query FAISS for top-40 chunks → LLM generates page
  File-priority chunks capped at 80, context capped at 120K chars.
  
Step 4 — Save cache: Serialize WikiCacheData to JSON on disk/blob (~500KB-5MB)

**Cloud mode** (Azure AI Search retrieval, no FAISS):

Step 1 — Clone: Same as local.

Step 2 — Embed (cloud): Fused pipeline saves vectors to blob.
- skip_accumulate=True — no document list held in memory (~120 MB peak).

Step 3 — Push to AI Search: Load vectors from blob in batches, push to index, trigger indexer, wait for completion (poll with timeout).

Step 4 — Generate Wiki (cloud): Lightweight RAG with prepare_for_cloud().
- Query Azure AI Search hybrid (BM25 + vector) for each page. No FAISS.

Step 5 — Save cache: Same as local.

</details>

## Files

| File | Purpose |
|------|---------|
| `code_processor.py` | CLI entry point: argument parsing, auth, mode dispatch, step functions |
| `aml_dispatcher.py` | Cloud entry point: copy config, setup AML resources, exit |
| `wiki_generator.py` | LLM-based structure + content generation, XML parsing |
| `codemap_generator.py` | Codemap processing: `expand_file_paths()`, `summarize_codemap()` |
| `cloud_setup.py` | Config overlay writers + Azure AI Search / AML pipeline management |
| `code_index_schema.json` | Azure AI Search index schema (fields, vectors, scoring profiles) |

## Architecture: Mode as a Switch

`--mode` is the single switch that controls config source, auth method, storage backend, and retrieval engine. Two separate CLIs exist:

| CLI | Who runs it | Purpose |
|-----|------------|---------|
| `python -m backend.processor.aml_dispatcher` | User on local machine | Setup cloud resources, then exit |
| `python -m backend.processor.code_processor` | User (local/docker) or AML pipeline (cloud) | Process a repository |

### Mode Flows

```
┌─────────────────────────────────────────────────────────────────────┐
│ LOCAL MODE (--mode=local)                                           │
│ Config: backend/config/                                             │
│ Auth:   PAT from env → Git Credential Manager (AAD/SSO natively)    │
│ Steps:  clone(local) → codemap → embed(FAISS) → wiki(FAISS)          │
│         → save(local)                                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DOCKER MODE (--mode=docker)                                         │
│ Config: backend/config/.local/  (cloud services disabled)           │
│ Auth:   PAT from env only (error if missing)                        │
│ Steps:  clone(local) → embed(FAISS, API key) → wiki(FAISS)         │
│         → save(local)                                               │
│ Note:   Outer shell builds image + launches container with          │
│         --mode=docker. Inside container, detected via               │
│         _DEEPWIKI_INSIDE_DOCKER env var.                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ CLOUD MODE — Dispatcher (aml_dispatcher.py)                         │
│ Runs on: User's local machine                                       │
│ Steps:  1. Copy config to .cloud/ (enable blob/search/AML)          │
│         2. Setup AML: compute, AI Search index/indexer, image,      │
│            pipeline, upload code                                    │
│         3. Exit — pipeline runs automatically on schedule            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ CLOUD MODE — Processing (code_processor.py --mode=cloud)            │
│ Runs on: Inside AML compute (triggered by scheduled pipeline)       │
│ Config: backend/config/.cloud/  (blob/search/AML enabled)           │
│ Auth:   UMI from infra.json managed_identity.client_id              │
│ Steps:  clone(AML temp) → embed(→blob, no FAISS)                    │
│         → push vectors to search → wait for indexer                 │
│         → wiki(AI Search) → save(→blob)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Config Overlay System

Each mode reads from a different config directory. The overlays are generated
copies of `backend/config/*.json` with mode-specific overrides:

| Directory | Generated by | Overrides |
|-----------|-------------|-----------|
| `config/` | — (default) | None — base config |
| `config/.cloud/` | `write_cloud_config()` | `azure_blob_storage`, `azure_ai_search`, `azure_ml` → `enabled: true` |
| `config/.local/` | `write_docker_config()` | `azure_blob_storage`, `azure_ai_search`, `azure_ml` → `enabled: false` |

Both `.cloud/` and `.local/` are in `.gitignore`. The `.cloud/` directory is
NOT in `.amlignore` so it gets uploaded to AML as part of the code snapshot.

`config.py` uses `set_config_dir()` to switch which directory `load_json_config()`
reads from. This is called in `main()` before any config is loaded.

## Pipeline Stages

```
Stage 1: Clone Repository
    git clone → local disk (all modes)
    Extract HEAD commit hash for citation URLs

Stage 2: Build Codemap (AST Analysis)
    Tree-sitter AST parsing for 8+ languages (Python, JS/TS, Java, Go, C#, C/C++)
    Extracts: symbols (functions, classes, methods), edges (imports, calls, inheritance)
    Codemap summary injected into wiki structure prompt for architecture-aware page layout
    Codemap edges used to expand file_paths for wiki page retrieval

Stage 3: Embed Documents (RAG Preparation)
    Fused read→split→embed→save pipeline, bounded per batch (~120 MB)
    Local/Docker: save vectors to local disk, accumulate for FAISS
    Cloud: save vectors to blob (skip_accumulate=True, no FAISS)

Stage 3.5: Push to AI Search (cloud only)
    Cloud-mode only — runs after Stage 3 and before Stage 4:
    Load vectors → push to AI Search index → trigger indexer →
    wait for completion (poll with timeout)

Stage 4: Generate Wiki
    File tree + README + codemap summary → LLM generates XML structure (pages + sections)
    Validates page file_paths exist in repo; logs semantic-only pages
    For each page:
      - Expand file_paths with codemap-connected files (imports, calls)
      - Build expanded retrieval query (title + description + related page titles)
      - RAG retrieval with file-priority + semantic merge
      - Full context passed to LLM (no truncation — quality is critical)
    Optional review pass: second LLM call to verify accuracy and fix Mermaid diagrams
    Local/Docker: retrieval via FAISS (file-priority capped at 80 chunks)
    Cloud: retrieval via Azure AI Search (hybrid: text + vector)

Stage 5: Save Wiki Cache
    Assemble WikiCacheData → save as JSON
    Local/Docker: ~/.adalflow/wikicache/*.json
    Cloud: blob deepwiki-data/wikicache/
```

## Authentication

`resolve_auth(mode)` — standalone function, returns access token.

| Mode | Strategy |
|------|----------|
| **local** | `REPO_ACCESS_TOKEN` / `ADO_PAT` env → Git Credential Manager (no token injected, GCM handles AAD/SSO natively) |
| **docker** | `REPO_ACCESS_TOKEN` env only → error if missing |
| **cloud** | `DefaultAzureCredential(managed_identity_client_id=...)` from `.cloud/infra.json` |

The Azure DevOps token scope is `499b84ac-1321-427f-aa17-267ca6975798/.default`.

## Wiki Structure Generation

The LLM receives the repo's file tree, README, and codemap summary, then outputs XML:

```xml
<wiki_structure>
  <title>Repository Wiki</title>
  <sections>
    <section id="1">
      <title>Getting Started</title>
      <pages><page_ref>1</page_ref><page_ref>1.1</page_ref></pages>
    </section>
  </sections>
  <pages>
    <page id="1">
      <title>Overview</title>
      <file_path>README.md</file_path>
      <importance>high</importance>
    </page>
  </pages>
</wiki_structure>
```

Parsed by `_parse_structure_xml()` with dash-to-dot ID normalization,
duplicate resolution, content filter retry, and truncated XML repair.

## Cloud Resources (managed by `cloud_setup.py`)

- **AI Search Index** — One per `{owner}_{repo}_{branch}`, stores vector embeddings
- **AI Search Data Source** — Points to blob `deepwiki-data/vectors/{repo}_{branch}/`
- **AI Search Indexer** — Scheduled (default `PT24H`), auto-syncs blob vectors to index
- **AML Environment** — Auto-registered from `Dockerfile.processor`
- **AML Compute Cluster** — Shared, auto-scales 0→4 instances, 600s idle timeout, UMI attached
- **AML Scheduled Pipeline** — One per repo+branch, recurrence schedule (default: every 480h)
- **Teardown** — `teardown_cloud_resources()` deletes indexer + data source + index + pipeline

## Usage

```bash
# Local mode with PAT
export REPO_ACCESS_TOKEN="your-pat"
python -m backend.processor.code_processor \
    --repo="https://dev.azure.com/org/proj/_git/repo" \
    --branch=main --mode=local

# Local mode with config file
python -m backend.processor.code_processor --config=backend/run.json

# Docker mode (builds image, runs container)
python -m backend.processor.code_processor \
    --repo="https://dev.azure.com/org/proj/_git/repo" \
    --branch=main --mode=docker

# Cloud mode — setup AML resources (run once from your machine)
python -m backend.processor.aml_dispatcher --config=backend/run.json

# Cloud mode — processing (runs automatically inside AML pipeline)
# AML command: python -m backend.processor.code_processor --mode=cloud ...
```

Config file format (`run.json`):
```json
{
  "repo": "https://dev.azure.com/org/proj/_git/repo",
  "branch": "main",
  "mode": "local",
  "language": "en"
}
```

CLI args override config file values.

## Content-Filter Auto-Relax (GuardSession)

The pipeline body in `_process()` is wrapped in
[`backend.utils.guard_session.GuardSession`](../utils/guard_session.py):
on entry it snapshots the Azure OpenAI RAI policy bound to the chat
and reasoning deployments, on a `content_filter` `BadRequestError`
the AzureAIClient retry decorator asks the session to relax one
safe-to-toggle filter row (e.g. `Profanity`) and retries the call
once, and on exit the original policy is restored byte-identical
via `If-Match: <etag>`. See
[content_filter_autorelax_plan.md](content_filter_autorelax_plan.md)
for the full design.

Kill switches (any one disables the feature):

| Variable | Effect |
|----------|--------|
| `_DEEPWIKI_INSIDE_DOCKER=1` | Auto-set by `--mode=docker`. Session is a no-op (snapshot would need ARM access not granted to the container). |
| `DEEPWIKI_GUARD_CHECKER_DISABLED=1` | Hard kill — no GET, no PUT, no contextvar. Use to roll back if the feature misbehaves. |
| `DEEPWIKI_AUTO_RELAX_FILTERS=0` | Snapshot still runs but `relax()` becomes a no-op. Useful in audit-only environments. |

The session needs `Microsoft.CognitiveServices/.../raiPolicies/{read,write}`
on the AOAI account. The custom role definition is at
[Deployments/parameters/DeepWikiRAIPolicyManager.RoleDefinition.json](../../Deployments/parameters/DeepWikiRAIPolicyManager.RoleDefinition.json);
see [Deployments/permission.md](../../Deployments/permission.md) for the
`az role definition create` / `az role assignment create` commands.

Without that role, `GuardSession` enters degraded mode at startup
(one WARNING line) and the pipeline runs unchanged with no
auto-relax.

Inspect what the session would snapshot for the next run:

```bash
python -m backend.utils.guard_session --inspect
```

## Dependencies

- **Invokes:** `repository/git_ops` (clone), `codemap/graph_builder` (AST analysis), `embedder/` (RAG), `wiki/cache` (save), `clients/azureai_client` (LLM), `clients/search_client` (AI Search), `clients/storage` (blob/local), `promptstore/` (wiki templates), `promptstore/codemap` (codemap formatting), `processor/codemap_generator` (codemap summarization)
- **Invoked by:** CLI directly, Docker container, AML pipeline job
