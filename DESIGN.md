# DeepWiki Improvement Design Document

**Goal:** Replicate Cognition's DeepWiki feature set for internal use.
**Date:** March 2026 | **Last updated:** March 18, 2026

### Scope Constraints

- **Git source:** Azure DevOps only.
- **UX design language:** Microsoft Fluent 2 design system.

---

## Table of Contents

0. [Implementation Status (Phases 0–2)](#0-implementation-status-phases-02)
1. [Current State vs Cognition](#1-current-state-vs-cognition)
2. [Phase 3: Backend Isolation — Standalone Code Processor](#2-phase-3-backend-isolation--standalone-code-processor)
3. [Phase 3.5: Backend Cleanup — Directory Reorganization](#3-phase-35-backend-cleanup)
4. [Phase 4: Frontend Read-Only Mode](#4-phase-4-frontend-read-only-mode)
5. [Phase 5: UX Polish](#5-phase-5-ux-polish)
6. [File Change Map](#6-file-change-map)
7. [Verification Plan](#7-verification-plan)

---

## 0. Implementation Status (Phases 0–2)

### Phase 0: Content Quality Foundation ✅

| Item | Status |
|------|--------|
| File-path-aware retrieval (`call_with_file_filter`) | ✅ |
| top_k_wiki = 40 | ✅ |
| chunk_enhancer removed (~480 lines deleted) | ✅ |
| Content filter sanitization | ✅ |
| Content filter retry with dir-only file tree | ✅ |
| Content safety prompt instructions | ✅ |
| Mermaid unfencing (10 diagram types) | ✅ |
| Heading IDs + anchor scroll | ✅ |
| Duplicate page ID deduplication | ✅ |
| Orphan page distribution (ID-prefix) | ✅ |
| Browser tab `visibilitychange` resume | ✅ |
| LLM-as-question detection | ✅ |
| UTF-16 BOM encoding fix | ✅ |
| Verbose LLM logging fix | ✅ |
| Mermaid error downgrade (warn not error) | ✅ |
| 15-25 page comprehensive prompt | ✅ |
| No-question/no-duplicate prompt rules | ✅ |
| FAISS top_k_wiki kwarg filtering | ✅ |
| WikiSection recursive model + `model_rebuild()` | ✅ |
| WikiSection `field_validator` for string/object compat | ✅ |
| Recursive section XML parser (replaces flatten) | ✅ |
| Section deduplication in flat list | ✅ |
| Mermaid diagram-end detection (list items, tables) | ✅ |
| Embedding progress + streaming timing logs | ✅ |

### Phase 1: Move Prompt to Backend ✅

| Item | Status |
|------|--------|
| `build_wiki_page_prompt()` in wiki_page.py | ✅ |
| `format_page_catalog()` in wiki_page.py | ✅ |
| `format_file_paths_list()` with commit_hash support | ✅ |
| `get_head_commit_hash()` in git_ops.py | ✅ |
| ws_handler routes wiki_page_request to backend prompt | ✅ |
| ws_handler resolves local repo path for commit hash | ✅ |
| Frontend 120-line prompt template removed | ✅ |
| Frontend sends `page_id`, `page_title`, `page_file_paths` | ✅ |
| `page_id` field added to ChatCompletionRequest | ✅ |
| Wiki cache 422 fix (subsections compat validator) | ✅ |
| Wiki cache route error logging (Request body parsing) | ✅ |

### Phase 2: Content Quality + Resilience ✅

| Item | Status |
|------|--------|
| `source_url.py` — Azure DevOps permalink builder (+ GitHub/GitLab/Bitbucket) | ✅ |
| Citation URLs in `format_context_text()` — repo_url + commit_hash per chunk | ✅ |
| `deepwiki://` cross-page link resolver in Markdown.tsx | ✅ |
| `onNavigateToPage` callback threaded from page.tsx → Markdown | ✅ |
| Page catalog in `build_wiki_page_prompt()` for cross-page links | ✅ |
| Commit hash sent as `<!-- meta:commit_hash=... -->` during streaming | ✅ |
| `commit_hash` + `indexed_at` fields in WikiCacheData model | ✅ |
| `commit_hash` saved/returned in wiki cache routes | ✅ |
| WebSocket `_safe_send()` helper — guards all sends against disconnect | ✅ |
| WebSocket `_safe_close()` helper — graceful close on network breakout | ✅ |
| Streaming loop `_client_connected` flag — stop on first send failure | ✅ |
| Fallback request send paths guarded with `_safe_send` | ✅ |
| WikiTreeView: skip overview page from child list (section header = overview) | ✅ |
| WikiTreeView: skip pages matching subsection IDs (avoid duplicate titles) | ✅ |
| WikiTreeView: section header clickable → selects overview page | ✅ |
| WikiTreeView: removed ID prefix from `renderSectionObj` display | ✅ |

---

## 1. Current State vs Cognition

Reference: `deepwiki.com/fastapi/fastapi` (March 13, 2026 — 9 sections, ~42 pages, commit `11614be9`).

### ✅ Features We Match

| Feature | Cognition | deepwiki-open |
|---------|-----------|---------------|
| 3-level hierarchy | 1, 2.1, 2.1.1 | WikiSection recursive + numbered IDs |
| Numbered sidebar labels | "2.1 Application and Routing" | `renderSectionObj()` shows `{id} {title}` |
| Mermaid diagrams | flowchart, sequence, class | 10 types + auto-unfencing + sanitization |
| `<details>` source block | At top of every page | In page prompt (backend) |
| No orphan pages | Every page in a section | ID-prefix distribution + section dedup |
| Heading anchors | Section links scroll | h1-h4 `id` + smooth scroll |
| Content safety | No raw secrets | Sanitization + prompt instructions |
| File-aware retrieval | Unknown | `call_with_file_filter()`, top_k=40 |
| Backend prompt control | Prompt on server | `build_wiki_page_prompt()` in promptstore |
| Commit-pinned URLs | `blob/11614be9/file.py` | `format_file_paths_list(commit_hash=...)` |
| Commit hash capture | "Last indexed: (11614b)" | `get_head_commit_hash()` in git_ops.py |

### ❌ Remaining Gaps

**P0 — Architecture (blocking all other work):**

| # | Gap | Status |
|---|-----|--------|
| 1 | **Backend isolation** — wiki generation runs inside WebSocket handler (single-threaded, timeout-prone, frontend-orchestrated) | ❌ Need standalone `code_processor.py` CLI |
| 2 | **Frontend read-only mode** — UI still shows generation forms, progress overlays, WebSocket generation loops | ❌ Need to strip generation UI, make frontend a pure wiki reader |
| 3 | **Azure AI Search** — retrieval uses in-process FAISS only (no shared index for production) | ❌ Need `search_client.py` + infra.json config |

**P1 — Content Quality (deferred until architecture is resolved):**

| # | Gap | Status |
|---|-----|--------|
| 4 | Frontend commit_hash extraction — `citationProcessor.tsx` uses `HEAD` | ⚠️ Will be resolved when wiki_generator.py writes commit_hash to cache |
| 5 | "Last indexed" display | ❌ Frontend doesn't show commit_hash/indexed_at from cache |
| 6 | Citation validation | ❌ Post-processing to strip hallucinated citations |
| 7 | File-level summaries | ❌ `file_summarizer.py` not created |

**P2 — UX Polish (deferred):**

| # | Gap |
|---|-----|
| 8 | Wiki search |
| 9 | Share button |
| 10 | Fluent 2 design |

---

## 2. Phase 3: Backend Isolation — Standalone Code Processor

**Goal:** Extract the RAG + wiki generation workflow from the WebSocket handler into a standalone CLI process that works in three modes: local, Docker, and cloud (Azure ML + AI Search).
**Status:** Design complete. Ready to implement.
**Sub-phases:** 3A (core processor, local mode) → 3B (Docker mode) → 3C (cloud infrastructure: AI Search + AML) → 3D (cloud mode integration)

### Current Architecture (problems)

```
Frontend (Next.js)
  ├─ WS /ws/chat {wiki_structure_request}    → FastAPI → LLM (no RAG)
  ├─ WS /ws/chat {wiki_page_request} × N    → FastAPI → clone + embed + FAISS → LLM
  └─ POST /api/wiki_cache                   → save JSON to local/blob
```

- **Single-threaded**: Only one wiki generation at a time
- **WebSocket timeout risk**: Embedding takes ~3 min; keepalive pings mask the fragility
- **Frontend orchestrates**: ~1200 lines of generation logic that belongs server-side
- **No offline/batch capability**: Can't generate wikis from CI/CD or Azure ML

### Target Architecture

```
code_processor.py (standalone CLI / Docker / Azure ML)
  ├─ Clone repo (PAT from .env or MSI from infra.json)
  ├─ Read + split + embed → save vectors to local / Blob
  ├─ [cloud] Create per-repo AI Search index + data source, push vectors
  ├─ Generate wiki structure via LLM
  ├─ Generate all wiki pages via LLM + retrieval (FAISS local, AI Search cloud)
  ├─ Save complete wiki cache to local / Blob
  └─ [cloud] Create/update AML scheduled pipeline for this repo

Frontend (Next.js) — READ-ONLY
  ├─ GET /api/wiki_cache → reads from local or Blob
  ├─ Renders cached wiki (sidebar + pages + Markdown)
  └─ No generation forms, no progress UI, no WebSocket generation
```

### 3.1 New files

| File | Sub-phase | Purpose |
|------|-----------|---------|
| `backend/processor/__init__.py` | 3A | Package init, exports `run_code_processor()` |
| `backend/processor/code_processor.py` | 3A | CLI entry point (all 3 modes) |
| `backend/processor/wiki_generator.py` | 3A | Server-side wiki orchestration (structure → pages → cache) |
| `backend/clients/search_client.py` | 3C | Azure AI Search index management + query + push documents |
| `backend/processor/cloud_setup.py` | 3C | Cloud-mode helpers: per-repo AI Search index/DS + AML pipeline |
| `backend/config/aml_pipeline.json` | 3C | AML compute/schedule config |
| `sample.run.json` | 3A | Template config file for CLI |

### 3.2 CLI interface

```bash
# Option A: Pass params directly
python -m backend.processor.code_processor \
  --repo="https://msdata.visualstudio.com/Database%20Systems/_git/orcasql-myfile" \
  --branch="main" --mode="local"

# Option B: Use a config file
python -m backend.processor.code_processor --config=run.json

# Option C: Config file with CLI overrides (CLI wins)
python -m backend.processor.code_processor --config=run.json --branch="dev"
```

**Parameters:**

| Param | Required | Default | Description |
|-------|----------|---------|-------------|
| `--config` | No | — | Path to JSON config file (e.g. `run.json`) |
| `--repo` | **Yes**\* | — | Azure DevOps repo URL (only ADO supported) |
| `--branch` | **Yes**\* | — | Branch name to process |
| `--mode` | **Yes**\* | — | Execution mode: `local`, `docker`, or `cloud` |
| `--comprehensive` | No (hidden) | `true` | Wiki depth |
| `--language` | No (hidden) | `en` | Wiki language |

\* Required either via CLI args or inside `--config` JSON file.

### 3.3 Mode behavior

| Mode | Retrieval | AI Search | AML Pipeline | Storage | Auth |
|------|-----------|-----------|-------------|---------|------|
| `local` | FAISS | ❌ N/A | ❌ N/A | `~/.adalflow/` | PAT from `.env` |
| `docker` | FAISS | ❌ N/A | ❌ N/A | `~/.adalflow/` (volume) | PAT from `.env` |
| `cloud` | AI Search | ✅ Per-repo index | ✅ Scheduled pipeline | Blob | MSI from `infra.json` |

### 3.4 `--mode=local` flow

```
1. Load PAT from .env
2. Clone → embed → FAISS retrieval → generate wiki → save cache
3. Output: ~/.adalflow/wikicache/ + ~/.adalflow/vectors/
```

### 3.5 `--mode=docker` flow

```
1. Build Docker image, create .local config (disable blob/appinsights)
2. docker run ... python -m backend.processor.code_processor --mode=local
3. Output: ~/.adalflow/ (via volume mount)
```

### 3.6 `--mode=cloud` flow

```
1. Read config from infra.json (AML, AI Search, Blob, OpenAI)
2. Derive repo identifier: {owner}-{repo}-{branch}

3. CREATE/UPDATE AI Search index: "deepwiki-{owner}-{repo}-{branch}"
   - Schema from Deployments/index/code_index_schema.json
   - Unique per repo+branch for isolation

4. CREATE/UPDATE AI Search data source: "ds-{owner}-{repo}-{branch}"
   - Blob path: vectors/{owner}_{repo}_{branch}/
   - Auth: Search system MSI → Storage Blob Data Reader

5. Clone → embed → save vectors to Blob → push to AI Search index
6. AI Search retrieval → generate wiki → save cache to Blob

7. CREATE/UPDATE AML scheduled pipeline: "deepwiki-{owner}-{repo}-{branch}"
   - Compute: STANDARD_D2_V2
   - Schedule: every 480 hours (configurable in aml_pipeline.json)
   - Command: python -m backend.processor.code_processor --mode=local
   - Identity: workspace MSI

8. Output: Blob vectors/ + wikicache/ + AI Search index populated
```

### 3.7 Cloud-mode: AI Search per-repo index

**Index naming:** `deepwiki-{owner}-{repo}-{branch}` (lowercase, sanitized)
- Example: `deepwiki-msdata-orcasql-myfile-main`
- Schema: loaded from `Deployments/index/code_index_schema.json`

**Data source naming:** `ds-{owner}-{repo}-{branch}`
- Linked to blob path: `vectors/{owner}_{repo}_{branch}/`
- Auth: Search system-assigned MSI → Storage Blob Data Reader (granted in deploy_infra)

**Azure AI Search index limits (Standard S1 tier):**

| Limit | Value | Impact |
|-------|-------|--------|
| Max indexes per service | **50** | = max 50 repo+branch combos |
| Max fields per index | 1000 | Not a concern (schema has ~12 fields) |
| Max document size | 32 MB | Not a concern (chunks are small) |

**Best practice:** Use 1 index per repo+branch. This gives:
- Clean isolation — delete index = delete all data for that repo
- No cross-repo query contamination
- Easy cleanup when a repo is removed
- 50 repos is plenty for initial rollout
- If >50 needed: upgrade to S2 (200 indexes) or consolidate to shared index with filter fields

### 3.8 Cloud-mode: AML Pipeline per-repo

**Pipeline naming:** `deepwiki-{owner}-{repo}-{branch}`

**Compute cluster:** A shared compute cluster `deepwiki-compute` with auto-scaling `max_instances`. Each pipeline gets its own dedicated instance to avoid contention.

**AML configuration (`backend/config/aml_pipeline.json`):**

```json
{
  "compute_name": "deepwiki-compute",
  "compute_size": "STANDARD_D2_V2",
  "compute_min_instances": 0,
  "schedule_interval_hours": 480,
  "environment_name": "deepwiki-processor",
  "environment_base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
}
```

**Compute instance auto-scaling:**

Each repo gets its own scheduled pipeline, and each pipeline needs a dedicated instance so jobs don't queue behind each other. The `cloud_setup.py` auto-manages `max_instances`:

```python
def ensure_compute_capacity(ml_client, compute_name, pipeline_name, config):
    """Ensure compute cluster has enough max_instances for all pipelines."""

    # 1. List all existing deepwiki pipelines by naming convention
    all_jobs = ml_client.schedules.list()
    existing_pipelines = set()
    for schedule in all_jobs:
        if schedule.name.startswith("deepwiki-"):
            existing_pipelines.add(schedule.name)

    # 2. Check if this pipeline is new
    is_new = pipeline_name not in existing_pipelines
    required_instances = len(existing_pipelines) + (1 if is_new else 0)

    # 3. Get or create compute cluster
    try:
        compute = ml_client.compute.get(compute_name)
        current_max = compute.max_instances
    except ResourceNotFoundError:
        # Create new cluster with initial capacity
        compute = AmlCompute(
            name=compute_name,
            size=config["compute_size"],
            min_instances=config.get("compute_min_instances", 0),
            max_instances=required_instances,
            idle_time_before_scale_down=600,  # 10 min idle → scale to 0
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"  ✓ Created compute cluster: {compute_name} (max_instances={required_instances})")
        return

    # 4. Scale up if needed (never scale down automatically)
    if required_instances > current_max:
        compute.max_instances = required_instances
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"  ✓ Scaled compute: {compute_name} max_instances {current_max} → {required_instances}")
    else:
        print(f"  ✓ Compute capacity sufficient: {compute_name} max_instances={current_max} (need {required_instances})")
```

**Scaling behavior:**
- `min_instances = 0` — cluster scales to zero when idle (cost-efficient)
- `max_instances` — auto-incremented when a new repo pipeline is added
- `idle_time_before_scale_down = 600` — nodes deallocate after 10 min idle
- Each pipeline's scheduled job claims 1 node; if all run simultaneously, cluster scales to N nodes
- Never auto-decrements `max_instances` (avoid disruption) — manual cleanup via `teardown_cloud_resources()`

**Pipeline behavior:**
- Uses shared compute cluster `deepwiki-compute`
- Runs `code_processor.py --mode=local` inside AML (blob enabled via cloud config)
- Schedule: recurring every 480 hours (configurable)
- Each repo gets its own named pipeline — visible in AML Studio
- Identity: workspace's user-assigned managed identity

### 3.9 `code_processor.py` — pipeline steps

```python
def run_code_processor(repo_url, branch, language, comprehensive, mode):
    repo_type = "azuredevops"  # Hardcoded

    # Step 1: Clone repo
    repo_path = download_repo(repo_url, type=repo_type, access_token=pat, branch=branch)
    commit_hash = get_head_commit_hash(repo_path)

    # Step 2: Read + split + embed + save vectors
    documents = read_all_documents(repo_path, ...)
    transformed = transform_documents_and_save_as_json(documents, repo_name, branch)

    # Step 3: Build retriever (mode-dependent)
    if mode == "cloud" and is_search_configured():
        search_client.push_documents(transformed, index_name)
        retriever = SearchServiceRetriever(search_client, index_name)
    else:
        retriever = FAISSRetriever(documents=transformed, ...)

    # Step 4: Generate wiki (structure + all pages)
    wiki_data = generate_wiki(repo_url, branch, repo_type, repo_path,
                              retriever, commit_hash, language, comprehensive)

    # Step 5: Save wiki cache
    save_wiki_cache(wiki_data)
```

### 3.10 `wiki_generator.py` — replaces frontend orchestration

```python
def generate_wiki(repo_url, branch, repo_type, repo_path,
                  retriever, commit_hash, language, comprehensive):
    # 1. Build file tree + read README
    file_tree = build_file_tree(repo_path)
    readme = read_readme(repo_path)

    # 2. Generate wiki structure via LLM (direct call, no WebSocket)
    structure_xml = call_llm(build_wiki_structure_prompt(file_tree, readme, ...))
    wiki_structure = parse_structure_xml(structure_xml)  # Uses xml_repair.py for resilience
    # If content filter truncates, retry with directory-only file tree
    # (same logic as ws_handler.py content filter retry)

    # 3. Generate each page via LLM + retrieval
    generated_pages = {}
    for page in wiki_structure.pages:
        context = retriever.call_with_file_filter(page.title, page.file_paths)
        content = call_llm(build_wiki_page_prompt(page, context, commit_hash, ...))
        generated_pages[page.id] = WikiPage(...)
        print(f"  ✓ Page {page.id}: {page.title}")

    return WikiCacheData(wiki_structure=wiki_structure, generated_pages=generated_pages,
                         commit_hash=commit_hash, indexed_at=datetime.utcnow().isoformat())
```

### 3.11 `cloud_setup.py` — per-repo cloud resource orchestration

```python
def setup_cloud_resources(repo_url, branch, config):
    """Create/update AI Search index + data source + AML pipeline for a repo."""
    repo_id = derive_repo_identifier(repo_url, branch)  # "msdata-orcasql-myfile-main"

    # 1. Create AI Search index (idempotent)
    index_name = f"deepwiki-{repo_id}"
    create_search_index(index_name, schema_path="Deployments/index/code_index_schema.json")

    # 2. Create AI Search data source
    ds_name = f"ds-{repo_id}"
    create_search_data_source(ds_name, blob_container, f"vectors/{repo_id.replace('-','_')}/")

    # 3. Create/update AML scheduled pipeline
    pipeline_name = f"deepwiki-{repo_id}"
    create_aml_pipeline(pipeline_name, repo_url, branch,
                        compute_size="STANDARD_D2_V2", schedule_hours=480)

def teardown_cloud_resources(repo_url, branch):
    """Remove all cloud resources for a repo (index + DS + pipeline)."""
```

### 3.12 Configuration

**`infra.json` additions:**

```json
{
  "azure_ai_search": {
    "enabled": false,
    "endpoint": "https://acsorcascodewiki.search.windows.net",
    "api_version": "2024-07-01"
  },
  "azure_ml": {
    "enabled": false,
    "workspace_name": "aml-orcas-codewiki",
    "resource_group": "RG-ORCAS-DEEPWIKI",
    "subscription_id": "4f3f8f41-5643-4664-8c12-ce6b78ceb81f"
  }
}
```

**New `backend/config/aml_pipeline.json`:**

```json
{
  "compute_size": "STANDARD_D2_V2",
  "schedule_interval_hours": 480,
  "environment_name": "deepwiki-processor",
  "environment_base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
}
```

### 3.13 Storage behavior by environment

| Environment | Repo clone | Vectors | Wiki cache | Retrieval | Auth |
|-------------|-----------|---------|------------|-----------|------|
| **Local** | `~/.adalflow/repos/` | `~/.adalflow/vectors/` | `~/.adalflow/wikicache/` | FAISS | PAT |
| **Docker** | Same (volume) | Same (volume) | Same (volume) | FAISS | PAT |
| **Cloud** | Ephemeral | Blob `vectors/` | Blob `wikicache/` | AI Search | MSI |

### 3.14 Logging for processor pipeline

**Current state:** `code_processor.py` uses `logging.basicConfig()` — no file output, no smart dedup.
The existing `setup_logging()` in `backend/tools/logger.py` already supports file logging (`logs/backend-YYMMDD.log`, daily rotation, 30-day retention) + console + optional App Insights.

**Fix (Phase 3A):**
- Replace `logging.basicConfig()` in `code_processor.py` with `setup_logging(log_prefix="processor")`
- Add optional `log_dir` parameter to `setup_logging()` so AML jobs write to `./outputs/` (auto-captured as job artifacts)
- No frontend logger needed in processor path

```python
# In code_processor.py main():
from backend.tools.logger import setup_logging
log_dir = Path("./outputs") if mode == "cloud" else None
setup_logging(log_prefix="processor", log_dir=log_dir)
```

### 3.15 Repo update strategy for scheduled pipeline runs

**Current behavior:** `download_repo(force_update=True)` does `git pull --ff-only`, falling back to `git reset --hard`. Existing vectors are kept unless `force_reprocess=True`.

**Decision: Git pull + full re-embed on every run.**

The processor should always use `force_update=True` (pull latest code) and `force_reprocess=True` (re-embed from scratch):

```python
# In code_processor.py:
download_repo(..., force_update=True)       # Always pull latest
request_rag.prepare_retriever(..., force_reprocess=True)  # Always re-embed
```

**Rationale:**
- Re-embedding is fast (~3 min for 1000 chunks) — runs ~18 times/year per repo on 480-hour schedule
- Full rebuild guarantees no stale chunks from deleted/renamed files
- Incremental diffing adds significant complexity (chunk-to-file mapping, deleted file handling) for minimal gain
- `git reset --hard` handles force-pushes and rebases safely

**Future optimization (deferred):** If repos >5000 files become a bottleneck, implement incremental embedding via `git diff {old_hash}..{new_hash}` — requires storing `commit_hash` in vectors directory metadata.

### 3.16 Vector storage memory optimization (deferred)

**Current behavior:** `transform_documents_and_save_as_json()` embeds ALL chunks in memory via `LocalDB`, then saves all to disk at once. Peak memory for 1000 chunks ≈ 25MB (vectors + text × 2).

**Assessment:** Acceptable for repos under ~5000 files (~85MB peak). No changes needed now.

**Future optimization:** Add batch-embed-then-flush pattern (200 chunks per batch) to `code_processor.py` for repos >5000 files. This would reduce peak memory from O(N) to O(batch_size) without modifying `LocalDB` internals.

---

## 3. Phase 3.5: Backend Cleanup — Directory Reorganization

**Goal:** Fix misplaced files, rename misleading packages, and clarify module boundaries — without changing any public API behavior.
**Depends on:** Phase 3A complete (processor files exist and import from current locations).
**Principle:** Every file should live in the package that owns its domain. Move files, update imports, verify with `flake8` + `pytest`.

### Tier 1 — Safe Moves (Do Now, Phase 3A Companion)

These are low-risk relocations with clear ownership. Each move updates all imports project-wide.

#### 3.1 Move `utils/xml_repair.py` → `modules/wiki/xml_repair.py`

**Why:** `xml_repair.py` contains wiki-domain logic (`repair_truncated_xml`, `_xml_unescape`). It's only consumed by `modules/wiki/` and `processor/wiki_generator.py`. It has no general-utility purpose.

| Before | After |
|--------|-------|
| `backend/utils/xml_repair.py` | `backend/modules/wiki/xml_repair.py` |
| `from backend.utils.xml_repair import ...` | `from backend.modules.wiki.xml_repair import ...` |

**Files to update imports:**
- `backend/modules/wiki/cache.py`
- `backend/modules/chat/ws_handler.py`
- `backend/processor/wiki_generator.py`
- `backend/modules/wiki/__init__.py` (re-export)

#### 3.2 Move `tools/embedder.py` → `clients/embedder.py`

**Why:** `embedder.py` is a client factory — it creates an `Embedder` instance backed by `AzureAIClient`. It belongs alongside `azureai_client.py`, `storage.py`, `blob_client.py` in `clients/`.

| Before | After |
|--------|-------|
| `backend/tools/embedder.py` | `backend/clients/embedder.py` |
| `from backend.tools.embedder import ...` | `from backend.clients.embedder import ...` |

**Files to update imports:**
- `backend/modules/rag/retriever.py`
- `backend/modules/rag/document.py`
- `backend/processor/code_processor.py` (if it imports embedder)
- `backend/clients/__init__.py` (re-export)

#### 3.3 Rename `tools/` → `infra/` (or flatten)

**Why:** After moving `embedder.py` out, `tools/` contains only `logger.py`. The name "tools" implies LLM tool-calling or CLI utilities, neither of which applies. Options:
- **Option A:** Rename `tools/` → `infra/` (holds `logger.py`, future infra utils)
- **Option B:** Move `logger.py` → `utils/logger.py` and delete `tools/` entirely

**Recommended:** Option A (`infra/`). Keeps a dedicated place for cross-cutting infrastructure.

| Before | After |
|--------|-------|
| `backend/tools/logger.py` | `backend/infra/logger.py` |
| `from backend.tools.logger import ...` | `from backend.infra.logger import ...` |

**Files to update imports:** All files that call `setup_logging()` (approximately 5-6 files).

#### 3.4 Clean up `utils/__init__.py`

After moving `xml_repair.py` out, `utils/` should contain only `paths.py`. Verify `__init__.py` re-exports are updated. If `utils/` has only `paths.py`, consider whether to keep the package or move `paths.py` elsewhere — but **defer** this decision (it's fine as-is).

### Tier 2 — Deferred (Post-Phase 4)

These changes are higher-risk, touch more code, or need design discussion first.

#### 3.5 Separate prompt templates from prompt builders in `promptstore/`

**Current state:** `promptstore/wiki_structure.py` contains both raw prompt templates (`WIKI_STRUCTURE_TEMPLATE`) and business logic (`build_wiki_structure_prompt()`, `file_tree_dirs_only()`, `LANGUAGE_DISPLAY_NAMES`).

**Desired state:** `promptstore/` is a pure template store. Builder functions move to the module that owns the domain:
- `build_wiki_structure_prompt()` → `modules/wiki/prompts.py`
- `file_tree_dirs_only()` → `modules/wiki/utils.py` or `utils/paths.py`
- `LANGUAGE_DISPLAY_NAMES` → `config/lang.json` (already partially there)

**Why deferred:** The builder function is shared between `ws_handler.py` and `wiki_generator.py`. Moving it requires careful import coordination and testing both WebSocket and processor paths.

#### 3.6 Refactor `ws_handler.py` god function

**Current state:** `handle_websocket_chat()` is ~680 lines handling connection lifecycle, wiki structure generation, page generation, RAG retrieval, and error recovery — all in one function.

**Desired state:** Extract into focused functions:
- `_generate_wiki_structure()` — structure XML generation + parsing
- `_generate_wiki_page()` — single page generation with RAG
- `_handle_wiki_generation()` — orchestration loop
- Keep `handle_websocket_chat()` as thin connection manager

**Why deferred:** This is the most critical real-time code path. Requires thorough WebSocket testing after refactor.

#### 3.7 Centralize storage mode branching

**Current state:** Multiple files check `storage.get_mode() == "blob"` and branch on local-vs-blob logic (e.g., `cache.py`, `retriever.py`).

**Desired state:** The `storage` singleton and `VectorStorage` abstraction already handle most branching. Audit remaining raw branching and push it behind the abstraction layer.

### Execution Checklist

For each Tier 1 move:
1. Create target file (copy content)
2. Update all imports project-wide (`grep -r "from backend.tools.embedder"`)
3. Update `__init__.py` re-exports in both old and new packages
4. Run `flake8 backend/` — must pass
5. Run `pytest tests/` — must pass
6. Delete old file
7. Commit with descriptive message

---

## 4. Phase 4: Frontend Read-Only Mode

**Goal:** Strip all generation UI from the frontend. Wiki page becomes a pure reader of cached data.
**Depends on:** Phase 3 (need the processor to generate content).

### 4.1 Home page changes (`src/app/page.tsx`)

| Action | Target | Lines |
|--------|--------|-------|
| **HIDE** | Repo URL input form + "Generate Wiki" submit button | L493-L519 |
| **HIDE** | `ConfigurationModal` (branch, language, token, wiki type) | L525-L562 |
| **KEEP** | `ProcessedProjects` grid — browse entry point for existing wikis | L577-L597 |
| **KEEP** | Navigation to `/{owner}/{repo}` for existing cached wikis | — |

### 4.2 Wiki page changes (`src/app/[owner]/[repo]/page.tsx`)

| Action | Target | Lines | Notes |
|--------|--------|-------|-------|
| **REMOVE** | `fetchRepositoryStructure()` | L1876-L2277 | No repo API calls needed |
| **REMOVE** | `determineWikiStructure()` | L1020-L1870 | No WebSocket structure generation |
| **REMOVE** | `generatePageContent()` | L694-L993 | No WebSocket page generation |
| **REMOVE** | `saveCheckpoint()` + completion save | L637-L690, L2862-L2940 | No cache writes from frontend |
| **REMOVE** | Loading overlay (pulsing dots, progress bar, page list) | L2963-L3068 | No generation progress |
| **REMOVE** | "Refresh Wiki" button | L3175-L3186 | Refresh = re-run processor |
| **REMOVE** | `ModelSelectionModal` | L3404-L3437 | No model selection in viewer |
| **SIMPLIFY** | `loadData` effect | L2541-L2856 | Cache check only; if not found → "Wiki not available" |
| **KEEP** | `WikiTreeView` sidebar navigation | L3114-L3379 | Core viewer functionality |
| **KEEP** | `Markdown` content rendering | L3245-L3310 | Core viewer functionality |
| **KEEP** | Ask chat panel (optional) | L3328-L3379 | Q&A uses retrieval, not generation |
| **ADD** | "Wiki not available" message | — | Shown when no cache exists |
| **ADD** | "Last indexed" display | — | Shows `commit_hash` + `indexed_at` from cache |

### 4.3 Components to simplify/remove

| Component | File | Action |
|-----------|------|--------|
| `WikiGenerationContext` | `src/contexts/WikiGenerationContext.tsx` | **REMOVE** — no generation tracking needed |
| `ConfigurationModal` | `src/components/ConfigurationModal.tsx` | **REMOVE** — no wiki config from UI |
| `ModelSelectionModal` | `src/components/ModelSelectionModal.tsx` | **REMOVE** — no model selection |
| `ProcessedProjects` | `src/components/ProcessedProjects.tsx` | **KEEP** — browse existing wikis |
| `WikiTreeView` | `src/components/WikiTreeView.tsx` | **KEEP** — sidebar navigation |
| `Ask` | `src/components/Ask.tsx` | **KEEP** — chat Q&A (optional) |
| `Markdown` | `src/components/Markdown.tsx` | **KEEP** — page rendering |

### 4.4 Environment-aware reading

The existing `GET /api/wiki_cache` route already handles both environments:
- **Local / Docker**: Reads from `~/.adalflow/wikicache/` via `read_wiki_cache()`
- **Azure cloud**: Reads from Blob via `blob_client` (when `azure_blob_storage.enabled=true`)

No API changes needed — the storage abstraction handles mode detection.

### 4.5 State variables to remove from `page.tsx`

```tsx
// These become unnecessary:
const [isGenerationStarted, setIsGenerationStarted] = useState(false);
const [pagesInProgress, setPagesInProgress] = useState<Set<string>>(new Set());
// The entire WebSocket generation machinery (~500 lines)
```

### 4.6 Simplified `loadData` flow

```tsx
// Current: cache check → fetch repo structure → generate structure → generate pages → save
// New:     cache check → render wiki OR show "not available"

useEffect(() => {
  const loadData = async () => {
    const cached = await fetch(`/api/wiki_cache?owner=${owner}&repo=${repo}&...`);
    if (cached.ok) {
      const data = await cached.json();
      setWikiStructure(data.wiki_structure);
      setGeneratedPages(data.generated_pages);
      setCommitHash(data.commit_hash);
      setIndexedAt(data.indexed_at);
    } else {
      setError("Wiki not available. Run code_processor to generate.");
    }
    setIsLoading(false);
  };
  loadData();
}, [owner, repo]);
```

---

## 5. Phase 5: UX Polish & Chat Integration

### 5.1 Chat Q&A in read-only mode

The `Ask` chat panel is kept after Phase 4, but retrieval must work without WebSocket generation.

**Local/Docker mode:** FastAPI backend still runs. Chat WebSocket handler uses FAISS retrieval from existing vectors on disk. No changes needed — `ws_handler.py` already loads vectors from `~/.adalflow/vectors/`.

**Cloud mode:** Chat should query AI Search instead of FAISS. Update `ws_handler.py` to auto-select retriever:
```python
if is_search_configured():
    # Use AI Search for retrieval (cloud)
    retriever = SearchServiceRetriever(search_client, index_name_for_repo)
else:
    # Fall back to FAISS (local/docker)
    retriever = load_faiss_from_vectors(repo_name, branch)
```

### 5.2 "Last indexed" display

Show `commit_hash` + `indexed_at` from cache in sidebar header:
```
Last indexed: 15 Mar 2026 (dd4edb9)
```

### 5.3 Citation URLs with commit hash

`citationProcessor.tsx` uses `commit_hash` from cache instead of `HEAD` for all source links.

### 5.4 Wiki search

Client-side full-text search across `generated_pages`.

### 5.5 Export wiki

Review existing export functionality after Phase 4 cleanup. Ensure it works in read-only mode.

### 5.6 Fluent 2 design system

Color tokens, typography (Segoe UI Variable), component patterns.

### 5.7 Share button

Copy permalink URL to clipboard.

---

## 6. File Change Map

### Phases 0–2 — Complete ✅

See §0 for full item list.

### Phase 3A (Core Processor — local mode)

| File | Change | Description |
|------|--------|-------------|
| `backend/processor/__init__.py` | CREATE | Package init, exports `run_code_processor()` ✅ |
| `backend/processor/code_processor.py` | CREATE | CLI entry point with `--repo`, `--branch`, `--mode`, `--config` ✅ |
| `backend/processor/wiki_generator.py` | CREATE | Server-side wiki orchestration (~200 lines) ✅ |
| `backend/sample.run.json` | CREATE | Template config file for CLI ✅ |
| `backend/sample.env` | MODIFY | Added `REPO_ACCESS_TOKEN` key ✅ |
| `backend/.env` | MOVE | Moved from root to `backend/` ✅ |
| `backend/config/infra.json` | MODIFY | Added `azure_ai_search` and `azure_ml` sections ✅ |
| `backend/types/config_types.py` | MODIFY | Added `AzureAISearchConfig`, `AzureMLConfig` models ✅ |
| `backend/config.py` | MODIFY | Added `get_search_config()`, `is_search_configured()`, `get_aml_config()` ✅ |
| `backend/main.py` | MODIFY | `load_dotenv()` path → `backend/.env` ✅ |
| `docker-compose.yml` | MODIFY | `env_file` → `backend/.env` ✅ |
| `test-local.ps1` | MODIFY | `.env` path → `backend/.env` ✅ |
| `Dockerfile` | MODIFY | `touch .env` / `source .env` → `backend/.env` ✅ |
| `.gitignore` | MODIFY | Added `backend/run.json`, `!backend/sample.env` ✅ |
| `backend/infra/logger.py` | MODIFY | Add `log_dir` parameter to `setup_logging()` ✅ |
| `backend/processor/code_processor.py` | MODIFY | Use `setup_logging()`, `force_reprocess=True` ✅ |

### Phase 3B (Docker mode)

No new files — `--mode=docker` builds existing Dockerfile and runs `--mode=local` inside container.

### Phase 3C (Cloud infrastructure: AI Search + AML)

| File | Change | Description |
|------|--------|-------------|
| `backend/clients/search_client.py` | CREATE | AI Search: create index, push documents, query |
| `backend/processor/cloud_setup.py` | CREATE | Per-repo: AI Search index/DS + AML pipeline creation |
| `backend/config/aml_pipeline.json` | CREATE | AML compute size, schedule interval, environment config |
| `backend/config/aml_conda.yml` | CREATE | Conda environment for AML (Python deps from pyproject.toml) |
| `backend/config/infra.json` | MODIFY | Add `azure_ai_search` and `azure_ml` sections |
| `backend/config.py` | MODIFY | Add `get_search_config()`, `get_aml_config()` readers |

### Phase 3D (Cloud mode integration)

No new files — wires 3A + 3C together in `code_processor.py`'s `--mode=cloud` path.

### Phase 3.5 (Backend Cleanup — Tier 1)

| File | Change | Description |
|------|--------|-------------|
| `backend/utils/xml_repair.py` | MOVE → `backend/modules/wiki/xml_repair.py` | Wiki-domain logic, not general utility ✅ |
| `backend/tools/embedder.py` | MOVE → `backend/clients/embedder.py` | Client factory, belongs with other clients ✅ |
| `backend/tools/` | RENAME → `backend/infra/` | Only `logger.py` remains; "tools" name is misleading ✅ |
| `backend/modules/wiki/__init__.py` | MODIFY | Re-export `xml_repair` functions ✅ |
| `backend/clients/__init__.py` | MODIFY | Note: no eager re-export (circular import) ✅ |
| ~6 files | MODIFY | Update imports for moved files ✅ |

### Phase 4 (Frontend Read-Only)

| File | Change | Description |
|------|--------|-------------|
| `src/app/page.tsx` | MODIFY | Hide generation form + ConfigurationModal; keep ProcessedProjects |
| `src/app/[owner]/[repo]/page.tsx` | MODIFY | Remove ~1200 lines of generation logic; cache-read-only |
| `src/contexts/WikiGenerationContext.tsx` | DELETE | No generation tracking |
| `src/components/ConfigurationModal.tsx` | DELETE | No wiki configuration from UI |
| `src/components/ModelSelectionModal.tsx` | DELETE | No model selection in viewer |

---

## 7. Verification Plan

### Phase 3A

| Test | Method |
|------|--------|
| Local CLI generates wiki | `python -m backend.processor.code_processor --config=backend/run.json` → produces wiki cache ✅ |
| Vectors saved correctly | `~/.adalflow/vectors/{repo}_{branch}/` contains per-chunk JSON files ✅ |
| Wiki cache has all pages | Cache JSON `generated_pages` keys match `wiki_structure.pages` IDs |
| Commit hash in cache | `commit_hash` matches `git rev-parse HEAD` |
| Auth: PAT from .env | Set `REPO_ACCESS_TOKEN` in `backend/.env` → clone succeeds ✅ |
| Auth: no PAT, no az login | Exits with clear error message ✅ |
| Auth: Docker without PAT | Exits with `ERROR: Docker mode requires REPO_ACCESS_TOKEN` ✅ |
| Logging to file | `logs/processor-YYMMDD.log` created with pipeline output |
| Re-run re-embeds | Second run with same repo → vectors regenerated fresh |

### Phase 3B

| Test | Method |
|------|--------|
| Docker generates wiki | `--mode=docker` → same output in volume mount |

### Phase 3C + 3D

| Test | Method |
|------|--------|
| AI Search index created | `deepwiki-{owner}-{repo}-{branch}` index exists in Search service |
| Data source created | `ds-{owner}-{repo}-{branch}` data source exists |
| Vectors indexed | Search query returns results for indexed repo |
| AML pipeline created | `deepwiki-{owner}-{repo}-{branch}` pipeline visible in AML Studio |
| Pipeline schedule set | Schedule interval = 480 hours |
| First run completes | AML job transitions to Completed |
| Output in Blob | `wikicache/` and `vectors/` populated in Blob |

### Phase 4

| Test | Method |
|------|--------|
| Home page shows only cached projects | No "Generate Wiki" form visible |
| Wiki page loads from cache | Navigate to `/{owner}/{repo}` → loads cached wiki immediately |
| "Wiki not available" message | Navigate to non-existent wiki → shows message, no errors |
| No WebSocket generation calls | Network tab shows zero WS connections for wiki gen |

### Regression

```powershell
flake8 backend/
npm run build
pytest tests/
```

---

## Appendix A: Cognition DeepWiki Reference

**Sidebar** (`deepwiki.com/fastapi/fastapi`, March 12, 2026):

```
1  FastAPI Overview
2  Core Framework Architecture
   2.1  Application and Routing System
   2.2  Dependency Injection System
   2.3  Request Processing Pipeline
   2.4  Response Handling and Serialization
   2.5  OpenAPI Schema Generation
   2.6  Pydantic Integration
3  Advanced Features
   3.1-3.8  (Security, Settings, Error Handling, DB, WebSocket, Hooks, Streaming, Content-Type)
4  Documentation System
   4.1-4.6  (Build, Multi-lang, Translation, UI, Deployment, CLI)
5  Testing and Quality Assurance
   5.1  Test Framework and Coverage
   5.2  Code Quality and Pre-commit Hooks
6  Project Infrastructure
   6.1-6.4  (Package Config, uv, CI/CD, Dev Workflow)
7  Deployment and Distribution
   7.1  Installation and CLI
   7.2  FastAPI Cloud Deployment
8  Release Management
   8.1-8.2  (Version History, Breaking Changes)
9  Community and Ecosystem
   9.1-9.5  (Recognition, Sponsorship, Resources, Automation, Contributing)
```

**Citation format:**
```markdown
Sources: [README.md L139-L143](https://github.com/fastapi/fastapi/blob/11614be9/README.md#L139-L143)
```

**Cross-page format:**
```markdown
For details, see [OpenAPI Schema Generation](deepwiki://2.5).
```

**Metadata:** `Last indexed: 11 March 2026 (11614b)`

---

## Appendix B: Target Backend Module Map

```
backend/
├── app.py                    # FastAPI routers + endpoints
├── main.py                   # Uvicorn entry point
├── config.py                 # Config readers (infra.json, aml_pipeline.json, etc.)
├── processor/                # NEW — standalone code processor
│   ├── __init__.py
│   ├── code_processor.py     # CLI entry point (local/docker/cloud)
│   ├── wiki_generator.py     # Structure → pages → cache
│   └── cloud_setup.py        # Per-repo: AI Search index/DS + AML pipeline
├── clients/
│   ├── azureai_client.py     # Azure OpenAI (LLM + embedding)
│   ├── blob_client.py        # Azure Blob storage
│   ├── storage.py            # Unified storage abstraction
│   ├── vector_storage.py     # JSON vector storage (per-chunk files)
│   └── search_client.py      # NEW — Azure AI Search (index + query + push)
├── modules/
│   ├── chat/
│   │   ├── ws_handler.py     # WebSocket handler (chat Q&A only after refactor)
│   │   ├── service.py        # Context formatting, conversation history
│   │   └── models.py         # ChatCompletionRequest
│   ├── wiki/
│   │   ├── cache.py          # Wiki cache save/load
│   │   ├── routes.py         # /api/wiki_cache endpoints
│   │   └── models.py         # WikiCacheData, WikiStructure, WikiPage
│   ├── rag/
│   │   ├── retriever.py      # RAG + FAISS/Search retriever
│   │   ├── database.py       # DatabaseManager (clone + embed lifecycle)
│   │   ├── document.py       # File reading + embedding pipeline
│   │   └── code_splitter.py  # Code-aware chunking
│   └── repository/
│       ├── git_ops.py        # Clone, pull, branch detection
│       └── file_content.py   # Single file content retrieval
├── promptstore/
│   ├── wiki_structure.py     # Structure generation prompts
│   ├── wiki_page.py          # Page generation prompts
│   └── chat_system.py        # Chat system prompts
├── config/
│   ├── infra.json            # Azure resource config
│   ├── embedder.json         # Embedding model config
│   ├── generator.json        # LLM model config
│   └── repo.json             # File filter defaults
└── utils/
    ├── paths.py              # ~/.adalflow/ path helpers
    ├── source_url.py         # Commit-pinned permalink builder
    └── xml_repair.py         # XML parsing utilities
```

