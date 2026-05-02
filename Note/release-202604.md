# DeepWiki April Release Notes

**Date:** 2026-05-02
**Branch:** `dev/xixia/v2`
**Period covered:** 2026-03-25 → 2026-05-02 (since [release-202603.md](./release-202603.md))

---

## Highlights

- **CodeMap** — new static AST-based symbol/dependency graph with interactive D3 visualization, tree-sitter parsing for 8+ languages, importance scoring, and incremental LOD layout
- **CodeTrace** — new AI-powered query-driven code-flow tracing module (RAG + LLM, 3-panel UI)
- **Task-specific Azure OpenAI routing** — `chat` / `reasoning` / `embedding` deployments split with per-task endpoints, API versions, and temperature
- **Health-check endpoints** — `/health/openai` and frontend prefetch with retry/cache for Ask UI gating
- **Content-filter sanitization** — Azure OpenAI 400 / RAI redaction at chat + wiki-gen layer; shared `backend/utils/sanitizer.py`
- **Wiki-gen review pass** — second-pass LLM call validates accuracy and fixes Mermaid diagrams
- **Repo cloning hardened** — branch-safe paths, atomic file writes, UMI fallback for cloud token acquisition, token sanitization in git ops
- **Filter refactor** — `repo.json` split into `excluded.json` + `included.json`; new `backend/utils/filter.py`
- **Memory optimization, round 2** — codemap memory fix plan, branch-name sanitization, vector-storage caching, tokenizer reuse
- **Frontend polish** — Orcas branding, AI disclaimer, dark-mode fixes, Mermaid clipping, responsive footer/project list, textarea autosize for Ask
- **AML role assignments** — Azure AI Enterprise Network Connection Approver added to ARM template

---

## New Features

### CodeMap — Static Symbol & Dependency Graph

End-to-end module shipped over 7 commits (`72e4e4b` → `4ace489`).

- **Backend** ([backend/modules/codemap/](../backend/modules/codemap/)) — `analyzer.py` (1,255 lines, tree-sitter AST traversal), `graph_builder.py` (importance scoring + LOD), `cache.py` (blob/local persistence), `models.py`, `routes.py`. Languages: Python, JavaScript, TypeScript, Java, Go, C#, C, C++.
- **Pipeline integration** — `processor/codemap_generator.py` runs as **Stage 2** (after clone, before embed); summary injected into wiki-structure prompt; edges expand `file_paths` for wiki-page retrieval.
- **API endpoint** — `GET /api/codemap` + frontend proxy at [src/app/api/codemap_cache/route.ts](../src/app/api/codemap_cache/route.ts).
- **Frontend** ([src/components/CodeMap.tsx](../src/components/CodeMap.tsx), 503+ lines) — interactive D3 force-layout, expand/collapse, click-to-highlight, search/filter, container edges for child containment, separated layout for connected vs disconnected components.
- **Importance scoring** — degree centrality + boost for symbol type; drives LOD (level-of-detail) culling at low zoom.
- **Incremental layout** — when LOD changes, existing visible nodes keep their positions; new nodes anchor to neighbors with overlap avoidance; "hero node fitting" frames the most important component on first paint.
- **Storage** — `~/.adalflow/codemap/` (local) or blob `codemap/{repo}/`; separate from wiki cache.
- **Tree-sitter grammars** added to `poetry.lock` (`7770d10`).

### CodeTrace — AI Code-Flow Tracing

New module ([backend/modules/codetrace/](../backend/modules/codetrace/)) for query-driven code traces (`299a297`, `2839fdc`).

- **Service flow** — RAG retrieval → LLM (reasoning deployment) → XML-parsed structured response → source-content extraction (`SourceChunk` model).
- **Endpoint** — `POST /api/codetrace` (FastAPI router); frontend proxy at [src/app/api/codetrace/route.ts](../src/app/api/codetrace/route.ts).
- **3-panel UI** — [src/app/\[owner\]/\[repo\]/codetrace/page.tsx](../src/app/[owner]/[repo]/codetrace/page.tsx): trace sections (left), source viewer with line highlighting (right), chat bar (bottom).
- **Mode dropdown** — replaces the old Deep Research toggle in [Ask.tsx](../src/components/Ask.tsx); 3-way switch (Chat / Deep Research / Code Trace).
- **Prompt** — [backend/promptstore/code_trace.py](../backend/promptstore/code_trace.py) (system + user templates).
- **No vector-schema changes** — text search on existing `content` field works for cloud mode; future Phase 4 in module README discusses dedicated index fields.

### Wiki-Gen Review Pass

`processor/wiki_generator.py` now runs a second LLM call after the initial page generation to verify accuracy and fix Mermaid diagram syntax (`e0ba830`). Toggleable; cloud and local both supported.

### Health-Check & Caching for Ask UI

- New `GET /health` and `GET /health/openai` endpoints (`5d15a53`).
- Frontend prefetches `/api/health/openai` at page mount; Ask button is gated on connectivity (`Ask.tsx`).
- `vector_storage.py` gains an in-memory cache for repeated retriever rebuilds; nginx config tuned for the new endpoint (`nginx.conf`).

### Content-Filter Sanitization

New shared utility [backend/utils/sanitizer.py](../backend/utils/sanitizer.py) (`c4fb4cc`):

- `sanitize_for_content_filter()` — single source of truth used by both chat (`modules/chat/service.py`) and wiki gen (`processor/wiki_generator.py`).
- Azure OpenAI client emits **content-filter diagnostics** when a response is blocked (logs filter category + severity).
- `chat/service.py` no longer keeps a private `_sanitize_for_content_filter` — moved to the shared utility.

### Task-Specific Azure OpenAI Configuration

`infra.json` and `config.py` refactored (`11bc17e`):

- Three sub-blocks: `azure_openai.chat`, `azure_openai.reasoning`, `azure_openai.embedding`.
- Each has its own `endpoint`, `api_version`, `deployment`, plus task-specific knobs (`temperature` for chat/reasoning, `dimensions` for embedding).
- `get_azure_openai_config(task='chat'|'reasoning'|'embedding')` is the single accessor; per-task wiring in `app.py`, `chat/http_handler.py`, `chat/ws_handler.py`, `processor/wiki_generator.py`.
- Auth matrix + model routing fully documented in root [README.md](../README.md) (`b94d784`).

### Filter Refactor (repo.json → excluded.json + included.json)

`9fcc7b0` split `backend/config/repo.json` into:

- `backend/config/excluded.json` — excluded directories + files (normalized, broken patterns removed).
- `backend/config/included.json` — supported file extensions (code + doc) for chunking.

New utility [backend/utils/filter.py](../backend/utils/filter.py) centralizes all path/branch sanitization. Old `repo.json` removed; `embedder/document.py`, `processor/wiki_generator.py`, and tests rewired.

### Frontend Design Document

`src/components/DESIGN.md` (`8a07ff2`, 524 lines) — comprehensive frontend architecture reference (component map, Mermaid rendering pipeline, state-management notes, theming).

---

## Bug Fixes

- **adalflow `ToEmbeddings` IndexError** — `data_components.py` line 96 mis-uses `batch_idx * batch_size + idx` where `batch_idx` is a start index. Bypassed by calling the embedder directly with explicit sub-batching (carry-over from Phase 3a; reinforced this cycle).
- **Mermaid diagram clipping** (`bd30167`, `0c749a4`) — fixed text padding + `mermaid.initialize` type assertion.
- **Dark-mode body overflow** (`516552b`) — page no longer overflows the viewport in dark theme; mermaid initialization adjusted.
- **Section/page IDs in DOM** (`3635d2d`) — wiki sections/pages render with stable `id` attributes for anchor links.
- **Ask input UX** (`33d4bd1`, `ab6a2c8`) — `<input>` swapped for autoresizing `<textarea>`; send button + Enter behavior gated on WebSocket connection state.
- **Footer responsiveness** (`a6ecfe2`, `44fb3a2`) — visibility logic + project-list layout fixed on small viewports.
- **AI Search document-key validation** (`335b52e`) — sanitize keys to satisfy AI Search regex (`[A-Za-z0-9_\-=]+`).
- **AI Search adaptive batch size** (`84c0175`) — auto-shrink upload batch on `RequestEntityTooLarge`; prevents 413 errors for large chunks.
- **Excluded-dirs config drift** (`a08810f`) — added `.next`, `node_modules` variants, build-output globs that were leaking into ingestion.
- **Codemap file-filter simplification** (`2407fc2`) — single shared filter path between embedder and codemap.
- **Tokens leaking in git error logs** (`e0ba830`) — `repository/git_ops.py` redacts PAT/UMI tokens before logging stderr.
- **Repository cache races** (`bf3ed87`) — branch-safe local paths (`{repo}_{branch}/`), atomic writes via temp + `os.replace()`.
- **MSAL token-cache log spam** (`b0a2cae`) — `MSAL.TokenCache` and `azure.identity._internal.decorators` loggers raised to WARNING.
- **Verbose AzureAI request logging** (`4dffc23`, `e1c92cf`) — request ID propagated through error handler; log format unified.
- **NPM vulnerabilities** (`9055ea9`) — `npm audit fix` for transitive deps.

---

## Infrastructure & Auth Changes

- **UMI token fallback** (`8da8af6`) — `step_clone()` retries with fresh `DefaultAzureCredential.get_token()` if cached token expired during long AML runs.
- **Search client auth** (`6c495a3`) — UMI-aware credential with key fallback for local; explicit endpoint validation.
- **Branch name sanitization** (`3395c04`) — `sanitize_branch_for_path()` in `utils/filter.py`; applied to blob path, AI Search index name, codemap cache key, vector-storage path. Unicode and `/` replaced consistently.
- **Storage refactor** (`133e0e5`) — removed deprecated pickle code paths from `clients/storage.py` and `clients/blob_client.py`; embedder no longer exports `__init__.py` shims.
- **Azure ML role assignments** (`e8db1ef`) — ARM template now grants `Azure AI Enterprise Network Connection Approver` to the AML workspace MSI for AI Search private-endpoint approval.
- **Compute SKU** — verified `STANDARD_D11_V2` retained from prior cycle (was tuned for the 14 GB envelope).
- **orjson** (`c9fecf9`) — added for ~3x faster JSON serialization on hot wiki-cache + vector-storage paths.

---

## Memory Optimization (CodeMap + Round 2)

Identified codemap-specific hotspots and applied targeted fixes (`79fb290`, doc'd in [backend/modules/codemap/MEMORY_FIX_PLAN.md](../backend/modules/codemap/MEMORY_FIX_PLAN.md)):

| Change | File(s) | Effect |
|--------|---------|--------|
| Streamed graph build (chunk symbol-table per file) | `codemap/graph_builder.py` | Prevents holding all ASTs simultaneously |
| Cache-write streaming (don't materialize full JSON before write) | `codemap/cache.py` | Lower transient peak during save |
| Tokenizer instance reuse (lazy module-level singleton) | `embedder/tokenizer.py` | Avoids re-loading tiktoken bpe per chunk |
| Vector-storage chunked load + LRU cache | `clients/vector_storage.py` | Repeated retriever rebuilds skip blob round-trips |
| Codemap cache invalidation by commit hash | `codemap/cache.py` | Stale graphs rebuilt on push |

These complement the embedder Phase 1–5 work shipped in March; codemap is now stable on the 14 GB AML compute for 47K-file repos.

---

## Documentation

- **Module READMEs (this cycle)** — Across the cleanup pass on 2026-05-02:
  - [backend/README.md](../backend/README.md) — added `codemap/`+`codetrace/` to module tree, renamed `clients/embedder.py` → `embedding_client.py`, expanded promptstore (added `codemap.py`+`code_trace.py`), expanded utils (added `filter.py`+`sanitizer.py`), endpoint count 7→10, refreshed Module Dependency Graph.
  - [backend/processor/README.md](../backend/processor/README.md) — removed broken `DESIGN.md` references, renumbered pipeline (cloud-mode `Push to AI Search` is now Stage 3.5).
  - [backend/modules/chat/README.md](../backend/modules/chat/README.md) — corrected Deep Research prompt names, real `build_system_prompt` signature, real RAG context format, sanitizer pointer.
  - [backend/modules/repository/README.md](../backend/modules/repository/README.md) — `get_head_commit_hash` import path corrected.
  - [backend/modules/codemap/README.md](../backend/modules/codemap/README.md) — broken `../codetrace/PLAN.md` link → `README.md`.
  - [backend/modules/codetrace/README.md](../backend/modules/codetrace/README.md) — removed `PLAN.md` row + references; replaced with anchor links to inline Phase 4 roadmap.
  - Root [README.md](../README.md) — fixed broken `Orcas CodeWiki-Open` URL with space; expanded V2 Project Structure to include `codemap/`, `codetrace/`, `utils/`, `main.py`; extended `infra.json` field table with `azure_ai_search.*` + `azure_ml.*` rows; expanded "Other Configuration Files" list (`excluded.json`, `included.json`, `lang.json`, `.cloud/`); replaced V1 "VMs, Container Apps" auth note with V2 "Azure Web App / AML Compute".
- **Frontend design doc** — [src/components/DESIGN.md](../src/components/DESIGN.md) added (524 lines).
- **CodeMap memory plan** — [backend/modules/codemap/MEMORY_FIX_PLAN.md](../backend/modules/codemap/MEMORY_FIX_PLAN.md) (468 lines).

---

## Files Changed (Highlights)

| File | Change |
|------|--------|
| `backend/modules/codemap/*` | NEW — analyzer, graph_builder, cache, routes, models (~3,200 lines) |
| `backend/modules/codetrace/*` | NEW — service, routes, models (~950 lines + frontend page) |
| `backend/promptstore/codemap.py` | NEW — codemap summary section for wiki-structure prompt |
| `backend/promptstore/code_trace.py` | NEW — CodeTrace system + user prompts |
| `backend/processor/codemap_generator.py` | NEW — pipeline Stage 2 |
| `backend/utils/sanitizer.py` | NEW — content-filter redaction |
| `backend/utils/filter.py` | NEW — branch + path sanitization |
| `backend/config/excluded.json` | NEW (split from `repo.json`) |
| `backend/config/included.json` | NEW (split from `repo.json`) |
| `backend/config/repo.json` | REMOVED |
| `backend/clients/azureai_client.py` | Task-specific config + content-filter diagnostics + request-id propagation |
| `backend/clients/embedding_client.py` | Renamed from `embedder.py`; per-task config |
| `backend/clients/search_client.py` | Adaptive batch sizing, doc-key sanitization, UMI auth, blob-path alignment |
| `backend/clients/vector_storage.py` | Chunked load, LRU cache, MSI fallback |
| `backend/clients/blob_client.py` | Config caching, atomic writes |
| `backend/config.py` | Per-task config helpers; cache-clear on dir switch (carry-over) |
| `backend/config/infra.json` | Three sub-blocks for chat/reasoning/embedding |
| `backend/processor/code_processor.py` | Cloud mode, UMI fallback, codemap integration, branch-safe paths |
| `backend/processor/wiki_generator.py` | Review pass, content-filter handling, file-tree truncation |
| `backend/modules/embedder/document.py` | Filter refactor, Phase-4/5 carry-over |
| `backend/modules/embedder/retriever.py` | Health-check support, single-pass validation, vector caching |
| `backend/modules/embedder/code_splitter.py` | Improved chunk-meta handling |
| `backend/modules/embedder/tokenizer.py` | NEW — singleton tokenizer |
| `backend/modules/repository/git_ops.py` | Token sanitization, atomic writes, UMI fallback |
| `backend/modules/chat/service.py` | Sanitizer extracted; cleanup |
| `backend/logger.py` | Connection-string AAD auth, MSAL log suppression, format unification |
| `Deployments/templates/AML.Template.json` | New role assignment for AI Search private endpoint |
| `nginx.conf` | `/health/openai` route |
| `pyproject.toml` / `poetry.lock` | tree-sitter, orjson |
| `src/app/[owner]/[repo]/codetrace/page.tsx` | NEW — 3-panel CodeTrace page |
| `src/app/api/codemap_cache/route.ts` | NEW — frontend codemap proxy |
| `src/app/api/codetrace/route.ts` | NEW — frontend codetrace proxy |
| `src/app/api/health/openai/route.ts` | NEW — health proxy |
| `src/components/CodeMap.tsx` | NEW — D3 graph viewer (503+ lines) |
| `src/components/Ask.tsx` | Mode dropdown (Chat / Deep Research / Code Trace), connection gating, AI disclaimer |
| `src/components/AzureIcon.tsx` → `OrcasLogo` | Rebranded |
| `src/components/DESIGN.md` | NEW — frontend architecture doc |
| `src/hooks/useCodeMap.ts` | NEW — codemap data hook |
| `src/hooks/useCodeTrace.ts` | NEW — codetrace data hook |
| `src/types/codemap.ts` / `codetrace.ts` | NEW — frontend types |
| `Note/release-202604.md` | This document |

---

## Known Limitations Carried Over

- FAISS construction still has a brief ~1.4 GB peak during reload; Phase 3b memory-mapped FAISS deferred (cloud chat uses AI Search anyway).
- CodeTrace returns single-shot responses; large traces take 10–15 s; no caching across requests.
- CodeMap LOD is heuristic — very dense graphs (>2,000 nodes) still need user-driven filtering.
- Ask page still relies on backend reachability check; if `/health/openai` is slow, the Ask button stays disabled.
