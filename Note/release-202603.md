# DeepWiki v2 Release Notes

**Date:** 2026-03-24  
**Branch:** `dev/xixia/v2`

---

## Highlights

- **3-mode processor architecture** (local / docker / cloud) with a single `--mode` switch
- **AML pipeline scheduler** for automated cloud processing
- **Blob-aware web app** — frontend proxies wiki reads through backend for Azure Blob Storage
- **Large repo support** — automatic file-tree truncation for repos exceeding 10,000 files
- **Memory optimization analysis** — identified 6 hotspots with phased optimization plan
- **Phase 1 memory optimization implemented** — ~40-50% peak memory reduction

---

## New Features

### Processor 3-Mode Design
- Refactored `code_processor.py` into discrete step functions: `resolve_auth()` → `step_clone()` → `step_embed()` → `step_generate_wiki()` → `step_save_wiki()` → `step_push_to_search()`
- Mode determines auth strategy (PAT / Azure CLI / MSI), storage backend (local disk / Azure Blob), and retrieval engine (FAISS / AI Search)
- Config overlay system: `.cloud/` and `.local/` directories under `backend/config/`

### AML Pipeline Dispatcher (`aml_dispatcher.py`)
- Standalone CLI for cloud resource provisioning: `python -m backend.processor.aml_dispatcher`
- Creates AML compute cluster, Docker environment, and scheduled pipeline jobs
- `RecurrenceTrigger` schedule — fires immediately on creation, no duplicate first-run submission
- Experiment name sanitization: `re.sub(r'[^a-zA-Z0-9_-]', '-', name)` for URL-decoded repo paths

### Blob-Aware Frontend
- New FastAPI endpoints: `/api/wiki_cache` and `/api/processed_projects` proxy to Python backend
- Frontend routes check `FASTAPI_PORT` env var — if set, proxy to backend; otherwise read local disk
- Supports both local disk and Azure Blob Storage transparently

### Large Repo File Tree Truncation
- Repos with >10,000 files auto-switch to directory-only tree for wiki structure generation
- Prevents exceeding model input token limits (272K for gpt-5.1)

---

## Bug Fixes

- **Config cache race condition**: `set_config_dir()` now clears ALL cached config objects (`_infra_config`, `_embedder_config`, etc.) to prevent stale config when switching directories
- **App Insights auth log spam**: Switched from credential-based AAD auth to connection string auth
- **`BACKEND_PORT` always truthy**: Next.js sets `process.env.PORT` automatically — changed frontend proxy guard to check `FASTAPI_PORT` only
- **AML duplicate pipeline runs**: Removed explicit `first_run` job submission — `RecurrenceTrigger` already fires immediately on schedule creation
- **AML experiment name validation**: Spaces in URL-decoded repo paths (e.g., `Database%20Systems`) caused `ValidationError` — now sanitized with regex
- **`re` import scope**: Added `import re` to `_create_or_update_pipeline()` function where `re.sub()` is called

---

## Infrastructure Changes

- **`publish-web.ps1`**: Creates `.cloud/` config overlay with blob + search + App Insights enabled; sets `DEEPWIKI_CONFIG_DIR=backend/config/.cloud` as container app setting
- **Blob client**: Connection pool size increased to 50 connections for concurrent uploads
- **Zero-downtime reprocessing**: Snapshot existing vector files before re-embedding; new chunks overwrite in-place; orphan files cleaned up after completion

---

## Documentation

- **README.md**: Complete rewrite with architecture diagram, 3-mode table, project structure, quick start guide, Azure resource requirements, and security notes
- **Module READMEs**: Updated all `backend/modules/*/README.md` for accuracy (endpoint counts, filename formats, save behavior)

---

## Memory Optimization Analysis

Identified 6 memory hotspots in the embedding + wiki generation pipeline:

| # | Severity | Hotspot | Impact (10K-file repo) |
|---|----------|---------|------------------------|
| 1 | CRITICAL | Duplicate content (`text` + `raw_content`) in Document objects | 2x raw doc memory (~400MB vs ~200MB) |
| 2 | CRITICAL | All data held simultaneously during embedding (raw + chunks + embeddings) | ~950MB peak |
| 3 | HIGH | FAISS double-stores vectors (document list + index) | 2x vector memory |
| 4 | HIGH | No garbage collection between pipeline stages | Embedding memory persists during wiki gen |
| 5 | MEDIUM | 22 recursive glob traversals (one per file extension) | Slow I/O |
| 6 | MEDIUM | LocalDB holds triple references to data | Prevents GC |

### Proposed Optimization Phases

- **Phase 1 (Quick Wins)**: ~~Remove `raw_content` duplication; explicit `del` + `gc.collect()` between stages; clear `transformed_docs` after FAISS construction; single `os.walk()` pass~~ — **COMPLETED** ✅
- **Phase 2 (Batched Processing)**: ~~Batch-embed in groups of ~500 chunks; stream-save to JSON per batch; release `documents` in indexer.py~~ — **COMPLETED** ✅
- **Phase 3a (Pipeline Cleanup)**: ~~Free retriever after wiki gen; single-pass embedding validation; bypass adalflow LocalDB/ToEmbeddings index bug~~ — **COMPLETED** ✅
- **Phase 3b (Architecture, future)**: ~~Memory-mapped FAISS index; streaming vector loading; generator-based document reader~~ — **DEFERRED** (FAISS requires all vectors at construction; cloud chat already uses AI Search; remaining ~200MB peak acceptable)
- **Phase 4 (Fused Read+Split)**: ~~Fused read+split batching in `transform_documents_and_save_as_json()`; reads 1000 files at a time instead of entire repo~~ — **COMPLETED** ✅
- **Phase 5 (Eliminate Accumulation)**: ~~Stop accumulating `all_transformed`; batch `step_push_to_search()`; reload from storage for FAISS~~ — **COMPLETED** ✅

### Phase 1 Implementation Details

| Change | File(s) | Memory Savings |
|--------|---------|----------------|
| Single `os.walk()` pass (was 22 `glob` traversals) | `document.py` | Faster I/O, reduced directory traversal overhead |
| Removed `raw_content` duplication from Document `meta_data` | `document.py` | ~50% raw document memory saved |
| Excluded `raw_content`/`token_count` from chunk spread (`**doc.meta_data`) | `code_splitter.py` | Eliminated N× full-file duplication across chunks |
| `del documents` + `gc.collect()` after splitting; `del db` + `gc.collect()` after embedding | `document.py` | ~350MB freed between pipeline stages |
| Strip `doc.vector = None` after FAISS index construction | `retriever.py` | ~600MB freed (50K chunks × 12KB/vector) |
| Use `raw_chunk_text` instead of `raw_content` for AI Search uploads | `search_client.py` | Correctness fix: chunks now upload their own text, not full file content |

**Estimated peak memory reduction:** ~40-50% for a typical 10K-file repository.  
**Wiki content quality:** No impact — all retrieval paths use `raw_chunk_text` (already set by code splitter).

### Phase 2 Implementation Details

| Change | File(s) | Memory Savings |
|--------|---------|----------------|
| `del documents; gc.collect()` after `transform_documents_and_save_as_json()` | `indexer.py` | Frees raw document memory before orphan cleanup + return |
| Batched embedding loop (500 chunks per batch) | `document.py` | Peak drops from ~900MB to ~9MB per batch during embedding |
| Per-batch `save_documents()` + `del db; gc.collect()` | `document.py` | LocalDB internals released after each batch instead of held for entire run |

**How it works:**
- After code-aware splitting, `enriched_chunks` is partitioned into batches of `EMBED_BATCH_SIZE = 500`
- Each batch: `LocalDB()` → `load(batch)` → `transform()` → `get_transformed_data()` → `save_documents()` → `del db; gc.collect()`
- `all_transformed` list accumulates across batches for FAISS construction (required)
- Per-batch logging: `"Batch 3/10: embedded + saved 500 chunks (1500/5000 total)"`

**Peak memory comparison (50K chunks, 10K-file repo):**

| Phase | During embedding | During FAISS | Total peak |
|-------|-----------------|--------------|------------|
| Before Phase 1 | ~1.8GB | ~1.2GB | ~1.8GB |
| After Phase 1 | ~900MB | ~200MB | ~900MB |
| After Phase 2 | ~9MB/batch | ~200MB | ~200MB |
| After Phase 3a | ~9MB/batch | ~200MB (freed before wiki save) | ~200MB |

**Estimated combined peak memory reduction (Phase 1+2+3a):** ~85-90% for a typical 10K-file repository.  
**Wiki content quality impact:** None — each chunk is embedded independently; batch boundaries don't affect embedding vectors, FAISS index, AI Search content, or retrieval results.

### Phase 3a Implementation Details

| Change | File(s) | Memory Savings |
|--------|---------|----------------|
| `del retriever; gc.collect()` after wiki generation, before save/push | `code_processor.py` | Frees FAISS index + transformed_docs (~70MB for 2.5K chunks) during wiki save |
| Single-pass `_validate_and_filter_embeddings()` (was two-pass) | `retriever.py` | Groups docs by size in one loop via `docs_by_size` dict; eliminates redundant re-traversal |
| Bypass adalflow `LocalDB` + `ToEmbeddings` (index bug fix) | `document.py` | Direct embedder calls with explicit sub-batching; eliminates `output[batch_idx * batch_size + idx]` IndexError |
| Cleaned `repo.json`: normalized `excluded_dirs`, removed broken patterns, removed unused `max_size_mb` | `repo.json` | Correctness: `./` prefix removed, directory entries moved from `excluded_files` to `excluded_dirs`, broken `packages/*/dist` patterns removed |

**adalflow `ToEmbeddings` bug:** `data_components.py` line 96 uses `output[batch_idx * self.batch_size + idx]` where `batch_idx` is a start index from `range(0, n, batch_size)`, not an enumerate counter. With `batch_size=10` and >10 chunks, produces index `100` instead of `10` → `IndexError`. Fixed by calling the embedder directly with proper sub-batching.

### Phase 4 Implementation Details

| Change | File(s) | Memory Savings |
|--------|---------|----------------|
| Fused read+split: `transform_documents_and_save_as_json()` now takes `repo_path` instead of pre-loaded `documents` list | `document.py` | **~5GB peak reduction** for 47K-file repos (was loading all files, now batches of 1000) |
| `_compute_file_url()` extracted as module-level function | `document.py` | Code cleanup for reuse |
| `indexer.py` passes repo path + filter params instead of pre-loaded document list | `indexer.py` | Removed unused `gc`, `read_all_documents` imports |

**How it works:**
- Walks directory once to collect file paths (~10MB for 35K files)
- Processes in batches of `FILE_BATCH_SIZE = 1000`: read → split into enriched chunks → release batch
- Peak during read+split: ~150MB (1000 files) instead of ~5.2GB (all 35K files)
- Enriched chunks still accumulate for embedding (~225MB), but raw file content is freed per batch

### Phase 5 Implementation Details (OOM Fix for Large Repos)

**Problem:** Processor OOMed on AML compute (14GB RAM) for 47K+ file repos due to three hotspots.

| # | Hotspot | Before | After |
|---|---------|:------:|:-----:|
| 1 | `all_transformed` accumulation during embedding | **830MB** sustained | **0** (return count, not list) |
| 2 | `step_push_to_search()` full vector reload | **725MB** peak | **~15MB** (strip vectors per batch) |
| 3 | FAISS construction (unavoidable spike) | **2.08GB** (on top of #1) | **1.44GB** (clean load from storage) |

| Change | File(s) | Memory Savings |
|--------|---------|----------------|
| `transform_documents_and_save_as_json()` returns `int` (chunk count) instead of `List[Document]` | `document.py` | **830MB → 0** during embedding — vectors saved per batch and released, never accumulated |
| `indexer.py` reloads from `vector_storage.load_documents()` after embedding for FAISS | `indexer.py` | Clean separation: embedding phase has ~240MB peak, FAISS phase has ~1.44GB brief spike |
| `step_push_to_search()` strips `doc.vector = None` after each push batch + `gc.collect()` | `code_processor.py` | Progressive vector release during AI Search upload |

**Processor memory timeline (47K-file repo, 50K chunks, 3072-dim, AFTER Phase 5):**

```
Stage                                    Peak Memory
────────────────────────────────────────────────────
1. Clone repo                            500 MB (baseline)
2. Fused read+split (1000 files/batch)   725 MB (+225 MB enriched_chunks)
3. Embedding (500 chunks/batch)          740 MB (+15 MB per embed batch)
4. FAISS construction (reload from disk) 1.44 GB ← Brief spike, acceptable
5. Wiki generation                       217 MB (vectors stripped)
6. step_push_to_search (batched push)    830 MB (vectors freed per batch)
7. Cleanup                               100 MB
```

**Peak reduced from ~2.08GB to ~1.44GB** (brief FAISS spike). Well within 14GB AML compute.

**Wiki content quality impact:** None — same vectors, same FAISS index, same retrieval. The only difference is *when* vectors are in memory, not *what* vectors.

**Trade-off:** One extra disk/blob read for FAISS loading after embedding. For local: ~2s. For blob: ~40s. Acceptable since embedding takes 20+ minutes.

---

## AI Search Data Source Fix

**Bug:** AI Search indexers showed 0 documents despite blobs existing in storage.

**Root cause:** Data source blob query path mismatch:
- `indexer.py` saves vectors at `vectors/orcasql-mysql_8.0-master/` (just repo name)
- `cloud_setup.py` configured data source to query `vectors/msdata_orcasql-mysql_8.0-master/` (owner\_repo name)
- `step_push_to_search` used `owner_repo` to load vectors from blob — found 0 docs, never pushed

**Fix:** Changed both `cloud_setup.py` and `step_push_to_search` to use bare `repo` name (matching what `indexer.py` saves).

**Additional fix:** After correcting data source paths, indexers needed a **reset** (`client.reset_indexer()`) to clear the high-water mark and re-scan from scratch.

**Result:**

| Index | Documents Indexed |
|-------|-------------------|
| `orcasql-breadth-elasticserverv2` | 15,878 |
| `orcasql-mysqlflex-mysql` | 7,994 |
| `orcasql-mysql-8-0-master` | 2,546 |
| `orcasql-myfile-main` | 2,409 |

---

## Files Changed (Key)

| File | Change |
|------|--------|
| `backend/processor/code_processor.py` | Complete rewrite with step functions |
| `backend/processor/aml_dispatcher.py` | New — standalone cloud setup CLI |
| `backend/processor/cloud_setup.py` | Config overlays, AML schedule, experiment name fix |
| `backend/config.py` | `set_config_dir()` with cache clearing |
| `backend/app.py` | Added `/api/wiki_cache` and `/api/processed_projects` |
| `backend/modules/wiki/cache.py` | `list_wiki_caches()` with blob metadata timestamps |
| `backend/tools/logger.py` | App Insights connection string auth |
| `backend/clients/blob_client.py` | Connection pool size increase |
| `backend/processor/wiki_generator.py` | Large repo file tree truncation |
| `backend/modules/embedder/document.py` | Phase 1: single `os.walk()`, removed `raw_content`, `gc.collect()` between stages |
| `backend/modules/embedder/code_splitter.py` | Phase 1: exclude `raw_content`/`token_count` from chunk meta spread |
| `backend/modules/embedder/retriever.py` | Phase 1: strip vectors after FAISS; Phase 3a: single-pass validation |
| `backend/clients/search_client.py` | Phase 1: use `raw_chunk_text` instead of `raw_content` for AI Search |
| `backend/modules/embedder/document.py` | Phase 4: fused read+split+embed pipeline; direct embedder calls |
| `backend/modules/embedder/indexer.py` | Phase 4: pass repo path instead of pre-loaded docs |
| `backend/config/repo.json` | Normalized `excluded_dirs`, removed broken patterns, removed unused `max_size_mb` |
| `backend/clients/vector_storage.py` | Updated docstring schema (`raw_chunk_text`) |
| `backend/processor/cloud_setup.py` | AI Search data source: use bare `repo` name for blob path |
| `backend/processor/code_processor.py` | `step_push_to_search`: use bare `repo` name for vector storage load |
| `src/app/api/wiki_cache/route.ts` | Backend proxy for blob storage |
| `src/app/api/wiki/projects/route.ts` | Backend proxy for blob storage |
| `publish-web.ps1` | `.cloud` config with AI Search + blob enabled |
| `README.md` | Complete documentation rewrite |
