# DeepWiki May Release Notes

**Date:** 2026-06-01
**Branch:** `dev/xixia/v2`
**Period covered:** 2026-05-02 → 2026-06-01 (since [release-202604.md](./release-202604.md))

---

## Highlights

- **Delta embedding (incremental reruns)** — the processor now re-embeds only the files that actually changed between runs. A sidecar manifest tracks per-file sha256 + chunk mapping; reruns skip unchanged files, purge stale chunks, and reuse cached vectors. Always on, no config knob.
- **Stable AI Search document keys** — deterministic `{repo}_{branch}_{path}_{chunk_index:03d}` keys enable selective upsert/delete of just the changed slice instead of full-index rebuilds.
- **Wiki/codemap fast-path skip** — if a wiki cache already exists for the current commit and the embedder is unchanged, the pipeline short-circuits right after clone.
- **`--full-reprocess` CLI flag** — opt-in escape hatch to ignore the manifest and rebuild everything (used after embedder model/dimension changes or schema bumps).

---

## New Features

### Delta Embedding — Incremental Reruns

Re-running the processor on the same repository is now incremental by default. Inspired by the OrcasCopilot TSG delta pattern, the embedder maintains a manifest sidecar and only does work proportional to what changed.

**How it works**

- **Manifest sidecar** ([backend/clients/vector_storage.py](../backend/clients/vector_storage.py)) — `_manifest.json` written next to the chunk files at `vectors/<owner>_<repo>_<branch>/_manifest.json`. Records schema version, HEAD commit hash, embedder signature (`deployment` / `model_name` / `vector_dim`), and per-source `sha256`, `size`, `chunk_count`, and `chunk_files`. The underscore prefix keeps it out of the chunk-file walk (`_is_chunk_filename`).
- **Delta detector** ([backend/modules/embedder/delta.py](../backend/modules/embedder/delta.py)) — `compute_file_delta()` decides what to embed/delete/reuse via an ordered set of signals:
  1. **cold_start** — no previous manifest → embed everything.
  2. **manifest_schema_changed** — manifest version bump → full reprocess.
  3. **embedder_changed** — deployment/model/dim signature differs → full reprocess.
  4. **fast_path_commit_match** — same HEAD commit + clean working tree → no embedding work.
  5. **git_diff** — `git diff --name-status` + `git status --porcelain` between the manifest commit and current HEAD.
  6. **sha256** — the final arbiter; a content-hash mismatch always forces re-embedding even when git says "unchanged", and a match overrides a stale mtime.
- **Git change detection** ([backend/modules/repository/git_ops.py](../backend/modules/repository/git_ops.py)) — `get_changed_files(local_path, prev_commit_hash)` combines committed diffs and working-tree status into a single `{path: status}` map.
- **Plumbed through the embed pipeline** ([backend/modules/embedder/document.py](../backend/modules/embedder/document.py), [backend/modules/embedder/indexer.py](../backend/modules/embedder/indexer.py)) — `transform_documents_and_save_as_json()` accepts a `delta_to_embed` set and returns the per-source chunk mapping so the manifest can be rebuilt accurately. `prepare_db_index()` computes the delta, deletes stale chunks for removed/changed sources, embeds only the changed slice, and loads unchanged documents back from the vector store so FAISS still receives the full corpus.
- **Selective AI Search push** ([backend/clients/search_client.py](../backend/clients/search_client.py)) — deterministic document keys (`_build_doc_key`) mean a changed file's rows can be deleted (`delete_documents_for_sources`) and re-upserted in place, instead of rebuilding the entire index. Cloud mode pushes only the changed slice.
- **Wiki/codemap fast-path** ([backend/modules/wiki/cache.py](../backend/modules/wiki/cache.py)) — `wiki_cache_exists_for_commit()` lets `_process()` skip the whole embed→wiki pipeline after clone when a cache for the same commit already exists.

**Measured impact** (47K-file `orcasql-breadth` repo, real rerun on 2026-06-01):

| Metric | Cold start | Delta rerun | Savings |
|--------|-----------|-------------|---------|
| Reason | `cold_start` | `git_diff` | — |
| Files embedded | 10,698 | 174 | **98.4% fewer** |
| Files reused (unchanged) | 0 | 10,542 | from cache |
| Stale sources purged | 0 | 156 (1,293 chunks) | — |
| Embed wall time | ~2,698 s (~45 min) | ~179 s (~3 min) | **~15× faster** |
| Chunks freshly embedded | 21,580 | 1,358 | ~94% fewer embedding calls |

Re-embedding cost is now proportional to the diff, cutting both wall-clock time and Azure OpenAI embedding token spend on routine reruns.

### `--full-reprocess` CLI Flag

[backend/processor/code_processor.py](../backend/processor/code_processor.py) gained a `--full-reprocess` flag that bypasses the manifest and rebuilds everything from scratch. Used after an embedder model/dimension change or an index-schema bump, or any time a clean rebuild is desired.

---

## Bug Fixes

- **Empty sha256 on full-reprocess paths** — `_full_reprocess()` in [backend/modules/embedder/delta.py](../backend/modules/embedder/delta.py) previously returned an empty `file_hashes` map, so cold-start / embedder-change / schema-change runs persisted a manifest with blank `sha256` for every file. The next run could then not do hash-based delta detection and silently degraded to a full reprocess. Now every file written on a full-reprocess path is hashed up front, leaving a valid manifest behind.
- **Manifest self-heal on fast path** — [backend/modules/embedder/indexer.py](../backend/modules/embedder/indexer.py) now opportunistically backfills missing `sha256` values for unchanged entries when the fast-path verifier already computed them, then rewrites the manifest. Existing manifests written before the fix above heal themselves on the next pull (`[Vec] Backfilled sha256 for N manifest entries`).
- **Stale chunk cleanup on rename/delete** — deleted, renamed, and content-changed sources have their old chunk files removed before new chunks are written, preventing orphan vectors (`delete_files_for_sources`).

---

## Documentation

- [backend/processor/README.md](../backend/processor/README.md) — added the **Incremental reruns (delta embedding)** section documenting the manifest sidecar, the delta decision order, selective AI Search push, the post-clone fast-path, and the `--full-reprocess` flag.
- This release note.

---

## Files Changed (Highlights)

| File | Change |
|------|--------|
| `backend/modules/embedder/delta.py` | NEW — delta detector (`compute_file_delta`, `DeltaResult`, `_full_reprocess`, `summarize_for_log`) |
| `backend/modules/repository/git_ops.py` | `get_changed_files()` — committed diff + working-tree status combined |
| `backend/modules/embedder/indexer.py` | Delta computation, stale-chunk purge, unchanged-doc reload, manifest build + self-heal |
| `backend/modules/embedder/document.py` | `delta_to_embed` slice + `return_chunks_by_source` for accurate manifest rebuild |
| `backend/clients/vector_storage.py` | Manifest read/write (atomic), sidecar filter, `delete_files_for_sources`, `iter_documents_for_sources` |
| `backend/clients/search_client.py` | Deterministic doc keys (`_build_doc_key`), `delete_documents_for_sources`, selective push |
| `backend/modules/wiki/cache.py` | `wiki_cache_exists_for_commit()` synchronous probe for post-clone fast-path |
| `backend/processor/code_processor.py` | Selective embed/push contract, wiki fast-path skip, `--full-reprocess` CLI flag |
| `backend/processor/README.md` | Delta-embedding documentation |
| `Note/release-202605.md` | This document |

---

## Known Limitations Carried Over

- **Chunk-filename collisions** — `_get_json_filename` strips the extension, so `foo.c` and `foo.h` in the same directory both map to `foo_001.json`. The manifest then lists both sources against the same chunk file; deleting one can drop the other's chunk. Pre-existing latent issue surfaced by manifest inspection (observed as a small `Deleted N/M orphan files` gap). Candidate fix: extension-preserving or hash-based chunk filenames. Deferred.
- Delta detection requires a git working tree; non-git sources fall back to full reprocess.
- FAISS construction still reloads unchanged documents into memory to assemble the full index (cloud chat uses AI Search and avoids this).
