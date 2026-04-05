# Codemap Memory Optimization Plan

> **Context**: After adding codemap + codetrace features, the AML processor (STANDARD_D11_V2, 14 GB RAM)
> shows unexpectedly high memory usage for super-large repos (50K+ files, e.g. SQL Server engine).
> The embedding pipeline was previously optimized and remains efficient. The regression is isolated
> to the codemap module's unbounded in-memory graph and serialization strategy.
>
> **CodeTrace is not affected** — it runs as an API service querying pre-built indexes, not during AML processing.

---

## Diagnosis Summary

| Issue | File | Impact |
|-------|------|--------|
| Unbounded `all_nodes` / `all_edges` lists | `graph_builder.py` | ~250 MB for 50K files |
| Triple-copy during serialization (object + dict + JSON string) | `cache.py` | ~1 GB peak |
| No `gc.collect()` after codemap step | `code_processor.py` | Fragmented heap carries into embedding |
| `file_filter` param never passed from processor | `code_processor.py` | Processes files embedder skips |
| `indent=2` JSON adds 3-4x text overhead | `cache.py` | ~500 MB JSON string |

**Estimated peak RSS for 50K-file repo**: ~2.5 GB during codemap JSON serialization alone.

---

## Fix 1: Add `gc.collect()` After Codemap Step

**File**: `backend/processor/code_processor.py`
**Risk**: None — data already serialized before cleanup
**Quality impact**: None

In `_process()`, after the `step_build_codemap()` call, add explicit cleanup.

### Current code (~line 575-585):

```python
    # Build codemap graph (all modes, unless skipped)
    if not skip_codemap:
        try:
            step_build_codemap(
                repo_dir, owner, repo, 'azuredevops', branch,
            )
        except Exception as e:
            logger.warning(f"Codemap generation failed (non-fatal): {e}")
            print(f"  WARNING: Codemap generation failed: {e}")
```

### Change to:

```python
    # Build codemap graph (all modes, unless skipped)
    if not skip_codemap:
        try:
            step_build_codemap(
                repo_dir, owner, repo, 'azuredevops', branch,
            )
        except Exception as e:
            logger.warning(f"Codemap generation failed (non-fatal): {e}")
            print(f"  WARNING: Codemap generation failed: {e}")
        finally:
            gc.collect()
```

The `finally` ensures cleanup even if codemap succeeds (the return value is already
discarded by the caller). This reclaims ~250–1000 MB of fragmented heap before the
embedding step begins.

---

## Fix 2: Eliminate Triple-Copy in Serialization

**File**: `backend/modules/codemap/cache.py` — `save_codemap_cache()`
**Risk**: None
**Quality impact**: None — same data, just compact formatting

### Problem

```python
def save_codemap_cache(data, ...):
    payload = data.model_dump()           # Copy 1: dict (~250 MB)
    content = json.dumps(payload, indent=2)  # Copy 2: string (~500 MB)
    # data still alive                       # Copy 0: still in scope (~250 MB)
    # Peak: ~1 GB
```

### Change to (blob path):

```python
def save_codemap_cache(data, ...):
    payload = data.model_dump()
    del data  # Free Pydantic object before building JSON string

    if is_blob_storage_configured():
        blob_path = get_codemap_blob_path(owner, repo, repo_type, branch)
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                logger.error("Blob storage configured but client unavailable")
                return False
            content = json.dumps(payload, separators=(',', ':'))
            del payload  # Free dict before upload
            if blob_client.upload_text(blob_path, content):
                logger.info(f"Codemap saved to blob: {blob_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save codemap to blob: {e}")
            return False

    # Local storage
    cache_path = get_codemap_cache_path(owner, repo, repo_type, branch)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, separators=(',', ':'))
        del payload
        logger.info(f"Codemap saved to: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save codemap to {cache_path}: {e}")
        return False
```

### Key changes

1. `del data` before `json.dumps()` — avoids Pydantic object + dict + string coexisting
2. `separators=(',', ':')` instead of `indent=2` — 3-4x smaller JSON string
3. `del payload` after use — frees dict before upload/write completes
4. Local path uses `json.dump(payload, f, ...)` to stream to file (avoids string copy)

### Memory savings

| Before | After | Saved |
|--------|-------|-------|
| ~1 GB peak (3 copies) | ~350 MB peak (1 copy at a time) | ~650 MB |

**Note**: `save_codemap_cache` takes `data` by reference from `step_build_codemap`.
The `del data` only decrements the refcount of the local parameter. The caller
(`step_build_codemap`) still holds its own reference via the `codemap` variable.
To fully benefit, also update `step_build_codemap` (see Fix 1 — `gc.collect()`
reclaims it after the function returns).

---

## Fix 3: Pass File Filter to Codemap

**File**: `backend/processor/code_processor.py` — `step_build_codemap()`
**Risk**: Low — aligns codemap file set with embedder
**Quality impact**: Positive — consistent coverage between codemap and wiki/search

### Current code:

```python
def step_build_codemap(repo_path, owner, repo, repo_type, branch):
    """Build codemap graph from cloned repository."""
    from backend.modules.codemap.graph_builder import build_codemap
    from backend.modules.codemap.cache import save_codemap_cache

    print("\n--- Step: Building codemap graph ---")
    codemap = build_codemap(repo_path)
    ...
```

### Change to:

```python
def step_build_codemap(repo_path, owner, repo, repo_type, branch):
    """Build codemap graph from cloned repository."""
    from backend.modules.codemap.graph_builder import build_codemap
    from backend.modules.codemap.cache import save_codemap_cache
    from backend.config import get_file_filters_config
    from backend.types import FileFilter

    print("\n--- Step: Building codemap graph ---")

    # Reuse the same file filter as the embedder (excluded.json)
    file_filters = get_file_filters_config()
    file_filter = FileFilter(
        excluded_dirs=set(file_filters["excluded_dirs"]),
        excluded_patterns=set(file_filters["excluded_files"]),
    )

    codemap = build_codemap(repo_path, file_filter=file_filter)
    ...
```

The `file_filter` parameter already exists on `build_codemap()` and is already
wired into `_collect_files()` — it's just never passed today.

---

## Fix 4: Add MAX_SYMBOLS Budget

**File**: `backend/modules/codemap/graph_builder.py` — `build_codemap()`
**Risk**: Low — graceful degradation for oversized repos
**Quality impact**: Files beyond budget get file-level nodes only (no internal symbols).
  Cross-file edges between file nodes are still resolved. Frontend "Files view" unaffected.

### Add constant at module level:

```python
# Maximum number of symbols to extract via AST analysis.
# Beyond this, files still appear as file nodes but without
# internal symbols (functions, classes, methods).
# Prevents unbounded memory growth for very large repos.
MAX_SYMBOLS = 200_000
```

### Modify the Phase 2 loop in `build_codemap()`:

```python
    # Phase 2: Analyze files in parallel
    all_nodes: List[SymbolNode] = []
    all_edges: List[SymbolEdge] = []
    language_stats: Dict[str, int] = defaultdict(int)
    file_nodes: List[SymbolNode] = []
    symbol_budget_exhausted = False
    current_symbol_count = 0

    workers = max_workers or min(os.cpu_count() or 4, len(files))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for rel_path, ext, abs_path in files:
            futures[pool.submit(
                _analyze_one_file, abs_path, rel_path, ext
            )] = (rel_path, ext)

        for future in as_completed(futures):
            rel_path, ext = futures[future]
            try:
                nodes, edges = future.result()
            except Exception as e:
                logger.debug(f"[CodeMap] Error analyzing {rel_path}: {e}")
                nodes, edges = [], []

            lang_name = LANGUAGE_DISPLAY.get(ext, ext)
            language_stats[lang_name] += 1

            # Create file-level node (always kept)
            file_node = SymbolNode(
                id=rel_path,
                name=os.path.basename(rel_path),
                kind='file',
                file_path=rel_path,
                language=lang_name,
            )
            file_nodes.append(file_node)

            # Only accumulate symbols if budget allows
            if not symbol_budget_exhausted:
                current_symbol_count += len(nodes)
                all_nodes.extend(nodes)
                all_edges.extend(edges)

                if current_symbol_count >= MAX_SYMBOLS:
                    symbol_budget_exhausted = True
                    logger.warning(
                        f"[CodeMap] Symbol budget ({MAX_SYMBOLS}) reached "
                        f"at {len(file_nodes)} files, "
                        f"{current_symbol_count} symbols. "
                        f"Remaining files will be file-level only."
                    )
```

### Why MAX_SYMBOLS instead of MAX_FILES

- A 50K-file repo with mostly config/data files may only produce 10K symbols — no issue.
- A 5K-file repo with dense C++ classes may produce 300K symbols — real problem.
- Symbol count directly correlates with memory consumption, file count doesn't.

### Estimated effect

| Repo | Files | Symbols (before) | Symbols (after) | Memory saved |
|------|-------|-------------------|------------------|--------------|
| SQL Server engine | 50K | ~350K | 200K (capped) | ~100 MB |
| Small repo (500 files) | 500 | 5K | 5K (no cap) | 0 |
| Medium repo (5K files) | 5K | 50K | 50K (no cap) | 0 |

---

## Fix 5: Free Lookup Tables After Reference Resolution

**File**: `backend/modules/codemap/graph_builder.py` — `build_codemap()`
**Risk**: None
**Quality impact**: None

### Add cleanup after Phase 3 / Phase 4:

```python
    all_nodes = file_nodes + all_nodes

    # Phase 3: Resolve cross-file references
    resolved_edges = _resolve_references(
        all_nodes, all_edges, repo_path,
    )

    # Free raw edges — no longer needed after resolution
    del all_edges

    # Phase 4: Deduplicate edges
    seen_edges: Set[Tuple[str, str, str]] = set()
    unique_edges: List[SymbolEdge] = []
    for edge in resolved_edges:
        key = (edge.source_id, edge.target_id, edge.kind)
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    # Free intermediate structures
    del resolved_edges
    del seen_edges
```

Also in `_resolve_references()`, add cleanup before return:

```python
def _resolve_references(nodes, edges, repo_path):
    node_by_id = {n.id: n for n in nodes}
    symbols_by_name = defaultdict(list)
    for n in nodes:
        if n.kind != 'file':
            symbols_by_name[n.name].append(n.id)
    file_path_set = {n.id for n in nodes if n.kind == 'file'}

    resolved = []
    for edge in edges:
        # ... resolution logic ...

    # Free lookup tables before returning
    del node_by_id
    del symbols_by_name
    del file_path_set

    return resolved
```

### Memory savings

For 50K files: `node_by_id` (~84 MB) + `symbols_by_name` (~15 MB) + `file_path_set` (~3 MB) = **~100 MB freed** before serialization.

---

## Fix 6 (Optional): Add RSS Logging

**File**: `backend/processor/code_processor.py`
**Risk**: None — observability only
**Quality impact**: None

### Add helper at module level:

```python
def _log_rss(label: str):
    """Log current process RSS for memory debugging."""
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"  [MEM] {label}: {rss_mb:.0f} MB RSS")
        logger.info(f"[MEM] {label}: {rss_mb:.0f} MB RSS")
    except ImportError:
        pass
```

### Add calls at each step boundary in `_process()`:

```python
    _log_rss("after clone")

    if not skip_codemap:
        try:
            step_build_codemap(...)
        except Exception as e:
            ...
        finally:
            gc.collect()
    _log_rss("after codemap")

    if mode == 'cloud':
        step_embed_cloud(...)
        _log_rss("after embed_cloud")

        step_push_to_search(...)
        _log_rss("after push_to_search")

        wiki_data = step_generate_wiki_cloud(...)
        _log_rss("after generate_wiki")

        step_save_wiki(...)
        _log_rss("after save_wiki")
```

This produces output like:

```
  [MEM] after clone: 480 MB RSS
  [MEM] after codemap: 520 MB RSS      ← should drop after gc.collect()
  [MEM] after embed_cloud: 650 MB RSS
  [MEM] after push_to_search: 700 MB RSS
  [MEM] after generate_wiki: 850 MB RSS
  [MEM] after save_wiki: 400 MB RSS
```

Use this to validate the fixes and catch future regressions.

---

## Quality Impact Summary

| Fix | Codemap Quality | CodeTrace Quality | Notes |
|-----|-----------------|-------------------|-------|
| Fix 1: gc.collect() | **None** | **None** | Data already serialized before cleanup |
| Fix 2: Compact JSON | **None** | **None** | Same data, different whitespace; frontend parses identically |
| Fix 3: File filter | **Low risk** | **None** | Aligns codemap with embedder file set — improves consistency but may hide files if `excluded.json` is overly aggressive. Review `excluded.json` patterns before applying. |
| Fix 4: MAX_SYMBOLS | **Degrades for huge repos** | **None** | Files beyond 200K symbol budget appear as file-only nodes (no internal functions/classes/methods). Cross-file call and inheritance edges to/from those symbols are **lost**. Frontend "Files view" unaffected; "Graph view" loses detail in the tail. |
| Fix 5: Free lookups | **None** | **None** | Frees temporary dicts after resolution is complete |
| Fix 6: RSS logging | **None** | **None** | Observability only |

**CodeTrace is completely unaffected by all fixes** — it queries the pre-built RAG index
(AI Search in cloud mode), never reads codemap data or its JSON cache.

---

## Apply Order

| Order | Fix | Files Changed | Effort |
|-------|-----|---------------|--------|
| 1 | Fix 1: gc.collect() after codemap | `code_processor.py` | 5 min |
| 2 | Fix 2: Eliminate triple-copy + compact JSON | `cache.py` | 15 min |
| 3 | Fix 5: Free lookup tables | `graph_builder.py` | 10 min |
| 4 | Fix 3: Pass file filter | `code_processor.py` | 10 min |
| 5 | Fix 6: RSS logging | `code_processor.py` | 10 min |
| 6 | Fix 4: MAX_SYMBOLS budget | `graph_builder.py` | 20 min |

**Fixes 1, 2, 5** are zero-risk, zero-quality-impact changes. Apply and deploy first.

**Fix 3** (file filter) is low-risk but review `excluded.json` to ensure no important source directories are filtered.

**Fix 6** (RSS logging) is observability only — deploy with Fix 3 or independently.

**Fix 4** (MAX_SYMBOLS) trades codemap completeness for bounded memory. Deploy last, monitor with Fix 6 to determine if the budget is actually hit for your repos.

---

## Validation

1. **Baseline**: Run SQL Server engine repo with `skip_codemap=True`, record peak RSS
2. **Before fix**: Run with codemap enabled, record peak RSS — expect ~2.5 GB peak at serialization
3. **After Fixes 1-3**: Run again — expect peak to drop to ~800 MB at codemap step
4. **After Fix 4**: Run with 200K symbol budget — verify codemap JSON still loads in frontend, file tree complete
5. **After Fix 6**: Check AML job logs / Application Insights for `[MEM]` entries

### Quick local test

```bash
python -c "
from backend.modules.codemap.graph_builder import build_codemap
import psutil, os
proc = psutil.Process()
print(f'Before: {proc.memory_info().rss / 1024**2:.0f} MB')
data = build_codemap('/path/to/large/repo')
print(f'After build: {proc.memory_info().rss / 1024**2:.0f} MB')
print(f'Nodes: {len(data.nodes)}, Edges: {len(data.edges)}')
from backend.modules.codemap.cache import save_codemap_cache
save_codemap_cache(data, 'test', 'repo', 'local')
print(f'After save: {proc.memory_info().rss / 1024**2:.0f} MB')
import gc; gc.collect()
print(f'After gc: {proc.memory_info().rss / 1024**2:.0f} MB')
"
```
