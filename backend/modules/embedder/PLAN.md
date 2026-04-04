# File Filtering Redesign — Two-Layer Config-Driven Pipeline

## Problem

File filtering is hardcoded in **3 files across 5 locations**, causing:
1. Files like `.xaml` silently skipped — not in the hardcoded extension whitelist, zero logging
2. `skip_dirs` duplicated 3 times (document.py ×2, wiki_generator.py ×1) — separate from `repo.json`'s `excluded_dirs`
3. No `.gitignore` awareness — repos' own ignore rules are never respected
4. Adding a new file type requires code changes instead of config changes

### Current Hardcoded Locations

| # | File | Function | What's hardcoded |
|---|------|----------|-----------------|
| 1 | `document.py` L66–68 | `read_all_documents()` | `code_extensions` (17), `doc_extensions` (6) |
| 2 | `document.py` L145–154 | `read_all_documents()` | `skip_dirs` (12 dirs) |
| 3 | `document.py` L457–463 | `transform_documents_and_save_as_json()` | `code_extensions` + `doc_extensions` (duplicate) |
| 4 | `document.py` L490–499 | `transform_documents_and_save_as_json()` | `skip_dirs` (duplicate) |
| 5 | `wiki_generator.py` L54–62 | `build_file_tree()` | `skip_dirs` (duplicate) |

Meanwhile, `repo.json` has its own `excluded_dirs` (25 dirs) and `excluded_files` (70+ patterns), but these are only used for `FileFilter` pattern matching — NOT for `os.walk()` pruning or extension gating.

---

## Design

### Two-Layer Filter Pipeline

```
File found by os.walk()
  │
  ├─ LAYER 1: EXCLUSION (any match → SKIP)
  │   ├─ Dir in excluded.json excluded_dirs?           → SKIP (prune os.walk)
  │   ├─ Matched by repo's root .gitignore?            → SKIP
  │   └─ Matched by excluded.json excluded_files?      → SKIP
  │
  ├─ LAYER 2: INCLUSION (must match to proceed)
  │   └─ Extension in included.json code or doc list?  → PROCESS
  │   └─ Not in either list?                           → SKIP
  │
  └─ Classify as code vs doc (via included.json), then PROCESS
```

A file must survive **both** layers. Exclusion catches known junk. Inclusion ensures only supported file types are embedded.

### Three Config Sources

1. **`excluded.json`** (renamed from `repo.json`) — global exclusion rules: dirs, file patterns, binary/media types
2. **`included.json`** (new) — supported file extensions split into `code` and `doc` categories
3. **`.gitignore`** (per-repo, dynamic) — parsed from the cloned repo's root `.gitignore` at runtime

---

## Config File Specifications

### `backend/config/excluded.json` (renamed from repo.json)

Current `repo.json` content is kept and extended with binary/media patterns:

```jsonc
{
  "file_filters": {
    "excluded_dirs": [
      // --- Current 25 dirs (unchanged) ---
      ".venv", "venv", "env", "virtualenv",
      "node_modules", "bower_components", "jspm_packages",
      ".git", ".svn", ".hg", ".bzr", ".github",
      "__pycache__", "dist", "build", "out", "bin", "target", "bld",
      "coverage", "htmlcov", ".nyc_output", ".tox", "lib-cov", ".output"
    ],
    "excluded_files": [
      // --- Current 70+ patterns (unchanged) ---
      "*.lock", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock",
      ".DS_Store", "Thumbs.db", "desktop.ini",
      ".env", ".env.*", "*.env",
      ".gitignore", ".gitattributes", ".gitmodules",
      "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css", "*.map",
      "*.exe", "*.dll", "*.so", "*.dylib", "*.o", "*.obj",
      "*.pyc", "*.pyd", "*.pyo", "*.class", "*.jar",
      "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z",
      // ... (all existing patterns kept)

      // --- NEW: Image / media / binary patterns ---
      "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico",
      "*.svg", "*.webp", "*.tiff", "*.tif",
      "*.mp4", "*.avi", "*.mov", "*.wmv", "*.flv", "*.webm", "*.mkv",
      "*.mp3", "*.wav", "*.ogg", "*.flac", "*.aac", "*.wma",
      "*.woff", "*.woff2", "*.ttf", "*.otf", "*.eot",
      "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
      "*.psd", "*.sketch", "*.fig", "*.ai",
      "*.sqlite", "*.db", "*.mdb", "*.bak"
    ]
  },
  "repository": {
    "max_size_mb": 50000
  }
}
```

**Changes from current `repo.json`:**
- Rename file from `repo.json` → `excluded.json`
- Add image, video, audio, font, office, design, database file patterns
- Remove `"*.json"` and `"*.txt"` from `excluded_files` — these are now managed by `included.json` (if they're in the supported list, they get processed; if not, they're skipped by Layer 2). **Note:** this changes behavior — `.json` and `.txt` files will now be processed if they're in `included.json`. If we want to keep excluding them, leave them in `excluded_files`.

### `backend/config/included.json` (new)

```json
{
  "supported_extensions": {
    "code": [
      ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
      ".go", ".rs", ".jsx", ".tsx", ".html", ".css", ".php",
      ".swift", ".cs", ".rb", ".kt", ".scala",
      ".sh", ".bash", ".ps1", ".psm1", ".bat", ".cmd",
      ".lua", ".dart", ".r",
      ".vue", ".svelte", ".razor", ".cshtml", ".astro",
      ".sql", ".graphql", ".proto",
      ".tf", ".bicep", ".hcl",
      ".xaml", ".xml", ".xsl", ".xslt",
      ".scss", ".sass", ".less", ".styl",
      ".m", ".mm", ".zig", ".nim",
      ".ex", ".exs", ".clj", ".cljs", ".erl", ".hrl",
      ".elm", ".purs", ".v", ".sv",
      ".pl", ".pm", ".vb", ".fs", ".fsx",
      ".coffee",
      ".csproj", ".sln", ".fsproj", ".vbproj",
      ".gradle", ".cmake",
      ".dockerfile"
    ],
    "doc": [
      ".md", ".txt", ".rst", ".adoc", ".tex", ".org"
    ]
  }
}
```

**Notes:**
- `code` vs `doc` classification matters because doc files > 81,920 tokens are skipped entirely, while code files of any size are split into chunks
- Extensions include the leading dot to match `os.path.splitext()` output
- Defaults in `IncludedConfig` Pydantic model match the current 17 code + 6 doc hardcoded extensions for backward compatibility

---

## New Utility: `backend/utils/filter.py`

Centralized filter helpers. Keeps `.gitignore` parsing isolated from document processing.

```python
"""
Centralized file filtering utilities.

Provides .gitignore parsing and the two-layer filter pipeline
used by document processing and wiki generation.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Set, Tuple

import pathspec

logger = logging.getLogger(__name__)


def load_gitignore(repo_path: str) -> Optional[pathspec.PathSpec]:
    """Load and compile the root .gitignore from a cloned repository.

    Args:
        repo_path: Path to the cloned repository root.

    Returns:
        Compiled PathSpec for matching, or None if no .gitignore found.
    """
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if not os.path.isfile(gitignore_path):
        logger.info(f"No .gitignore found at {repo_path}")
        return None

    try:
        with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        pattern_count = len([l for l in lines if l.strip() and not l.startswith("#")])
        logger.info(f"Loaded .gitignore with {pattern_count} patterns from {repo_path}")
        return spec
    except Exception as e:
        logger.warning(f"Failed to parse .gitignore at {repo_path}: {e}")
        return None


def is_gitignored(spec: Optional[pathspec.PathSpec], relative_path: str) -> bool:
    """Check if a file path matches the .gitignore spec.

    Args:
        spec: Compiled PathSpec from load_gitignore(), or None.
        relative_path: File path relative to repo root.

    Returns:
        True if the file should be ignored.
    """
    if spec is None:
        return False
    # Normalize to forward slashes for pathspec
    normalized = relative_path.replace("\\", "/")
    return spec.match_file(normalized)


def get_excluded_dirs(config_excluded_dirs: Set[str]) -> Set[str]:
    """Merge config excluded_dirs into the set used for os.walk() pruning.

    The config set from excluded.json is the single source of truth.
    No more hardcoded skip_dirs anywhere.

    Args:
        config_excluded_dirs: Set from excluded.json file_filters.excluded_dirs

    Returns:
        The same set (pass-through for clarity + future hook for overrides).
    """
    return set(config_excluded_dirs)
```

---

## Type Changes: `backend/types/config_types.py`

### Add `IncludedConfig`

```python
class IncludedConfig(BaseModel):
    """Supported file extensions from included.json."""
    code: List[str] = Field(
        default_factory=lambda: [
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".jsx", ".tsx", ".html", ".css", ".php",
            ".swift", ".cs",
        ]
    )
    doc: List[str] = Field(
        default_factory=lambda: [".md", ".txt", ".rst"]
    )
```

Defaults match current hardcoded values → backward compatible if `included.json` is missing.

### Update `FileFiltersConfig`

```python
class FileFiltersConfig(BaseModel):
    """File filtering configuration from excluded.json."""
    excluded_dirs: List[str] = Field(default_factory=list)
    excluded_files: List[str] = Field(default_factory=list)
```

No structural changes needed — just update the docstring from `repo.json` to `excluded.json`.

---

## Config Loading: `backend/config.py`

### Changes

1. **Rename references**: `load_json_config("repo.json")` → `load_json_config("excluded.json")` in:
   - `get_file_filters_config_obj()` (L404)
   - `get_repository_config_obj()` (L423)

2. **Add included.json loader**:

```python
_included_config: Optional[IncludedConfig] = None

def get_included_config_obj() -> IncludedConfig:
    """Load supported file extensions from included.json. Caches on first access."""
    global _included_config
    if _included_config is None:
        config_dict = load_json_config("included.json")
        if config_dict:
            try:
                ext_data = config_dict.get("supported_extensions", {})
                _included_config = from_dict(IncludedConfig, ext_data)
                logger.info("Successfully loaded included.json")
            except Exception as e:
                logger.error(f"Failed to parse included.json: {e}")
                raise
        else:
            _included_config = IncludedConfig()  # Use defaults
            logger.info("included.json not found, using default extensions")
    return _included_config

def get_included_config() -> Dict[str, Any]:
    """Get included extensions as a dict."""
    config = get_included_config_obj()
    return {"code": config.code, "doc": config.doc}
```

3. **Update `get_file_filters_config()`** to also return `max_file_size_mb`:

```python
def get_file_filters_config() -> Dict[str, Any]:
    config = get_file_filters_config_obj()
    return {
        "excluded_dirs": config.excluded_dirs,
        "excluded_files": config.excluded_files,
        "max_file_size_mb": config.max_file_size_mb,
    }
```

4. **Clear new cache in `set_config_dir()`**:

```python
def set_config_dir(path: str) -> None:
    global _included_config
    # ... existing clears ...
    _included_config = None
```

---

## Consumer Refactors

### `document.py` — `read_all_documents()` (L30–210)

**Before:**
```python
code_extensions = [".py", ".js", ".ts", ...]  # hardcoded
doc_extensions = [".md", ".txt", ...]          # hardcoded
# ...
skip_dirs = {'.git', 'node_modules', ...}      # hardcoded

for root, dirs, files in os.walk(path):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in all_ext_set:    # <-- silent skip, no log
            continue
```

**After:**
```python
from backend.config import get_file_filters_config, get_included_config
from backend.utils.filter import load_gitignore, is_gitignored

# Load from config (single source of truth)
file_filters = get_file_filters_config()
included = get_included_config()
excluded_dirs_set = set(file_filters["excluded_dirs"])
code_ext_set = set(included["code"])
doc_ext_set = set(included["doc"])
all_ext_set = code_ext_set | doc_ext_set

# Load .gitignore from cloned repo
gitignore_spec = load_gitignore(path)

# Skip counters for summary logging
skipped_gitignore = 0
skipped_excluded = 0
skipped_ext = 0
for root, dirs, files in os.walk(path):
    # Layer 1a: Prune excluded directories (from excluded.json)
    dirs[:] = [d for d in dirs if d not in excluded_dirs_set]

    for fname in files:
        full_path = os.path.join(root, fname)
        relative_path = os.path.relpath(full_path, path)

        # Layer 1b: Check .gitignore
        if is_gitignored(gitignore_spec, relative_path):
            skipped_gitignore += 1
            logger.debug(f"Skipped (gitignored): {relative_path}")
            continue

        # Layer 1c: Check excluded file patterns + size
        try:
            file_size = os.path.getsize(full_path)
        except OSError:
            continue
        if not file_filter.should_process_file(relative_path, file_size):
            skipped_excluded += 1
            logger.debug(f"Skipped (excluded): {relative_path}")
            continue

        # Layer 2: Check included extensions
        ext = os.path.splitext(fname)[1].lower()
        if ext not in all_ext_set:
            skipped_ext += 1
            logger.debug(f"Skipped (unsupported ext '{ext}'): {relative_path}")
            continue

        is_code = ext in code_ext_set
        # ... process file ...

logger.info(
    f"Processed {len(documents)} files. "
    f"Skipped: {skipped_gitignore} gitignored, "
    f"{skipped_excluded} excluded, "
    f"{skipped_ext} unsupported ext"
)
```

### `document.py` — `transform_documents_and_save_as_json()` (L390–600)

Same refactor as above — this function is a near-duplicate of the filtering logic in `read_all_documents()`.

### `wiki_generator.py` — `build_file_tree()` (L40–70)

**Before:**
```python
skip_dirs = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.next', '.nuxt', 'coverage', '.tox',
    'egg-info', '.eggs',
}
for root, dirs, files in os.walk(repo_path):
    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
```

**After:**
```python
from backend.config import get_file_filters_config

file_filters = get_file_filters_config()
excluded_dirs_set = set(file_filters["excluded_dirs"])

for root, dirs, files in os.walk(repo_path):
    dirs[:] = [d for d in dirs if d not in excluded_dirs_set and not d.startswith('.')]
```

---

## Implementation Checklist

### Phase 1: Dependencies + utility
- [ ] Add `pathspec = ">=0.11.0"` to `pyproject.toml` dependencies
- [ ] Create `backend/utils/filter.py` with `load_gitignore()` and `is_gitignored()`

### Phase 2: Config files
- [ ] Create `backend/config/included.json`
- [ ] Rename `backend/config/repo.json` → `backend/config/excluded.json`
- [ ] Add image/media/binary patterns to `excluded.json`
- [ ] Decide: keep or remove `"*.json"`, `"*.txt"` from `excluded_files` (they conflict with `included.json` `doc` list)

### Phase 3: Type + config loader updates
- [ ] Add `IncludedConfig` to `backend/types/config_types.py`
- [ ] Update `config_types.py` section comment: `repo.json` → `excluded.json`
- [ ] Add `get_included_config_obj()` / `get_included_config()` to `backend/config.py`
- [ ] Change `load_json_config("repo.json")` → `load_json_config("excluded.json")` (2 locations)
- [ ] Add `_included_config` to `set_config_dir()` cache clearing

### Phase 4: Remove hardcoded lists + add .gitignore integration
- [ ] Refactor `document.py` `read_all_documents()`:
  - Remove hardcoded `code_extensions`, `doc_extensions`, `skip_dirs`
  - Load from `get_file_filters_config()` + `get_included_config()`
  - Add `load_gitignore()` call
  - Apply two-layer filter: exclude → include
  - Add skip counters + summary log
- [ ] Refactor `document.py` `transform_documents_and_save_as_json()`:
  - Same changes (this is the cloud-mode pipeline)
- [ ] Refactor `wiki_generator.py` `build_file_tree()`:
  - Remove hardcoded `skip_dirs`, load from `get_file_filters_config()`

### Phase 5: Update references
- [ ] `test-local.ps1` L129: `"repo.json"` → `"excluded.json"`, add `"included.json"`
- [ ] `publish-web.ps1` L151: same
- [ ] Update comments referencing `repo.json` in:
  - `document.py` L84
  - `wiki_generator.py` L53
  - `config_types.py` L158, L162, L168
  - `backend/README.md` L53
  - `backend/modules/embedder/README.md` L37

### Phase 6: Verification
- [ ] `pytest backend/` — no regressions
- [ ] Grep for `skip_dirs = {` in backend/ — should find zero hits
- [ ] Grep for `code_extensions = [` or `code_extensions = {` — zero hits
- [ ] Grep for `doc_extensions = [` or `doc_extensions = {` — zero hits (only config references)
- [ ] Grep for `"repo.json"` in Python files — zero hits
- [ ] Verify `.xaml` is in `included.json` code extensions
- [ ] Test: repo with `.gitignore` excluding `*.log` → `.log` files are skipped
- [ ] Test: `.xaml` file → processed (no longer blocked by extension whitelist)

---

## File Change Summary

| Action | File | Description |
|--------|------|-------------|
| NEW | `backend/utils/filter.py` | `.gitignore` parser + filter helpers via `pathspec` |
| NEW | `backend/config/included.json` | Supported file extensions (code + doc) |
| RENAME | `backend/config/repo.json` → `excluded.json` | + binary/media patterns |
| MODIFY | `backend/modules/embedder/document.py` | Remove 4 hardcoded lists, add two-layer filter + gitignore |
| MODIFY | `backend/processor/wiki_generator.py` | Remove hardcoded `skip_dirs` |
| MODIFY | `backend/types/config_types.py` | Add `IncludedConfig`, update `FileFiltersConfig` docstring |
| MODIFY | `backend/config.py` | `repo.json` → `excluded.json`, add `included.json` loader |
| MODIFY | `pyproject.toml` | Add `pathspec` dependency |
| MODIFY | `test-local.ps1` | Update config file list |
| MODIFY | `publish-web.ps1` | Update config file list |
| UNCHANGED | `backend/types/processor_types.py` | `FileFilter` class — no changes needed |
| UNCHANGED | `backend/modules/embedder/code_splitter.py` | `LANGUAGE_MAP` is classification, not filtering |

## Open Decisions

1. **`*.json` and `*.txt` in `excluded_files`**: Current `repo.json` excludes `*.json` and `*.txt`. But `included.json` lists `.txt` under `doc` and could list `.json`. If both configs exist, exclusion wins (Layer 1 runs first). Options:
   - **Keep them in `excluded_files`** → `.json`/`.txt` never processed (current behavior)
   - **Remove from `excluded_files`** → `.json`/`.txt` processed as docs (new behavior)
   - **Recommendation**: Remove `*.txt` from `excluded_files` (it's useful doc content). Keep `*.json` excluded (too noisy for embeddings in most repos).

2. **`*.cfg`, `*.ini` in `excluded_files`**: Similar question. These are config files that may contain useful info but are usually noise. **Recommendation**: keep excluded.

3. **`.yaml`/`.yml` classification**: Currently in `doc_extensions` (hardcoded). Move to `included.json` `doc` list? Or `code` list? **Recommendation**: Add to `doc` list in `included.json`. But note: current `excluded_files` doesn't exclude `*.yaml`/`*.yml`, so they'd be processed.
