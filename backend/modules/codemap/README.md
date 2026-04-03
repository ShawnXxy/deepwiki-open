# CodeMap Module

Static, interactive code relationship visualization for DeepWiki. Extracts **file dependencies**, **function call graphs**, and **class hierarchies** from source code using tree-sitter AST parsing — no LLM calls, no embeddings, zero token cost.

> **See also:** [CodeTrace](../codetrace/PLAN.md) — a separate, LLM-powered
> query-driven code trace feature that generates structured explanations of
> code execution flows. CodeMap is for static structure; CodeTrace is for
> answering "how does X work?" questions.

## How It Works

```
Repository (cloned to disk)
        │
        ▼
┌─────────────────────┐
│   graph_builder.py   │  Walks repo files in parallel
│                     │  (ThreadPoolExecutor)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    analyzer.py       │  Parses each file with tree-sitter
│                     │  Extracts: definitions, calls,
│                     │  imports, inheritance
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Reference Resolver  │  Resolves cross-file edges:
│  (graph_builder.py)  │  import paths → file nodes
│                     │  call names → function nodes
│                     │  class names → class nodes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     cache.py         │  Saves to ~/.adalflow/codemap/
│                     │  or Azure Blob (codemap/ prefix)
└─────────────────────┘
```

## Architecture

| File | Purpose |
|------|---------|
| `models.py` | Pydantic data models: `SymbolNode`, `SymbolEdge`, `CodeMapData`, `CodeMapMetadata` |
| `analyzer.py` | Per-file AST extraction using tree-sitter. One extractor per language. |
| `graph_builder.py` | Walks repo, runs analyzers in parallel, resolves cross-file references, deduplicates edges |
| `cache.py` | Read/write/delete codemap JSON (local disk or Azure Blob Storage) |
| `routes.py` | FastAPI endpoint: `GET /api/codemap?owner=...&repo=...&branch=...` |
| `__init__.py` | Re-exports public API: `build_codemap`, `CodeMapData`, `SymbolNode`, `SymbolEdge` |

## Data Model

### SymbolNode (graph vertex)

```python
SymbolNode(
    id="backend/app.py:15:health_check",   # Stable ID: file_path:line:name
    name="health_check",                    # Symbol name
    kind="function",                        # "file" | "function" | "method" | "class" | "module"
    file_path="backend/app.py",             # Relative path from repo root
    start_line=15,                          # 1-based line number
    end_line=22,
    parent_id="backend/app.py:10:App",      # For methods: containing class ID (None for top-level)
    language="Python",                      # Display language name
    signature="health_check()",             # Brief signature for tooltips
)
```

- **File nodes** use `file_path` as their ID (e.g., `"backend/app.py"`)
- **Symbol nodes** use `file_path:line:name` format for stable, deterministic IDs

### SymbolEdge (graph edge)

```python
SymbolEdge(
    source_id="backend/app.py:15:health_check",
    target_id="backend/utils.py:8:format_response",
    kind="calls",    # "imports" | "calls" | "inherits" | "implements"
)
```

### CodeMapData (root container)

```python
CodeMapData(
    nodes=[...],      # All SymbolNode instances
    edges=[...],      # All SymbolEdge instances (deduplicated)
    metadata=CodeMapMetadata(
        owner="org", repo="myrepo", repo_type="azuredevops",
        branch="main", commit_hash="abc123",
        generated_at="2026-04-02T10:30:00+00:00",
        total_files=115, total_symbols=507, total_edges=869,
        language_stats={"Python": 63, "TypeScript": 51, "JavaScript": 1},
    ),
)
```

## Supported Languages

| Language | Extensions | Definitions | Calls | Imports | Inheritance |
|----------|-----------|-------------|-------|---------|-------------|
| **Python** | `.py` | `def`, `class` (+ async, decorated) | `foo()`, `obj.method()` | `import X`, `from X import Y` | `class Foo(Bar, Mixin)` |
| **JavaScript** | `.js`, `.jsx` | `function`, `class`, arrow functions, `const f = () => {}` | `foo()`, `obj.method()` | `import { X } from './Y'` | `class Foo extends Bar` |
| **TypeScript** | `.ts`, `.tsx` | Same as JS + `interface`, `type` declarations | Same as JS | Same as JS | `extends`, `implements` |
| **C** | `.c`, `.h` | Functions, `struct`, `typedef` | `foo()`, `ptr->func()` | `#include "file.h"` (local only, system headers skipped) | — |
| **C++** | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` | `class`, `struct`, functions, methods, namespaces | `foo()`, `obj.method()` | `#include "file.h"` (local only) | `class Foo : public Bar` |
| **Java** | `.java` | `class`, `interface`, methods, constructors | `method()` invocations | `import com.example.X` | `extends`, `implements` |
| **Go** | `.go` | `func`, `type struct/interface`, methods with receivers | `foo()`, `pkg.Func()` | `import "path"` | Embedded structs (composition) |
| **C#** | `.cs` | `class`, `interface`, `struct`, methods, constructors | `Method()`, `obj.Method()` | `using Namespace` | `class Foo : Bar, IInterface` |

Files with unsupported extensions are included as **file-level nodes** (no internal symbols) — they still appear in the graph if referenced by import edges from supported files.

## Reference Resolution

Raw AST extraction produces **unresolved edges** (e.g., `calls: "foo"`, `imports: "backend.modules.wiki"`). The graph builder resolves these to actual node IDs:

### Import Resolution

| Language | Input | Resolution Strategy |
|----------|-------|-------------------|
| Python | `backend.modules.wiki` | Try `backend/modules/wiki.py`, then `backend/modules/wiki/__init__.py` |
| JS/TS | `./utils/helper` | Try `utils/helper.ts`, `.tsx`, `.js`, `.jsx`, `/index.ts`, etc. |
| Java | `com.example.MyClass` | Try `com/example/MyClass.java` |
| Go | `"github.com/owner/repo/pkg/utils"` | Match against repo file paths |

### Call Resolution

1. Build a symbol index: `Dict[name, List[node_id]]`
2. If unique match → resolve directly
3. If ambiguous → prefer same-file match, then same-directory match
4. If no match → skip (external library call, not in repo)

### Inheritance Resolution

Same lookup as call resolution, but restricted to `kind="class"` nodes.

## Storage

Codemap files are stored **separately from wiki cache** — in a dedicated folder:

```
~/.adalflow/
├── wikicache/    ← Wiki JSON (UNCHANGED, never mixed)
├── codemap/      ← CodeMap graph JSON (NEW)
│   └── codemap_{repo_type}_{owner}_{repo}_{branch}.json
├── repos/        ← Cloned repos
└── vectors/      ← Embedding vectors
```

**Azure Blob Storage:** Uses `codemap/` prefix (distinct from `wikicache/`).

## Pipeline Integration

CodeMap is an **optional, non-blocking** step in the code processor pipeline:

```
step_clone()
    ↓
step_build_codemap()   ← NEW (skippable via --skip-codemap)
    ↓                     Failure is non-fatal: logs warning, continues
step_embed()            ← Unchanged
    ↓
step_generate_wiki()    ← Unchanged
    ↓
step_save_wiki()        ← Unchanged
```

- **No LLM calls** — pure AST parsing, zero token cost
- **No embeddings** — does not use or depend on the vector pipeline
- **Non-fatal** — if codemap fails, wiki generation proceeds normally
- **Fast** — ~0.7s for 115 files (parallel analysis with ThreadPoolExecutor)

### CLI Usage

```bash
# Normal run (includes codemap)
python -m backend.processor.code_processor --repo=URL --branch=main --mode=local

# Skip codemap generation
python -m backend.processor.code_processor --repo=URL --branch=main --mode=local --skip-codemap
```

## Frontend

The codemap renders as an interactive graph in a **"Code Map" tab** on the wiki viewer page (`src/app/[owner]/[repo]/page.tsx`), alongside the existing "Wiki" tab.

| Component | File | Purpose |
|-----------|------|---------|
| `CodeMap.tsx` | `src/components/CodeMap.tsx` | React Flow graph with custom nodes, dagre layout |
| `useCodeMap.ts` | `src/hooks/useCodeMap.ts` | Data fetching hook |
| `codemap_cache/route.ts` | `src/app/api/codemap_cache/route.ts` | Next.js API route |
| `codemap.ts` | `src/types/codemap.ts` | TypeScript interfaces |

### View Modes

| Mode | Shows | Layout |
|------|-------|--------|
| **Files** | File nodes + import edges only | Top-down (TB) |
| **Symbols** | All nodes + all edges | Left-right (LR) |
| **Classes** | Class/file nodes + inheritance edges | Left-right (LR) |

### Node Types

| Node | Icon | Border Color |
|------|------|-------------|
| File | 📄 | Blue |
| Class | 🏛️ | Purple |
| Function | ⚡ | Green |
| Method | 🔧 | Gray |

### Edge Types

| Edge | Style | Color |
|------|-------|-------|
| Imports | Dashed arrow | Blue |
| Calls | Solid animated arrow | Gray |
| Inherits | Solid arrow | Green |
| Implements | Dashed arrow | Purple |

### Navigation

- **Code Map → Wiki**: Click any file node → navigates to the wiki page that references that file
- **Search**: Filter nodes by name in real-time (non-matching nodes dim to 20% opacity)

## Backward Compatibility

| Concern | Guarantee |
|---------|-----------|
| Existing wiki caches | Untouched. `WikiCacheData` model not modified. |
| Pipeline without codemap | Works identically with `--skip-codemap`. |
| Frontend without codemap data | Code Map tab shows "not yet generated" message. |
| Docker volumes | `~/.adalflow/codemap/` auto-created by `paths.py`. |

## Programmatic Usage

```python
from backend.modules.codemap import build_codemap, CodeMapData

# Build graph from a local repo
data: CodeMapData = build_codemap("/path/to/repo")

# Inspect results
print(f"Files: {data.metadata.total_files}")
print(f"Symbols: {data.metadata.total_symbols}")
print(f"Edges: {data.metadata.total_edges}")

for node in data.nodes:
    if node.kind == "function":
        print(f"  {node.name} @ {node.file_path}:{node.start_line}")

for edge in data.edges:
    if edge.kind == "calls":
        print(f"  {edge.source_id} → {edge.target_id}")
```

## Limitations

- **Call resolution is best-effort**: Dynamic dispatch, monkey-patching, and calls to external libraries are not resolved.
- **No type inference**: `obj.method()` extracts `method` as the call target but cannot determine `obj`'s type to resolve which class defines `method`.
- **Unsupported languages**: Files in non-supported languages (Rust, Ruby, PHP, etc.) appear as file nodes without internal symbol detail.
- **Large repos**: For repos with 5000+ files, graph rendering may be slow. Consider using the "Files" view mode which hides individual symbols.
