# CodeMap Feature — Implementation Plan

## Overview

Add an interactive code map visualization that shows **file dependencies**, **function
call graphs**, and **class hierarchies** extracted via **tree-sitter AST parsing**.

- **Backend** generates a structured graph (nodes + edges) stored in a **dedicated
  `~/.adalflow/codemap/` folder** (separate from `wikicache/`).
- **Frontend** renders it with **React Flow** in a new "Code Map" tab on the wiki
  viewer page.
- **Fully additive** — existing wiki cache files and generation pipeline are untouched.
  Old wikis continue to work. CodeMap is an optional, independently generated artifact.

---

## Storage Layout

```
~/.adalflow/
├── wikicache/          ← Existing wiki JSON cache (UNCHANGED)
├── codemap/            ← NEW: CodeMap graph JSON files
│   └── codemap_{repo_type}_{owner}_{repo}_{branch}.json
├── repos/              ← Cloned git repositories (UNCHANGED)
├── vectors/            ← Embedding vector JSON chunks (UNCHANGED)
└── embedding_cache/    ← Embedding API cache (UNCHANGED)
```

**Azure Blob Storage** (cloud mode):
```
codemap/codemap_{repo_type}_{owner}_{repo}_{branch}.json
```

The `codemap/` prefix is distinct from the `wikicache/` prefix used for wiki files.

---

## Phase 1: Backend — Data Models

### File: `backend/modules/codemap/models.py`

```python
from typing import List, Optional, Dict
from pydantic import BaseModel


class SymbolNode(BaseModel):
    """A code symbol (file, class, function, method) as a graph node."""
    id: str                          # Stable ID: "file_path:line:name"
    name: str                        # Symbol name (e.g., "build_codemap")
    kind: str                        # "file" | "class" | "function" | "method" | "module"
    file_path: str                   # Relative path from repo root
    start_line: Optional[int] = None # Line number (1-based), None for file nodes
    end_line: Optional[int] = None
    parent_id: Optional[str] = None  # For methods: their containing class node ID

    # Display metadata
    language: Optional[str] = None   # "Python", "TypeScript", etc.
    signature: Optional[str] = None  # Brief signature preview for tooltip


class SymbolEdge(BaseModel):
    """A relationship between two symbols."""
    source_id: str             # ID of the source SymbolNode
    target_id: str             # ID of the target SymbolNode
    kind: str                  # "imports" | "calls" | "inherits" | "implements"


class CodeMapMetadata(BaseModel):
    """Metadata about the codemap generation."""
    owner: str
    repo: str
    repo_type: str
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    generated_at: Optional[str] = None    # ISO 8601 timestamp
    total_files: int = 0
    total_symbols: int = 0
    total_edges: int = 0
    language_stats: Dict[str, int] = {}   # {"Python": 42, "TypeScript": 18, ...}


class CodeMapData(BaseModel):
    """Root model for the complete code map graph."""
    nodes: List[SymbolNode]
    edges: List[SymbolEdge]
    metadata: CodeMapMetadata
```

**Key design decisions:**
- `id` format is `file_path:line:name` for symbols, plain `file_path` for file nodes
  — ensures stable, deterministic IDs across regenerations.
- `parent_id` enables grouping methods under their class in the frontend.
- `language_stats` lets the frontend show a language breakdown badge.

---

## Phase 2: Backend — AST Analysis Engine

### File: `backend/modules/codemap/analyzer.py`

Core function: `analyze_file(file_path, content, language) → (List[SymbolNode], List[SymbolEdge])`

#### tree-sitter Extraction Per Language

| Language | Definitions | Calls | Imports | Inheritance |
|----------|------------|-------|---------|-------------|
| **Python** | `function_definition`, `class_definition` | `call` (attribute + identifier) | `import_statement`, `import_from_statement` | `argument_list` in class def |
| **JavaScript** | `function_declaration`, `class_declaration`, `arrow_function`, `variable_declarator` | `call_expression` | `import_statement` | `class_heritage` |
| **TypeScript** | Same as JS + `interface_declaration`, `type_alias_declaration` | `call_expression` | `import_statement` | `class_heritage`, `extends_clause`, `implements_clause` |
| **Java** | `method_declaration`, `class_declaration`, `interface_declaration` | `method_invocation` | `import_declaration` | `superclass`, `super_interfaces` |
| **Go** | `function_declaration`, `method_declaration`, `type_declaration` (struct/interface) | `call_expression` | `import_declaration` | Embedded struct fields (composition) |
| **C#** | `method_declaration`, `class_declaration`, `interface_declaration` | `invocation_expression` | `using_directive` | `base_list` |

#### Extraction Algorithm

```
for each tree-sitter node in AST:
    if node.type in definition_types[language]:
        → Create SymbolNode(kind, name, file_path, start_line, end_line)
        → If inside a class: set parent_id to the class node's ID

    if node.type in import_types[language]:
        → Create SymbolEdge(kind="imports", source=current_file, target=imported_path)

    if node.type in call_types[language]:
        → Extract callee name
        → Create SymbolEdge(kind="calls", source=enclosing_function, target=callee_name)
        → (Target resolution deferred to graph_builder phase)

    if node.type in inheritance_types[language]:
        → Extract parent class/interface name
        → Create SymbolEdge(kind="inherits", source=current_class, target=parent_name)
```

#### tree-sitter Parser Initialization

```python
# One parser instance per language, lazily initialized
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_go
import tree_sitter_c_sharp
from tree_sitter import Language, Parser

LANGUAGES = {
    'py': Language(tree_sitter_python.language()),
    'js': Language(tree_sitter_javascript.language()),
    'jsx': Language(tree_sitter_javascript.language()),
    'ts': Language(tree_sitter_typescript.language_typescript()),
    'tsx': Language(tree_sitter_typescript.language_tsx()),
    'java': Language(tree_sitter_java.language()),
    'go': Language(tree_sitter_go.language()),
    'cs': Language(tree_sitter_c_sharp.language()),
}
```

#### Language Extension Mapping (reuse from code_splitter.py)

```python
# Subset of LANGUAGE_MAP from code_splitter.py for supported AST languages
SUPPORTED_EXTENSIONS = {'py', 'js', 'jsx', 'ts', 'tsx', 'java', 'go', 'cs'}
```

---

## Phase 3: Backend — Graph Builder

### File: `backend/modules/codemap/graph_builder.py`

Core function: `build_codemap(repo_path, file_filter) → CodeMapData`

#### Algorithm

```
1. Walk repo directory (reuse os.walk pattern from embedder/document.py)
   - Apply FileFilter exclusions (from backend/types/processor_types.py)
   - Only process files with extensions in SUPPORTED_EXTENSIONS
   - For unsupported extensions: create file-level node only (no symbol detail)

2. For each supported file:
   - Read content
   - Call analyze_file(file_path, content, language)
   - Collect nodes and edges

3. Resolve cross-file references:
   a. Import resolution:
      - Python: "from backend.modules.wiki import cache" → "backend/modules/wiki/cache.py"
      - JS/TS: "import { foo } from './utils/helper'" → "src/utils/helper.ts"
      - Java: "import com.example.MyClass" → "com/example/MyClass.java"
      - Go: "import \"github.com/owner/repo/pkg/utils\"" → "pkg/utils/"
      - C#: using directives → namespace-to-file mapping (best effort)
   
   b. Call resolution:
      - Build symbol index: Dict[name, List[SymbolNode]]
      - For each "calls" edge with unresolved target:
        → Look up callee name in symbol index
        → If unique match: resolve edge target to that node's ID
        → If ambiguous: prefer same-file match, then same-package match
        → If no match: edge points to a "phantom" external node (excluded from graph)
   
   c. Inheritance resolution:
      - Same lookup strategy as call resolution
      - Match class name → class SymbolNode ID

4. Deduplicate edges (same source + target + kind)

5. Build metadata:
   - Count files per language
   - Record commit_hash from .git/HEAD
   - Timestamp

6. Return CodeMapData(nodes, edges, metadata)
```

#### Performance Strategy

For large repos (5000+ files):
- Use `ThreadPoolExecutor(max_workers=os.cpu_count())` for parallel file analysis
- Each file analysis is independent — ideal for parallelization
- Graph resolution (step 3) is single-threaded (needs full symbol index)

---

## Phase 4: Backend — Cache & Storage

### File: `backend/modules/codemap/cache.py`

```python
from backend.paths import get_codemap_path  # NEW path function

CODEMAP_CACHE_DIR = get_codemap_path()      # ~/.adalflow/codemap/
CODEMAP_BLOB_PREFIX = "codemap"             # Blob: codemap/codemap_*.json
```

#### Functions

| Function | Description |
|----------|-------------|
| `get_codemap_filename(owner, repo, repo_type, branch)` | `codemap_{repo_type}_{owner}_{repo}_{branch}.json` |
| `get_codemap_path(owner, repo, repo_type, branch)` | Full local path |
| `get_codemap_blob_path(owner, repo, repo_type, branch)` | Blob storage path |
| `read_codemap_cache(owner, repo, repo_type, branch) → Optional[CodeMapData]` | Read from blob or local |
| `save_codemap_cache(data: CodeMapData) → bool` | Save to blob or local |
| `delete_codemap_cache(owner, repo, repo_type, branch) → bool` | Delete cache file |

Follows the exact same pattern as `backend/modules/wiki/cache.py` — tries blob first
when configured, falls back to local disk.

### File: `backend/paths.py` — Add new path function

```python
def get_codemap_path() -> str:
    """Get the path for codemap graph storage."""
    path = os.path.join(get_adalflow_root_path(), "codemap")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
```

---

## Phase 5: Backend — Routes & Pipeline Integration

### File: `backend/modules/codemap/routes.py`

```python
router = APIRouter(prefix="/api", tags=["codemap"])

@router.get("/codemap")
async def get_codemap(owner: str, repo: str, repo_type: str, branch: str = None):
    """Read codemap from cache."""
    data = await read_codemap_cache(owner, repo, repo_type, branch)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Codemap not found"})
    return data.model_dump()
```

### File: `backend/app.py` — Register router

```python
from backend.modules.codemap.routes import router as codemap_router
app.include_router(codemap_router)
```

### File: `backend/processor/code_processor.py` — Add pipeline step

```python
def step_build_codemap(repo_path, owner, repo, repo_type, branch, file_filter):
    """Build codemap graph from cloned repository."""
    from backend.modules.codemap.graph_builder import build_codemap
    from backend.modules.codemap.cache import save_codemap_cache

    codemap = build_codemap(repo_path, file_filter)
    codemap.metadata.owner = owner
    codemap.metadata.repo = repo
    codemap.metadata.repo_type = repo_type
    codemap.metadata.branch = branch

    save_codemap_cache(codemap)
    return codemap
```

Insert into pipeline after `step_clone()`, before `step_embed()`:
```
resolve_auth() → step_clone() → step_build_codemap() → step_embed() → ...
```

Add CLI flag: `--skip-codemap` (default: codemap enabled)

### File: `backend/processor/wiki_generator.py` — Server-side integration

Add `step_build_codemap()` call within `generate_wiki()` after clone completes.

---

## Phase 6: Frontend — TypeScript Types

### File: `src/types/codemap.ts`

```typescript
export interface CodeMapNode {
  id: string;
  name: string;
  kind: 'file' | 'class' | 'function' | 'method' | 'module';
  filePath: string;
  startLine?: number;
  endLine?: number;
  parentId?: string;
  language?: string;
  signature?: string;
}

export interface CodeMapEdge {
  sourceId: string;
  targetId: string;
  kind: 'imports' | 'calls' | 'inherits' | 'implements';
}

export interface CodeMapMetadata {
  owner: string;
  repo: string;
  repoType: string;
  branch?: string;
  commitHash?: string;
  generatedAt?: string;
  totalFiles: number;
  totalSymbols: number;
  totalEdges: number;
  languageStats: Record<string, number>;
}

export interface CodeMapData {
  nodes: CodeMapNode[];
  edges: CodeMapEdge[];
  metadata: CodeMapMetadata;
}
```

---

## Phase 7: Frontend — API Route

### File: `src/app/api/codemap_cache/route.ts`

- `GET /api/codemap_cache?owner=...&repo=...&repo_type=...&branch=...`
- Fallback chain (same pattern as `wiki_cache/route.ts`):
  1. Try backend FastAPI: `GET http://localhost:8001/api/codemap?...`
  2. Try local disk: `~/.adalflow/codemap/codemap_{repo_type}_{owner}_{repo}_{branch}.json`
- Returns `CodeMapData` JSON or 404

---

## Phase 8: Frontend — React Flow Visualization

### File: `src/components/CodeMap.tsx`

#### Dependencies

```json
{
  "@xyflow/react": "^12.x",
  "dagre": "^0.8.5"
}
```

(`yarn add @xyflow/react dagre`)

#### Custom Node Types

| Node Type | Visual | Content |
|-----------|--------|---------|
| `FileNode` | 📄 File icon, rounded rect | File name, language badge, symbol count |
| `ClassNode` | 🏛️ Blue border, rect | Class name, method count |
| `FunctionNode` | ⚡ Green border, small rect | Function name, signature preview |
| `MethodNode` | 🔧 Smaller, indented under class | Method name |

#### Edge Styles

| Edge Kind | Style | Color |
|-----------|-------|-------|
| `imports` | Dashed line, arrow | Blue (#3B82F6) |
| `calls` | Solid line, arrow | Gray (#6B7280) |
| `inherits` | Solid thick line, diamond | Green (#10B981) |
| `implements` | Dashed thick line, diamond | Purple (#8B5CF6) |

#### Layout Algorithm (dagre)

```typescript
import dagre from 'dagre';

function layoutGraph(nodes: CodeMapNode[], edges: CodeMapEdge[], direction: 'TB' | 'LR') {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: direction, nodesep: 60, ranksep: 80 });
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach(node => {
    const width = node.kind === 'file' ? 200 : 160;
    const height = node.kind === 'file' ? 60 : 40;
    g.setNode(node.id, { width, height });
  });

  edges.forEach(edge => {
    g.setEdge(edge.sourceId, edge.targetId);
  });

  dagre.layout(g);
  // Map dagre positions back to React Flow nodes
}
```

#### Interactive Features

1. **View Mode Toggle**: "Files" | "Symbols" | "Classes"
   - Files: Only file nodes + import edges
   - Symbols: All nodes + call edges (grouped by file)
   - Classes: Class/interface nodes + inheritance edges

2. **Search/Filter**: Text input filters nodes by name. Non-matching nodes dim (opacity: 0.2).

3. **Click-to-Focus**: Click a node → highlight all connected nodes and edges,
   dim everything else.

4. **Minimap**: React Flow `<MiniMap />` plugin in bottom-right corner.

5. **Zoom/Pan**: Built-in React Flow — scroll to zoom, drag to pan.

6. **Node Details Panel**: Click a node → sidebar shows:
   - Full file path
   - Line numbers
   - List of incoming/outgoing edges
   - Link to source file (same URL generation as wiki citations)

---

## Phase 9: Frontend — Wiki Viewer Integration

### File: `src/app/[owner]/[repo]/page.tsx`

#### Tab Switcher

Add horizontal tabs above the main content area:
```
┌──────────┬─────────────┐
│   Wiki   │  Code Map   │   ← New tab bar
├──────────┴─────────────┤
│                        │
│   (content area)       │
│                        │
└────────────────────────┘
```

- **State**: `const [activeView, setActiveView] = useState<'wiki' | 'codemap'>('wiki')`
- **Wiki tab**: Current behavior unchanged (sidebar + markdown content + Ask panel)
- **Code Map tab**: Full-width `<CodeMap />` component (sidebar and Ask panel hidden)
- **Lazy loading**: `const CodeMap = lazy(() => import('@/components/CodeMap'))`

#### Bidirectional Navigation

| Direction | Trigger | Action |
|-----------|---------|--------|
| CodeMap → Wiki | Click file node | Switch to Wiki tab, find page where `filePaths` contains clicked file, navigate to it |
| Wiki → CodeMap | "View in Code Map" button on page | Switch to CodeMap tab, find file nodes matching current page's `filePaths`, highlight them |

---

## File Manifest

### Backend — Create

| File | Purpose |
|------|---------|
| `backend/modules/codemap/__init__.py` | Module init, re-exports |
| `backend/modules/codemap/models.py` | Pydantic models: SymbolNode, SymbolEdge, CodeMapData |
| `backend/modules/codemap/analyzer.py` | tree-sitter AST extraction per language |
| `backend/modules/codemap/graph_builder.py` | Walk repo, build graph, resolve references |
| `backend/modules/codemap/cache.py` | Cache read/write/delete (local + blob) |
| `backend/modules/codemap/routes.py` | FastAPI GET /api/codemap endpoint |

### Backend — Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add: `tree-sitter>=0.24`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`, `tree-sitter-java`, `tree-sitter-go`, `tree-sitter-c-sharp` |
| `backend/paths.py` | Add `get_codemap_path()` returning `~/.adalflow/codemap/` |
| `backend/app.py` | Register `codemap_router` |
| `backend/processor/code_processor.py` | Add `step_build_codemap()` + `--skip-codemap` flag |
| `backend/processor/wiki_generator.py` | Add codemap step in `generate_wiki()` |

### Frontend — Create

| File | Purpose |
|------|---------|
| `src/types/codemap.ts` | TypeScript interfaces |
| `src/components/CodeMap.tsx` | React Flow graph visualization |
| `src/hooks/useCodeMap.ts` | Data fetching + dagre layout hook |
| `src/app/api/codemap_cache/route.ts` | Next.js API route (proxy/local read) |

### Frontend — Modify

| File | Change |
|------|--------|
| `package.json` | Add `@xyflow/react`, `dagre` |
| `src/app/[owner]/[repo]/page.tsx` | Add tab switcher, lazy-load CodeMap |

---

## Dependencies

### Python (add to pyproject.toml)

```toml
tree-sitter = ">=0.24.0"
tree-sitter-python = ">=0.23.0"
tree-sitter-javascript = ">=0.23.0"
tree-sitter-typescript = ">=0.23.0"
tree-sitter-java = ">=0.23.0"
tree-sitter-go = ">=0.23.0"
tree-sitter-c-sharp = ">=0.23.0"
```

These are individual grammar packages (~2MB each), NOT the bundled
`tree-sitter-languages` package (~100MB). Total addition: ~12MB.

### Node.js (add to package.json)

```bash
yarn add @xyflow/react dagre
```

---

## Backward Compatibility

| Concern | Guarantee |
|---------|-----------|
| Existing wiki cache files | **Untouched.** WikiCacheData model not modified. |
| Wiki generation pipeline | **Unchanged.** Codemap is a new step, not replacing any. |
| Old CLI invocations | **Work identically.** `--skip-codemap` defaults to enabled; no flag = codemap generated. |
| Frontend without codemap data | **Graceful fallback.** Code Map tab shows "not yet generated" message. |
| Docker volumes | **Compatible.** `~/.adalflow/codemap/` auto-created, Docker mounts `~/.adalflow`. |
| Azure Blob Storage | **Separate prefix.** `codemap/` prefix never collides with `wikicache/`. |

---

## Testing Plan

| Test | File | What to verify |
|------|------|----------------|
| AST extraction (Python) | `tests/test_codemap_analyzer.py` | Functions, classes, imports, calls extracted correctly |
| AST extraction (JS/TS) | `tests/test_codemap_analyzer.py` | Arrow functions, ES modules, class heritage |
| AST extraction (Java) | `tests/test_codemap_analyzer.py` | Method declarations, package imports, inheritance |
| Graph builder | `tests/test_codemap_graph.py` | Import resolution, call matching, deduplication |
| Cache CRUD | `tests/test_codemap_cache.py` | Save, read, delete in local mode |
| Integration | Manual | Run code_processor with `--mode local` → verify codemap JSON |
| Frontend render | Manual | Load in browser → nodes, edges, zoom, pan |
| Navigation | Manual | Codemap → Wiki and Wiki → Codemap navigation |
| Lint | CI | `flake8 backend/` and `npm run lint` pass |
| Docker | `test-local.ps1` | Codemap works in containerized environment |

---

## Execution Order

Recommended implementation sequence:

```
Phase 1: models.py            (no dependencies, pure types)
Phase 2: analyzer.py           (depends on models + tree-sitter)
Phase 3: graph_builder.py      (depends on analyzer)
Phase 4: cache.py + paths.py   (depends on models)
Phase 5: routes.py + app.py    (depends on cache)
   └─ code_processor.py        (depends on graph_builder + cache)
   └─ wiki_generator.py        (depends on graph_builder + cache)
Phase 6: src/types/codemap.ts  (no dependencies)
Phase 7: API route              (depends on types)
Phase 8: CodeMap.tsx + hook     (depends on types + API route)
Phase 9: page.tsx integration   (depends on CodeMap component)
```

Phases 1-5 (backend) and Phase 6 (frontend types) can begin in parallel.
