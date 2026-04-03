# CodeTrace Module

AI-powered, query-driven code flow tracing for DeepWiki. Users ask a natural
language question about the codebase, and the system generates a structured
trace showing how different parts of the code connect to answer it.

> **See also:** [CodeMap](../codemap/README.md) — static AST-based symbol graph
> visualization. CodeTrace and CodeMap serve different purposes.

| | CodeMap | CodeTrace |
|---|---|---|
| **What** | Static dependency/call graph | LLM-generated code flow trace |
| **Trigger** | Opens automatically on tab | User asks a question |
| **Engine** | tree-sitter AST (free) | RAG + LLM (costs tokens) |
| **Output** | Interactive node graph | Numbered sections with code refs |
| **Route** | Tab on wiki page | `/[owner]/[repo]/codetrace` |

---

## How It Works

```
User asks: "How does authentication work?"
        │
        ▼
┌─────────────────────────┐
│  Ask Panel (Ask.tsx)     │  Mode dropdown → "Code Trace" selected
│  User submits question   │  
└────────┬────────────────┘
         │  Browser navigates to:
         │  /[owner]/[repo]/codetrace?q=How+does+authentication+work
         ▼
┌─────────────────────────┐
│  CodeTrace Page          │  3-panel layout renders
│  (codetrace/page.tsx)    │  Calls POST /api/codetrace
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Next.js API Route       │  src/app/api/codetrace/route.ts
│  POST /api/codetrace     │  Proxies to backend, normalizes keys
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  FastAPI Endpoint        │  backend/modules/codetrace/routes.py
│  POST /api/codetrace     │  Prepares RAG, calls service
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  CodeTrace Service       │  backend/modules/codetrace/service.py
│  generate_code_trace()   │
│                         │  1. RAG retrieval (existing FAISS/AI Search)
│                         │  2. Format context from retrieved chunks
│                         │  3. Build system + user prompts
│                         │  4. Call LLM (Azure OpenAI)
│                         │  5. Parse XML response → structured sections
│                         │  6. Extract source contents from RAG docs
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  CodeTraceResult         │  Returned to frontend:
│                         │  - title: overall trace description
│                         │  - sections[]: numbered flow sections
│                         │  - source_files[]: referenced files
│                         │  - source_contents{}: file → code chunks
└─────────────────────────┘
```

---

## Architecture

### Files

| File | Purpose |
|------|---------|
| `__init__.py` | Module init |
| `models.py` | Pydantic models: `CodeTraceResult`, `CodeTraceSection`, `CodeReference`, `SourceChunk`, `CodeTraceRequest` |
| `service.py` | Core logic: RAG retrieval → LLM call → XML parsing → source extraction |
| `routes.py` | FastAPI `POST /api/codetrace` endpoint |
| `PLAN.md` | Full implementation plan with design decisions and future roadmap |

### Prompt template

Lives in `backend/promptstore/code_trace.py` (follows the promptstore convention):

| Prompt | Purpose |
|--------|---------|
| `CODE_TRACE_SYSTEM_PROMPT` | Instructs LLM to analyze code and produce structured XML trace |
| `CODE_TRACE_USER_PROMPT` | Formats the user question + RAG context for the LLM |

---

## Algorithm

### Step 1: RAG Retrieval

```python
retrieved = rag.call(question, language=language)
```

Uses the **existing RAG pipeline** (same as Ask/Chat). No new embedding or
index infrastructure. Retrieves top-k relevant code chunks from FAISS (local)
or Azure AI Search (cloud).

### Step 2: Context Formatting

```python
context = format_context_text(retrieved, repo_url, commit_hash, repo_type)
```

Reuses `format_context_text()` from `backend/modules/chat/service.py`.
Formats retrieved chunks with file paths, line numbers, function/class names,
and raw code. Truncated at 100KB.

### Step 3: LLM Call

```python
system_prompt = CODE_TRACE_SYSTEM_PROMPT.format(...)
user_prompt = CODE_TRACE_USER_PROMPT.format(question=..., context=...)
response = model_client.call(api_kwargs=..., model_type=ModelType.LLM)
```

Uses the `reasoning` model deployment (same as wiki generation) for
higher-quality structured output. Temperature 0.7, max 16K tokens.

### Step 4: XML Parsing

The LLM responds with structured XML:

```xml
<code_trace>
  <title>Authentication Flow Architecture</title>

  <section id="1">
    <title>Token Validation Middleware</title>
    <motivation>Validates JWT tokens on every request</motivation>
    <details>The middleware intercepts requests and checks...</details>
    <code_ref id="1a">
      <file_path>src/middleware/auth.py</file_path>
      <start_line>42</start_line>
      <end_line>67</end_line>
      <annotation>JWT validation entry point</annotation>
    </code_ref>
    <connects_to>2</connects_to>
  </section>

  <section id="2">
    <title>User Session Management</title>
    ...
  </section>
</code_trace>
```

Parsed by `_parse_trace_xml()` into `CodeTraceResult` with fallback
handling for malformed XML (returns raw text as single section).

### Step 5: Source Content Extraction

```python
source_contents = _extract_source_contents(retrieved)
```

Extracts raw source code from the RAG-retrieved documents (not from the LLM
response). Groups chunks by file path, converts line numbers from 0-based
to 1-based, maps file extensions to syntax highlighting language names.

This means the right panel shows **actual source code** from the repo,
not LLM-generated code.

---

## Data Models

### CodeTraceResult (root response)

```python
CodeTraceResult(
    query="How does authentication work?",
    title="Authentication Flow Architecture",
    sections=[...],              # List[CodeTraceSection]
    source_files=["src/auth.py", "src/middleware.py"],
    source_contents={            # Dict[file_path → List[chunk]]
        "src/auth.py": [
            {"file_path": "src/auth.py", "start_line": 1, "end_line": 45,
             "content": "def validate_token(...):\n    ...", "language": "python"},
        ],
    },
    generated_at="2026-04-03T10:30:00+00:00",
)
```

### CodeTraceSection

```python
CodeTraceSection(
    id="1",
    title="Token Validation Middleware",
    motivation="Validates JWT tokens on every request",
    details="The middleware intercepts...",  # Markdown
    code_refs=[CodeReference(...)],
    connections=["2"],  # Links to section 2
)
```

### CodeReference

```python
CodeReference(
    ref_id="1a",                    # Referenced as [1a] in details text
    file_path="src/middleware/auth.py",
    start_line=42,
    end_line=67,
    snippet="",                     # Optional inline snippet
    annotation="JWT validation entry point",
)
```

### SourceChunk

```python
SourceChunk(
    file_path="src/auth.py",
    start_line=1,           # 1-based
    end_line=45,
    content="def validate_token(...):\n    ...",
    language="python",      # For syntax highlighting
)
```

---

## Frontend

### Entry Point: Mode Dropdown in Ask Panel

The Ask panel (`src/components/Ask.tsx`) has a mode dropdown:

```
Mode: [ Chat ▼ ]
       ┌──────────────┐
       │ ● Chat       │  Default — normal Q&A
       │ ○ Deep Research │  Multi-turn investigation
       │ ○ Code Trace │  Opens codetrace page
       └──────────────┘
```

When "Code Trace" is selected and user submits, the browser navigates to:
```
/[owner]/[repo]/codetrace?q=<question>&repo_type=azuredevops&branch=main&repo_url=...
```

### Page Route

`src/app/[owner]/[repo]/codetrace/page.tsx` — standalone page (NOT a tab).

### 3-Panel Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Wiki    owner/repo    Code Trace               │ Header
├────────────────────────────────────┬─────────────────────────────┤
│                                    │                             │
│  Trace Sections (60%)              │  Source Files (40%)         │
│  ┌─ 1 ─────────────────────────┐  │  [file1.py] [file2.py] ... │
│  │ Section title               │  │  ┌─────────────────────┐   │
│  │ Motivation: ...             │  │  │  42: def validate()  │   │
│  │ [1a file.py:42] ───────────────│──│► 43:   token = ...  │   │
│  │ Details: ...                │  │  │  44:   if valid:     │   │
│  └─────────────┬───────────────┘  │  │  ...                │   │
│                ▼                   │  │                     │   │
│  ┌─ 2 ─────────────────────────┐  │  └─────────────────────┘   │
│  │ Next section                │  │                             │
│  └─────────────────────────────┘  │                             │
├────────────────────────────────────┴─────────────────────────────┤
│  [Ask follow-up...]                                    [Trace]  │
└──────────────────────────────────────────────────────────────────┘
```

### Interaction Flow

1. **Click a code ref box** (e.g., `[1a] auth.py:42`) on the left panel
2. Right panel **switches to that file's tab** and **scrolls to the line**
3. Referenced lines get **yellow highlight** background
4. **Follow-up questions** in the bottom chat bar regenerate the trace

### Frontend Files

| File | Purpose |
|------|---------|
| `src/app/[owner]/[repo]/codetrace/page.tsx` | Page route + 3-panel layout |
| `src/hooks/useCodeTrace.ts` | POST hook — calls `/api/codetrace` |
| `src/types/codetrace.ts` | TypeScript interfaces |
| `src/app/api/codetrace/route.ts` | Next.js API proxy (snake→camelCase) |
| `src/components/Ask.tsx` | Mode dropdown (Chat / Deep Research / Code Trace) |

---

## Dependencies

- **No new Python packages** — reuses existing RAG, AzureAI client, embedder
- **No new npm packages** — reuses existing React icons, markdown components
- **No vector/index schema changes** — uses existing chunk metadata as-is
- **LLM cost**: ~2000–5000 tokens per trace query

---

## Key Design Decisions

1. **Separate page route** — Code trace is a deep-dive tool with its own layout,
   not a tab on the wiki viewer.

2. **Reuses existing RAG** — No new embedding pipeline. Same FAISS/AI Search
   retriever used by Ask/Chat. Only the prompt template is different.

3. **Source code from RAG, not LLM** — The right panel shows actual code from
   the repository (extracted from RAG chunks), not LLM-generated code.
   The LLM only provides the structural analysis and annotations.

4. **XML output format** — Same parsing approach as `wiki_structure.py`.
   Fallback to raw text if XML parsing fails.

5. **Mode dropdown replaces toggle** — The Deep Research toggle was replaced
   with a 3-option dropdown (Chat / Deep Research / Code Trace) for cleaner UX.

6. **No vector schema changes for MVP** — Existing chunk metadata already has
   `file_path`, `start_line`, `end_line`, `functions`, `classes`. Text search on
   AI Search `content` field works for cloud mode. See PLAN.md for future
   roadmap on adding dedicated index fields.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| RAG returns no documents | Returns "No relevant code found" with suggestion to rephrase |
| LLM returns empty response | Returns "Generation failed" |
| XML parse fails | Falls back to raw LLM text as single section |
| Backend unavailable | Frontend shows error message in left panel |
| Network error | Hook sets `error` state, displayed in UI |

---

## Backward Compatibility

- **Existing Ask/Chat** — `deepResearch` boolean is derived from `chatMode === 'deepresearch'`.
  All existing deep research logic continues to work identically.
- **Existing wiki cache** — No changes to WikiCacheData model.
- **Existing vectors** — No changes to embedding pipeline or storage format.
- **Existing routes** — No changes to existing API endpoints.

---

## Roadmap

### Phase 1 — MVP (Current) ✅

- [x] Backend: models, service, routes, prompt template
- [x] Frontend: codetrace page with 3-panel layout
- [x] Ask panel: mode dropdown (Chat / Deep Research / Code Trace)
- [x] Source file viewer with tabs + line highlighting
- [x] RAG-based retrieval (reuses existing pipeline)
- [x] XML structured output from LLM
- [x] Error handling + fallbacks

### Phase 2 — Streaming & Follow-up

- [ ] **WebSocket streaming** — Stream trace sections incrementally as the LLM
  generates them, instead of waiting for the full response. Show sections
  appearing one by one (similar to how Deep Research shows iterations).
- [ ] **Conversational follow-up** — Maintain conversation context so users can
  ask "What about error handling in that flow?" and get a refined trace that
  builds on the previous one rather than starting fresh.
- [ ] **Trace history** — Cache generated traces (like wiki cache) so revisiting
  the same URL shows the previous result instantly without re-calling the LLM.

### Phase 3 — Enhanced Source Viewer

- [ ] **Full file loading** — Load complete source files from the cloned repo
  (via a new API endpoint) instead of only showing RAG-retrieved chunks.
  This gives the user full file context, not just the relevant snippets.
- [ ] **Syntax highlighting** — Integrate `react-syntax-highlighter` or a
  similar library in the right panel for proper language-aware coloring.
- [ ] **Multi-line highlight ranges** — Support highlighting multiple
  non-contiguous line ranges per file (one per code ref).
- [ ] **Click-to-navigate** — Click a file path in the trace section to open
  that file in the source viewer AND link to the original source on GitHub/ADO.

### Phase 4 — AI Search Index Enhancement (Cloud Mode)

- [ ] **Add metadata fields to AI Search schema** — Add `start_line`,
  `end_line`, `section_type`, `functions_text`, `classes_text` as filterable
  fields in `code_index_schema.json`.
- [ ] **Update push_documents()** — Include chunk metadata when pushing to
  Azure AI Search index.
- [ ] **Symbol definition lookup** — Fast filtered query:
  `section_type eq 'function' AND functions_text eq 'process_data'`
  instead of full-text search on `content`.
- [ ] **Requires re-indexing** existing repos to populate the new fields.
  New repos get them automatically.

### Phase 5 — Advanced Trace Features

- [ ] **Recursive trace** — When a trace section references a function,
  offer a "Trace deeper" button that generates a sub-trace for that
  specific function's internal logic.
- [ ] **Trace comparison** — Compare traces for the same question across
  different branches (useful for understanding how a refactor changed
  the code flow).
- [ ] **Export** — Export trace as Markdown, PDF, or shareable link with
  embedded results (not just the query URL).
- [ ] **Trace from CodeMap** — Click a node in the CodeMap graph and
  select "Trace this" to auto-generate a code trace for that symbol.

### Phase 6 — Visual Flow Diagram

- [ ] **Mermaid flow diagram** — Generate a Mermaid `flowchart TD` diagram
  from the trace sections and their connections. Render inline between
  sections using the existing Mermaid component.
- [ ] **Interactive graph view** — Alternative to the section-list view:
  render trace sections as React Flow nodes connected by edges,
  similar to CodeMap but with LLM-generated content inside each node.

---

## Known Limitations

- **Single-shot response** — Currently waits for the full LLM response before
  rendering. Large traces may take 10-15 seconds.
- **Source code is RAG chunks only** — The right panel shows only the chunks
  retrieved by RAG, not complete files. Some context may be missing.
- **No conversation memory** — Each trace is independent. Follow-up questions
  do not build on previous traces.
- **Cloud mode symbol lookup** — AI Search index lacks dedicated metadata
  fields. Text search on `content` works but is less precise than filtered
  queries. See Phase 4 roadmap.
- **No caching** — Traces are not persisted. Refreshing the page re-generates
  the trace (costs LLM tokens).
