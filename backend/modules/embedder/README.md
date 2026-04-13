# Embedder Module

Chunking, embedding, and retrieval engine for DeepWiki.

## Responsibility

Takes code repositories and produces searchable vector embeddings:
1. **Read** files from cloned repos (with inclusion/exclusion filters)
2. **Split** code at logical boundaries (function/class definitions)
3. **Enrich** chunks with structural metadata (file path, language, section type)
4. **Embed** via Azure OpenAI `text-embedding-3-large` (3072 dimensions, batch mode)
5. **Store** as JSON vector files (one file per chunk, local disk or Azure Blob)
6. **Retrieve** via FAISS semantic search (with file-priority filtering)

## Files

| File | Purpose |
|------|---------|
| `retriever.py` | Main `RAG` class — FAISS search + file-filtered retrieval |
| `indexer.py` | `DatabaseManager` — orchestrates load/create/migrate pipeline |
| `document.py` | Read files, split, embed, save as JSON vectors |
| `code_splitter.py` | Boundary-aware splitting (14 regex patterns, 8+ languages) |
| `tokenizer.py` | Token counting (tiktoken `cl100k_base`) + safe file reading |
| `response.py` | `RAGAnswer` dataclass (rationale + answer) |
| `memory.py` | Conversation history tracking (dialog turns with UUIDs) |

## How It Works

### Document Reading (`document.py`)

`read_all_documents()` walks the cloned repo directory and reads code + documentation files:

- **Code files:** `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.jsx`, `.tsx`, `.html`, `.css`, `.php`, `.swift`, `.cs`
- **Doc files:** `.md`, `.txt`, `.rst`, `.json`, `.yaml`, `.yml`

File filtering uses the `FileFilter` type (from `backend/types/processor_types.py`):
- **Exclusion mode** (default): Loads filters from `backend/config/excluded.json` — excludes `.git`, `node_modules`, `__pycache__`, etc. Also respects the repo's root `.gitignore`.
- **Inclusion mode**: When `included_dirs` or `included_files` are provided, only those are processed.
- **Size limit**: 10 MB per file (configurable via `max_file_size_mb`)

Each document gets metadata: `file_path`, `type` (extension), `url` (commit-pinned link).

### Code-Aware Splitting (`code_splitter.py`)

Instead of naive token-based splitting, code files are split at **logical boundaries**:

```
┌─────────────────────────────┐
│  find_logical_boundaries()  │  14 regex patterns detect:
│                             │  - Python: def, class, async def
│                             │  - JS/TS: function, export, const =
│                             │  - Java/C#: public/private methods
│                             │  - Go: func
│                             │  - Rust: fn, impl, pub
│                             │  - Ruby: def, class, module
│                             │  - PHP: function, class
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  split_code_at_boundaries() │  Merge lines into ~2000-token chunks
│                             │  Never split mid-function
│                             │  Oversized sections: split at nested block
│                             │  boundaries (inner defs, indentation drops)
│                             │  Falls back to blank-line groups
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  build_enriched_chunk_text()│  Prepend structural context:
│                             │  [File: src/main.py | Language: python
│                             │   | Section: function
│                             │   | Functions: process_data, handle_request]
└─────────────────────────────┘
```

Documentation files (`.md`, `.txt`) use heading/paragraph boundary detection instead.

### Embedding & Storage (`document.py` → `vector_storage`)

`transform_documents_and_save_as_json()` processes chunks through a pipeline:

1. Code-aware splitting produces enriched chunks (as above)
2. `SafeEmbedder` calls Azure OpenAI Embeddings API in batches (configured by `embedder.json`)
3. Each chunk is saved as an individual JSON file in `~/.adalflow/vectors/{repo}_{branch}/`

```
vectors/myorg_myrepo_main/
├── src/backend/main_001.json    ← chunk 1 of main.py
├── src/backend/main_002.json    ← chunk 2 of main.py
├── src/utils/helper_001.json
└── README_001.json
```

Each JSON file contains: `text`, `vector` (3072-dim float array), `meta_data`, `file_path`, `chunk_index`, `total_chunks`.

### Database Management (`indexer.py`)

`DatabaseManager.prepare_db_index()` handles storage with backward compatibility:

| Priority | Format | Location | Status |
|----------|--------|----------|--------|
| 1 | JSON vectors | `vectors/{repo}_{branch}/` | **Current** (preferred) |
| 2 | Pickle database | `databases/{repo}_{branch}.pkl` | **Legacy** (read-only support) |

**Zero-downtime reprocessing** (`force_reprocess=True`):
1. Snapshot existing vector files
2. Embed and overwrite in-place (old files remain readable during processing)
3. Delete orphan files after completion (files from deleted/renamed source files)

This avoids the downtime window that deleting-all-then-recreating would cause.

### Retrieval (`retriever.py`)

The `RAG` class builds a FAISS index from loaded documents and provides two retrieval modes:

**`call(query)`** — Standard semantic search:
- FAISS top-k nearest neighbor search using document embeddings
- Returns ranked list of relevant chunks

**`call_with_file_filter(query, file_paths, top_k)`** — File-priority retrieval:
1. Collect chunks from declared `file_paths` via pre-built index (capped at 80 per call)
2. Run FAISS semantic search for supplementary context
3. Merge: file chunks first (priority), then semantic chunks (deduplicated by `(file_path, chunk_index)` tuple)

This ensures wiki pages always reference their declared source files while also discovering related context.
The deduplication uses content-based keys rather than object identity, so it works correctly
for both local FAISS and cloud AI Search modes.

### Conversation Memory (`memory.py`)

The `Memory` class tracks dialog turns for multi-turn chat:
- Each turn stores a `UserQuery` and `AssistantResponse` with UUID tracking
- `CustomConversation` wraps adalflow's conversation with safe error handling
- Used by the chat module for context-aware follow-up questions

## Usage

```python
from backend.modules.embedder import RAG

# Initialize and prepare
rag = RAG(provider='azure')
rag.prepare_retriever(
    repo_url_or_path='https://dev.azure.com/org/proj/_git/repo',
    type='azuredevops',
    access_token='pat-token',
    branch='main',
    force_reprocess=True,  # Zero-downtime re-embedding
)

# Standard retrieval
results = rag.call("How does authentication work?")

# File-priority retrieval (for wiki page generation)
results = rag.call_with_file_filter(
    query="Authentication Module",
    file_paths=["src/auth/handler.py", "src/auth/middleware.py"],
    top_k=40,
)
```

## Dependencies

- **Invokes:** `repository/git_ops` (clone repos), `clients/embedding_client` (Azure OpenAI embeddings), `clients/vector_storage` (JSON store)
- **Invoked by:** `chat/` (Ask/Chat Q&A), `processor/` (CLI wiki generation)
