# Code Processing Pipeline

## Overview

This pipeline converts a code repository into searchable vector embeddings for RAG-based wiki generation. It uses regex-based structural analysis (no AST parsers) to split code at logical boundaries and enrich each chunk with file context before embedding.

The pipeline uses boundary-aware splitting and structural enrichment with zero LLM cost during indexing. Quality investment is made at generation time (file-path-aware retrieval, higher top_k) rather than at embedding time.

## Modules

| File | Role |
|------|------|
| `document.py` | Reads source files, orchestrates the pipeline (split → embed → save) |
| `code_splitter.py` | Boundary-aware splitting, structural metadata extraction, enrichment |
| `service.py` | Formats retrieved chunks with structural headers for LLM context |
| `wiki_page.py` | Prompt instructions for the LLM to interpret structural metadata |

## Pipeline Flow

```
read_documents()                      # Read all source files (no size limit for code)
  → split_and_enrich_documents()      # Split + enrich with structural metadata
  → prepare_embed_only_pipeline()     # Embed pre-split chunks (no re-splitting)
  → Save as JSON vector files
```

## Phase 1: Boundary-Aware Splitting

### Step 1 — Read Documents (`document.py: read_documents`)

Reads all code and documentation files from the cloned repository. Files are filtered by extension and exclusion patterns from `repo.json`.

**No file size limit for code files.** Since the code splitter handles any size by producing ~2000-token chunks, even a 135K-token file is processed (producing ~67 chunks). Documentation files are capped at ~80K tokens to skip auto-generated data files (e.g., binary vectors, test fixtures).

### Step 2 — Split at Logical Boundaries (`code_splitter.py: split_code_at_boundaries`)

Code files are split at **logical boundaries** instead of arbitrary token counts.

**Algorithm:**

1. **Find boundaries** (`find_logical_boundaries`): Scan the file line-by-line using 14 compiled regex patterns that detect function/class definitions across 8 language families (Python, JS/TS, Java/C#, Go, Rust, Ruby, PHP). Also detects blank-line separators (2+ consecutive blank lines) and decorator blocks.

2. **Group into chunks**: Walk through boundaries sequentially, accumulating lines into a chunk. When adding the next boundary section would exceed `target_tokens` (2000), finalize the current chunk and start a new one.

3. **Force-split oversized sections**: If a single boundary section exceeds `max_tokens` (2800) — e.g., a very long function — split it at line boundaries into sub-chunks of `target_tokens`.

4. **Merge tiny chunks**: If a chunk is below `min_tokens` (100), merge it into the previous chunk to avoid overly fragmented embeddings.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `target_tokens` | 2000 | Ideal chunk size |
| `max_tokens` | 2800 | Force-split if a single section exceeds this |
| `min_tokens` | 100 | Merge into neighbor if below this |

**Documentation files** use a simpler splitter (`_split_doc_text`) that splits at paragraph boundaries (blank lines) with the same target token size.

### Step 3 — Extract Structural Metadata (`code_splitter.py: extract_code_elements`)

Each chunk is analyzed with regex to extract:

- **Function names**: `def`, `function`, `func`, `fn`, Java/C# method signatures with access modifiers
- **Class names**: `class`, `struct`, `interface`, `enum`, `trait`, `impl`
- **Import statements**: First 5 `import`/`require`/`using`/`from...import` lines (truncated to 80 chars)

### Step 4 — Classify Section Type (`code_splitter.py: _detect_section_type`)

Each chunk is classified by its dominant content:

| Type | Heuristic |
|------|-----------|
| `imports` | ≥50% of lines are import/require/using statements |
| `class` | First non-blank line matches a class/struct/interface/enum pattern |
| `function` | First non-blank line matches a function definition pattern |
| `declaration` | Contains export/const/var at top level |
| `configuration` | File extension is .json/.yaml/.yml/.toml/.ini |
| `documentation` | File extension is .md/.txt/.rst |
| `code` | Default fallback |

### Step 5 — Build Enriched Embedding Text (`code_splitter.py: build_enriched_chunk_text`)

A structural header is prepended to each chunk's text for embedding. This gives the vector index richer signals to match on:

```
[File: src/rag/retriever.py | Language: Python | Section: function | Lines: 45-120]
[Defines: find_logical_boundaries, extract_code_elements]
<original code text>
```

The raw code text is preserved separately in `meta_data['raw_chunk_text']` — the LLM sees clean code in prompts, while the vector index searches over the enriched text.

Documentation files get a lighter header via `build_enriched_doc_text`:
```
[File: docs/setup.md | Type: documentation]
<original text>
```

## Output

Each chunk is saved as a JSON vector file with metadata:

```python
{
    'file_path': str,              # Relative path in repo
    'is_code': bool,
    'type': str,                   # File extension
    'raw_chunk_text': str,         # Clean text for display
    'section_type': str,           # function/class/imports/code/...
    'start_line': int,             # Line range in source file
    'end_line': int,
    'functions': List[str],        # Detected function names
    'classes': List[str],          # Detected class names
    'chunk_index': int,            # Position within file's chunks
    'total_chunks_in_file': int,
}
```

## Downstream Usage

### File-Path-Aware Retrieval (`retriever.py: call_with_file_filter`)

Wiki page generation uses a two-tier retrieval strategy instead of blind semantic search.

**Why:** The wiki structure prompt already identifies the most relevant source files per page in `WikiPage.filePaths`. Using the page title as a bare RAG query produces poor retrieval — "Dependency Injection System" won't match code containing `Depends`, `solve_dependencies`, etc. File-path-aware retrieval exploits the known-relevant file list.

**Algorithm:**

```
Input: query (page title), file_paths (declared relevant files), top_k_wiki (40)

Step 1 — File-filtered collection:
  For each doc in transformed_docs:
    if doc.file_path IN file_paths → add to file_chunks[]
  
  Result: ALL chunks from declared relevant files (complete file context)

Step 2 — Semantic supplementation:
  Run FAISS similarity search on query with top_k=top_k_wiki
  For each result not already in file_chunks → append

Step 3 — Return merged list (file chunks first, then semantic)
```

**Why file chunks first:** When the wiki generation LLM receives all chunks from a file in order, it effectively sees the entire file — function definitions in context, imports at the top, class structure intact. This is strictly more powerful than the ±2 neighbor window previously used by chunk_enhancer, because:

1. The context reaches the **generation LLM** (which writes the wiki page), not just the embedding LLM
2. The window is the **entire file**, not just ±2 chunks (~4K tokens)
3. It costs **zero extra LLM calls** — just a metadata filter over already-loaded documents

**Fallback:** If `call_with_file_filter` fails for any reason, it falls back to standard `call()` (blind semantic search with default top_k).

**Configuration:**

| Parameter | Config File | Default | Purpose |
|-----------|------------|---------|---------|
| `top_k` | `embedder.json` | 12 | Semantic search depth for interactive chat |
| `top_k_wiki` | `embedder.json` | 40 | Semantic search depth for wiki page generation |

### Context Formatting (`service.py: format_context_text`)

Retrieved chunks are grouped by file and formatted with structural headers for the LLM:

```
## File Path: src/rag/retriever.py
(Type: py | Classes: FAISSRetriever | Functions: search, build_index)

### [function] (lines 45-120)
<code>
```

**Wiki generation** (`wiki_page.py`): The prompt instructs the LLM to interpret file headers, structural summaries, section markers (`[function]`, `[class]`), and line references for accurate source citations.
