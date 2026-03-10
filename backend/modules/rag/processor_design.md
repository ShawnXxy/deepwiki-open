# Code Processing Pipeline

## Overview

This pipeline converts a code repository into searchable vector embeddings for RAG-based wiki generation. It uses regex-based structural analysis (no AST parsers) to split code at logical boundaries and enrich each chunk with file context before embedding.

The pipeline has two phases:
- **Phase 1** (always on): Regex-based splitting and structural enrichment — zero LLM cost.
- **Phase 2** (on by default): LLM-enhanced chunk completion and reference extraction — 2 LLM calls per code chunk.

## Modules

| File | Role |
|------|------|
| `document.py` | Reads source files, orchestrates the full pipeline (split → enhance → embed → save) |
| `code_splitter.py` | Boundary-aware splitting, structural metadata extraction, enrichment |
| `chunk_enhancer.py` | Optional LLM enhancement: chunk completion + key reference extraction |
| `service.py` | Formats retrieved chunks with structural headers for LLM context |
| `wiki_page.py` | Prompt instructions for the LLM to interpret structural metadata |

## Pipeline Flow

```
read_documents()                      # Read all source files (no size limit for code)
  → split_and_enrich_documents()      # Phase 1: split + enrich
      → _attach_neighbor_context()    # Add ±2 neighbor chunks per chunk
  → llm_enhance_chunks()             # Phase 2 (if enabled): LLM enhancement
  → prepare_embed_only_pipeline()    # Embed pre-split chunks (no re-splitting)
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

### Step 6 — Attach Neighbor Context (`code_splitter.py: _attach_neighbor_context`)

After all chunks for a file are created, each chunk receives the raw text of its ±2 neighboring chunks (same file) in `previous_chunks` and `next_chunks`. This context is used by the LLM enhancement step to understand how each chunk fits within the file.

## Phase 2: LLM-Enhanced (On by Default)

> Module: `chunk_enhancer.py`
> Reference pattern: `handling_embedder_ref.md`
> Disable: set `embedder.json` → `llm_enhance.enabled: false`

Each **code chunk** (documentation chunks are skipped) receives two LLM calls:

### Call 1 — Enhanced Context (`_call_enhanced_context`)

**Input**: The chunk's code (as `SNIPPET`) + neighboring chunks (as `CONTEXT`) + repo description.

**Task**: Complete partial code snippets into full logical blocks (e.g., wrap a code fragment in its enclosing function/class). Return a description first, then the enhanced code.

**Token budget**: `max_context_window - max_output_tokens * 2` reserved for input. If context exceeds the budget, outer neighbor chunks are alternately removed (first the oldest previous, then the farthest next) until it fits.

### Call 2 — Key Object Extraction (`_call_key_objects`)

**Input**: The enhanced code from Call 1.

**Task**: Identify up to 10 key codebase-specific references (classes, functions, modules) and output as JSON `[{name, description}]`.

### Error Handling

Content filter errors are caught per-chunk. If Call 1 fails, the chunk is returned unenhanced (not dropped). If Call 2 fails, the enhanced content is kept without key objects. This ensures no chunks are lost to content filtering. (See `handling_embedder_ref.md` for the reference pattern.)

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
    'previous_chunks': List[str],  # ±2 neighbor raw texts
    'next_chunks': List[str],
    # Phase 2 only (when llm_enhance enabled):
    'raw_content': str,            # Pre-enhancement text
    'key_external_objects': str,   # JSON [{name, description}]
    'llm_enhanced': bool,
}
```

## Downstream Usage

**RAG retrieval** (`service.py`): Retrieved chunks are formatted with structural headers for the LLM:

```
## File Path: src/rag/retriever.py
(Type: py | Classes: FAISSRetriever | Functions: search, build_index)

### [function] (lines 45-120)
<code>
```

**Wiki generation** (`wiki_page.py`): The prompt instructs the LLM to interpret file headers, structural summaries, section markers (`[function]`, `[class]`), and line references for accurate source citations.
