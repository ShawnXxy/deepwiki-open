# Embedder Module

Chunking, embedding, and retrieval engine for DeepWiki.

## Responsibility

Takes code repositories and produces searchable vector embeddings:
1. **Read** files from cloned repos (with filters)
2. **Split** code at logical boundaries (function/class definitions)
3. **Enrich** chunks with structural metadata (file path, language, section type)
4. **Embed** via Azure OpenAI (batch mode)
5. **Store** as JSON vector files (local disk or Azure Blob)
6. **Retrieve** via FAISS semantic search (with file-priority filtering)

## Files

| File | Purpose |
|------|---------|
| `retriever.py` | Main RAG class — FAISS search + file-filtered retrieval |
| `indexer.py` | Orchestrates clone → embed → store pipeline |
| `document.py` | Read files, split, embed, save as JSON vectors |
| `code_splitter.py` | Boundary-aware splitting (14 regex patterns, 8+ languages) |
| `tokenizer.py` | Token counting (tiktoken) + safe file reading |
| `response.py` | RAGAnswer dataclass (rationale + answer) |
| `memory.py` | Conversation history tracking (dialog turns) |

## Usage

```python
from backend.modules.embedder import RAG

rag = RAG(provider='azure')
rag.prepare_retriever(repo_url, type='azuredevops', branch='main')
results = rag.call_with_file_filter(query, file_paths, top_k=40)
```

## Dependencies

- **Invokes:** `repository/git_ops` (clone repos), `clients/embedder` (Azure OpenAI embeddings)
- **Invoked by:** `chat/` (Ask/Chat), `processor/` (CLI wiki generation)
