# DeepWiki Backend

Two independent components: a **CLI processor** that generates wikis offline, and an optional **FastAPI server** for Ask/Chat Q&A.

## Architecture

```
backend/
├── app.py              # FastAPI server (Ask/Chat only — 5 endpoints)
├── main.py             # Server entry point (uvicorn)
├── config.py           # Configuration management (reads infra.json)
├── paths.py            # Storage layout (~/.adalflow directories)
├── logger.py           # Logging with smart deduplication + App Insights
│
├── processor/          # Standalone CLI wiki generator
│   ├── code_processor.py   # CLI entry point (--repo, --mode, --config)
│   ├── wiki_generator.py   # LLM-based wiki orchestration
│   └── cloud_setup.py      # Azure AI Search + AML pipeline setup
│
├── modules/            # Domain modules (single responsibility each)
│   ├── repository/     # Git operations (clone, pull, commit hash)
│   ├── embedder/       # Chunking, embedding, FAISS retrieval
│   ├── wiki/           # Wiki cache management + export
│   └── chat/           # Ask/Chat Q&A (WebSocket + HTTP streaming)
│
├── clients/            # Azure service clients
│   ├── azureai_client.py   # Azure OpenAI (LLM + embeddings)
│   ├── embedder.py         # Embedding client factory (SafeEmbedder)
│   ├── search_client.py    # Azure AI Search (create/query/push)
│   ├── blob_client.py      # Azure Blob Storage
│   ├── storage.py          # Unified storage abstraction
│   └── vector_storage.py   # JSON vector file storage
│
├── promptstore/        # LLM prompt templates + builders
│   ├── wiki_structure.py   # Wiki structure templates + builder
│   ├── wiki_page.py        # Page content template + builder
│   ├── chat_system.py      # Chat system prompt builder
│   ├── deep_research.py    # Multi-turn research templates
│   ├── simple_chat.py      # Single-turn Q&A template
│   └── rag.py              # RAG system prompt + context template
│
├── types/              # Pydantic models
│   ├── config_types.py     # InfraConfig, AzureMLConfig, etc.
│   ├── git_types.py        # RepoType, WikiCacheIdentifier
│   └── processor_types.py  # FileFilter, ProcessorConfig
│
├── utils/              # Utilities
│   └── url_builder.py      # Commit-pinned source file URL builder
│
└── config/             # JSON configuration files
    ├── infra.json          # Azure endpoints, MSI, blob, search, AML
    ├── embedder.json       # Embedding settings (batch_size, chunk_size)
    ├── excluded.json        # File exclusion filters (excluded_dirs, excluded_files)
    ├── included.json        # Supported file extensions (code, doc)
    └── lang.json           # Supported languages
```

## Two Operating Modes

### 1. CLI Processor (no server needed)

Generates wiki cache files offline. The frontend reads these directly.

```bash
# Local mode
python -m backend.processor.code_processor --config=backend/run.json

# Docker mode
python -m backend.processor.code_processor --mode=docker --repo=URL --branch=main

# Cloud mode — setup AML resources (run from your machine)
python -m backend.processor.aml_dispatcher --config=backend/run.json
# Processing runs automatically inside AML pipeline
```

### 2. FastAPI Server (optional, for Ask/Chat only)

Only needed if you want the Ask/Chat Q&A feature.

```bash
python -m backend.main
```

**Endpoints (7 total):**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws/chat` | WebSocket | Streaming chat with RAG context |
| `/chat/completions/stream` | POST | HTTP streaming fallback |
| `/models/config` | GET | Available model info from infra.json |
| `/filters/config` | GET | Default file exclusion patterns |
| `/api/wiki_cache` | GET | Read wiki cache (blob or local) |
| `/api/processed_projects` | GET | List processed wiki projects |
| `/health` | GET | Deployment health check |

## Module Dependency Graph

```
repository/  ← foundation (zero module deps)
     ↑
embedder/    ← depends on repository (clone for indexing)
     ↑
chat/        ← depends on embedder (RAG retrieval for Q&A)

wiki/        ← standalone (cache read/write, no module deps)

processor/   ← orchestrator (invokes repository + embedder + wiki)
```

## Storage Layout

All data under `~/.adalflow/` (consistent across Windows/Linux/Docker):

| Path | Purpose |
|------|---------|
| `wikicache/` | Generated wiki JSON cache files |
| `repos/` | Cloned git repositories |
| `vectors/` | Embedding vector JSON chunks |
| `embedding_cache/` | Embedding API response cache |

## Environment Compatibility

| Environment | Auth | Storage | Wiki Viewer | Ask/Chat |
|-------------|------|---------|-------------|----------|
| Local | `az login` or PAT | Local disk | `npm run dev` | `python -m backend.main` |
| Docker | API Key (.env) | Volume mount | Built-in | Built-in |
| Azure | MSI | Azure Blob | Container App | Container App |

## Quick Start

```powershell
# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run backend
python -m backend.main

# Server starts at http://localhost:8001
```

See individual module READMEs for detailed documentation.
