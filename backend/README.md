# DeepWiki Backend

The backend is a FastAPI application that provides AI-powered wiki generation, RAG-based Q&A, and repository analysis for code repositories.

## Architecture Overview

```
backend/
├── app.py              # FastAPI application setup and route registration
├── main.py             # Application entry point with uvicorn server
├── config.py           # Configuration management (reads JSON config files)
├── prompts.py          # Prompt management (reserved for future use)
│
├── modules/            # Core business logic modules
│   ├── chat/          # Chat and deep research functionality
│   ├── rag/           # Retrieval-Augmented Generation pipeline
│   ├── repository/    # Git operations and file content retrieval
│   └── wiki/          # Wiki cache, export, and structure management
│
├── clients/           # External service clients
│   ├── azureai_client.py    # Azure OpenAI client (LLM + embeddings)
│   ├── blob_client.py       # Azure Blob Storage client
│   ├── storage.py           # Unified storage abstraction (blob/local)
│   └── vector_storage.py    # JSON-based vector embeddings storage
│
├── promptstore/       # Centralized prompt templates
│   ├── rag.py              # RAG system prompts
│   ├── simple_chat.py      # Simple chat prompts
│   ├── deep_research.py    # Multi-turn research prompts
│   ├── wiki_structure.py   # Wiki structure generation prompts
│   ├── wiki_page.py        # Wiki page content prompts
│   └── chat_system.py      # Chat system prompt builder
│
├── tools/             # Utility tools
│   ├── embedder.py    # Embedding utilities with token handling
│   └── logger.py      # Logging with smart deduplication
│
├── types/             # Type definitions (Pydantic models)
│   ├── config_types.py     # Configuration type classes
│   ├── git_types.py        # Git-related types
│   ├── processor_types.py  # File processing types
│   └── converter.py        # Type conversion utilities
│
├── utils/             # General utilities
│   └── paths.py       # Centralized path management (~/.adalflow)
│
└── config/            # JSON configuration files
    ├── infra.json     # Azure infrastructure settings
    ├── embedder.json  # Embedding model configuration
    ├── generator.json # LLM generation configuration
    ├── repo.json      # File filter settings
    └── lang.json      # Language configuration
```

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Frontend  │────►│  FastAPI     │────►│  Modules    │
│  (Next.js)  │     │  (app.py)    │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
             ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
             │ modules/chat │           │ modules/rag  │           │ modules/wiki │
             │              │           │              │           │              │
             │ - WebSocket  │           │ - Retriever  │           │ - Cache      │
             │ - HTTP API   │           │ - Database   │           │ - Export     │
             │ - Service    │           │ - Documents  │           │ - Routes     │
             └──────────────┘           └──────────────┘           └──────────────┘
                    │                            │                            │
                    └────────────────────────────┼────────────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
             ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
             │   clients/   │           │ promptstore/ │           │    tools/    │
             │              │           │              │           │              │
             │ - Azure AI   │           │ - RAG        │           │ - Embedder   │
             │ - Blob       │           │ - Wiki       │           │ - Logger     │
             │ - Storage    │           │ - Chat       │           │              │
             └──────────────┘           └──────────────┘           └──────────────┘
```

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/chat` | WebSocket | Streaming chat with RAG context |
| `/chat/completions/stream` | POST | HTTP streaming chat |
| `/api/wiki_cache` | GET/POST/DELETE | Wiki cache operations |
| `/api/export/wiki` | POST | Export wiki to Markdown/JSON |
| `/api/processed_projects` | GET | List processed repositories |
| `/local_repo/structure` | GET | Get local repo file tree |
| `/models/config` | GET | Available model configuration |
| `/auth/status` | GET | Authentication status |

> **Note**: This deployment uses Azure OpenAI exclusively for all LLM and embedding operations.

## Storage Paths

All data is stored under `~/.adalflow/`:

| Path | Purpose |
|------|---------|
| `~/.adalflow/repos/` | Cloned Git repositories |
| `~/.adalflow/wikicache/` | Generated wiki JSON files |
| `~/.adalflow/vectors/` | FAISS vector embeddings |
| `~/.adalflow/embedding_cache/` | Cached embeddings |

## Environment Compatibility

| Environment | Auth Method | Storage |
|-------------|-------------|---------|
| Local Terminal | Developer Identity | Local disk |
| Local Docker | API Key (.env) | Local disk (volume mount) |
| Azure Container App | MSI | Azure Blob Storage |

## Quick Start

```powershell
# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run backend
python -m backend.main

# Server starts at http://localhost:8001
```

See individual module READMEs for detailed documentation.
