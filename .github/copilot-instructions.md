# Copilot Instructions for DeepWiki-Open

## Architecture Overview
DeepWiki-Open generates AI-powered wikis for code repositories using a **FastAPI backend** (`api/`) and **Next.js frontend** (`src/`).

**Note:** This project only supports Azure OpenAI as the model provider, using **Managed Identity (MSI)** for authentication.

### Data Flow
1. Frontend sends repo URL via WebSocket (`src/utils/websocketClient.ts` → `api/websocket_wiki.py`)
2. Backend clones repo, chunks files, generates embeddings (`api/data_pipeline.py`)
3. RAG retrieves relevant chunks for queries (`api/rag.py`)
4. LLM generates responses streamed back via WebSocket

### Critical Files
- `api/config.py`: Central configuration loader - Azure OpenAI settings, MSI config
- `api/config/infra.json`: Managed Identity (MSI) configuration
- `api/config/generator.json`: Azure OpenAI model parameters (temperature)
- `api/config/embedder.json`: Azure OpenAI embedding model settings
- `api/websocket_wiki.py`: Main chat handler using Azure OpenAI
- `api/azureai_client.py`: Azure OpenAI client with MSI authentication

## Azure OpenAI Configuration

### All Configuration in infra.json
All Azure OpenAI settings are in `api/config/infra.json` - no `.env` file needed:
```json
{
  "managed_identity": {
    "name": "mid-deepwiki-ea",
    "client_id": "6e58add9-816c-4307-b9bd-bb7ce616d486"
  },
  "azure_openai": {
    "endpoint": "https://your-resource.openai.azure.com",
    "api_version": "2024-12-01-preview",
    "deployment": "o4-mini"
  },
  "azure_openai_embedding": {
    "endpoint": "https://your-resource.openai.azure.com",
    "api_version": "2024-12-01-preview",
    "deployment": "text-embedding-3-large"
  }
}
```

### Key Patterns
- All config loaded from `infra.json` via `get_infra_config()`
- MSI client_id via `get_managed_identity_client_id()`
- Azure OpenAI config via `get_azure_openai_config()` and `get_azure_openai_embedding_config_from_infra()`
- `AzureAIClient` uses `DefaultAzureCredential` with the configured MSI client_id
- Reasoning models (o1, o3, o4-mini) only support `temperature=1.0` - no `top_p`
- Config validation: `is_azure_openai_configured()` checks endpoint pattern and MSI config

## Development Workflow

### Running Locally
```bash
# Backend (from project root)
.venv\Scripts\python.exe -m api.main  # Windows
uv run -m api.main                     # or with uv

# Frontend
npm run dev
```

### Testing
```bash
python tests/run_tests.py              # All tests
python tests/run_tests.py --unit       # Fast unit tests
python tests/run_tests.py --api        # API endpoint tests (requires running server)
```

### Debugging Tips
- Logs: `api/logs/application.log` (auto-created)
- Add debug logging with `logger.info(f"Variable: {var}")` - uses `api.logging_config`
- Config loading happens at module import - **restart server after config changes**

## Key Conventions

### Empty/None Handling
Provider and model can be empty string from frontend. Always use truthy checks:
```python
if not provider:  # handles None AND ''
    provider = "azure"  # Always Azure OpenAI
```

### Embedding Response Parsing
AdalFlow expects `Embedding` objects, not raw lists. In `azureai_client.py`:
```python
from adalflow.core.types import Embedding
embeddings.append(Embedding(embedding=vector, index=idx))
```

### Cross-Platform Paths
Always use `pathlib` or `os.path` for file operations - codebase runs on Windows/Linux.

## Frontend Notes
- WebSocket URL configured in `src/utils/networkConfig.ts`
- i18n strings: `src/messages/{lang}.json`
- Wiki pages: dynamic routes at `src/app/[owner]/[repo]/page.tsx`
- Data storage: cloned repos and vector stores persisted in `~/.adalflow/`
