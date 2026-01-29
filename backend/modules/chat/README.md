# Chat Module

The chat module provides real-time conversational AI capabilities for repository Q&A, including both simple chat and deep research modes.

## Module Structure

```
modules/chat/
├── __init__.py       # Module exports
├── models.py         # Pydantic request/response models
├── ws_handler.py     # WebSocket handler for streaming chat
├── http_handler.py   # HTTP handler for streaming chat
└── service.py        # Shared service utilities
```

## Components

### models.py - Request Models

**ChatMessage**: Individual chat message with role and content.

**ChatCompletionRequest**: Main request model containing:
- `repo_url`: Repository URL to query
- `messages`: Conversation history
- `wiki_structure_request`: Flag for wiki structure generation
- `file_tree`, `readme`, `comprehensive`: Wiki generation params
- File filter options (`excluded_dirs`, `included_files`, etc.)
- `force_reprocess`: Force fresh embedding generation

### ws_handler.py - WebSocket Handler

The primary chat endpoint (`/ws/chat`) supporting:

1. **Keepalive Mechanism**: `prepare_retriever_with_keepalive()` sends ping messages during long-running embedding operations to prevent WebSocket timeout.

2. **Wiki Structure Generation**: Detects `wiki_structure_request=True` and builds prompts from `promptstore/wiki_structure.py` templates.

3. **RAG-based Chat**: Retrieves relevant code context and streams responses.

```python
# Wiki structure prompt building
def build_wiki_structure_prompt(request, owner, repo) -> str:
    template = WIKI_STRUCTURE_PROMPT if request.comprehensive else WIKI_STRUCTURE_CONCISE_PROMPT
    return template.format(
        owner=owner, repo=repo,
        file_tree=request.file_tree,
        readme=request.readme,
        language_name=language_name,
        page_count=page_count
    )
```

### http_handler.py - HTTP Streaming

POST endpoint (`/chat/completions/stream`) for non-WebSocket clients:
- Same RAG functionality as WebSocket
- Returns `StreamingResponse` with SSE format
- Error handling with graceful degradation

### service.py - Shared Utilities

**build_system_prompt()**: Constructs system prompts based on mode:
- Simple chat mode → `SIMPLE_CHAT_SYSTEM_PROMPT`
- Deep research mode → Iteration-specific prompts (first, intermediate, final)

**format_conversation_history()**: Formats memory into XML-like structure for context.

**format_context_text()**: Groups retrieved documents by file path.

**get_language_info()**: Resolves language code to display name.

## Workflow

### Standard Chat Flow

```
┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│   Frontend   │────►│  ws_handler    │────►│    RAG      │
│  (WebSocket) │     │                │     │  Module     │
└──────────────┘     └────────────────┘     └──────┬──────┘
                              │                     │
                              │                     ▼
                              │              ┌─────────────┐
                              │              │  Retriever  │
                              │              │  (FAISS)    │
                              │              └──────┬──────┘
                              │                     │
                              ▼                     ▼
                     ┌────────────────┐     ┌─────────────┐
                     │  Azure OpenAI  │◄────│   Context   │
                     │   (Streaming)  │     │  Documents  │
                     └────────────────┘     └─────────────┘
```

### Wiki Structure Generation Flow

```
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│   Frontend   │────►│  ws_handler    │────►│  promptstore/   │
│              │     │                │     │  wiki_structure │
│ wiki_struct  │     │ Detects flag   │     └────────┬────────┘
│ _request:    │     │ Builds prompt  │              │
│   true       │     └────────────────┘              │
└──────────────┘              │                      │
                              ▼                      ▼
                     ┌────────────────┐     ┌─────────────────┐
                     │  Azure OpenAI  │◄────│  Filled prompt  │
                     │   (Streaming)  │     │  (file_tree,    │
                     └────────────────┘     │   readme, etc.) │
                              │             └─────────────────┘
                              ▼
                     ┌────────────────┐
                     │  XML Response  │
                     │  <wiki_struct  │
                     │   ure>...</>   │
                     └────────────────┘
```

## Usage Examples

### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/chat');
ws.send(JSON.stringify({
    repo_url: 'https://github.com/owner/repo',
    messages: [{ role: 'user', content: 'How does auth work?' }],
    type: 'github',
    language: 'en'
}));
```

### Wiki Structure Request

```javascript
ws.send(JSON.stringify({
    repo_url: 'https://github.com/owner/repo',
    messages: [{ role: 'user', content: 'Generate wiki structure' }],
    wiki_structure_request: true,
    file_tree: '...',
    readme: '...',
    comprehensive: true,
    language: 'en'
}));
```

## Dependencies

- `backend.modules.rag`: RAG pipeline for document retrieval
- `backend.promptstore`: Prompt templates
- `backend.config`: Model configuration
- `adalflow`: LLM framework
