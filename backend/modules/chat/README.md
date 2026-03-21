# Chat Module

Real-time code Q&A backend for DeepWiki.

## Responsibility

Provides conversational Ask/Chat over code repositories via two transport protocols:
- **WebSocket** (`ws_handler.py`) — Primary, used in local/Docker development
- **HTTP streaming** (`http_handler.py`) — Fallback for cloud environments behind load balancers

Supports two chat modes:
- **Simple chat** — Direct Q&A with single LLM response
- **Deep research** — Multi-turn investigation (5 iterations: plan → investigate → synthesize → conclude)

## Files

| File | Purpose |
|------|---------|
| `ws_handler.py` | WebSocket streaming handler with keepalive pings |
| `http_handler.py` | HTTP `StreamingResponse` fallback for cloud |
| `service.py` | Shared utilities: prompt building, context formatting, content sanitization |
| `models.py` | `ChatCompletionRequest`, `ChatMessage` — Pydantic request models |

## How It Works

### Request Flow

```
Frontend (Ask.tsx)
    │
    ├─ WebSocket ──► ws://localhost:8001/ws/chat
    │                     │
    │                     ▼
    │               ws_handler.py
    │               ├─ Parse ChatCompletionRequest
    │               ├─ Prepare RAG retriever (with keepalive pings)
    │               ├─ Build system prompt
    │               ├─ Stream LLM response (token-by-token)
    │               └─ Send completion message
    │
    └─ HTTP ──────► POST /chat/completions/stream
                          │
                          ▼
                    http_handler.py
                    └─ Same logic, StreamingResponse output
```

### WebSocket Handler (`ws_handler.py`)

`handle_websocket_chat()` is the main entry point, registered at `/ws/chat` in `app.py`.

**Key behaviors:**

1. **Retriever preparation with keepalive** — Embedding can take minutes for large repos. `prepare_retriever_with_keepalive()` runs embedding in a thread pool while sending 30-second keepalive pings to prevent WebSocket timeout.

2. **File-priority retrieval** — When the request specifies `filePaths`, uses `RAG.call_with_file_filter()` to prioritize chunks from those files.

3. **Deep Research mode** — When `deepResearch=True`, the handler runs 5 LLM iterations:
   - Iteration 1: Research plan (selected prompt: `DEEP_RESEARCH_SYSTEM_PROMPT`)
   - Iterations 2-4: Progressive investigation (prompt: `DEEP_RESEARCH_UPDATE_PROMPT`)
   - Iteration 5: Final synthesis (prompt: `DEEP_RESEARCH_CONCLUSION_PROMPT`)
   - Each iteration's output is prefixed with `<deep_research_iteration N>` tags.

4. **Wiki generation support** — The handler also processes `wiki_structure_request` and `wiki_page_request` messages, building server-side prompts and streaming wiki content back. This is the WebSocket path (alternative to the CLI processor).

5. **LLM keepalive pings** — For reasoning models (`o1-mini`, `o4-mini`) that may think for 30+ seconds, sends 15-second keepalive pings during generation.

6. **Content filter retry** — If Azure OpenAI returns a content filter error, automatically retries with a sanitized file tree (directories only, README omitted).

7. **Commit hash metadata** — Emits `commit_hash:{hash}` as a special message so the frontend can build source citation URLs.

### Shared Service (`service.py`)

Utility functions used by both handlers:

- **`build_system_prompt(mode, iteration)`** — Selects the right prompt template:
  - Simple chat → `SIMPLE_CHAT_SYSTEM_PROMPT`
  - Deep research iteration 1 → `DEEP_RESEARCH_SYSTEM_PROMPT`
  - Deep research iterations 2-4 → `DEEP_RESEARCH_UPDATE_PROMPT`
  - Deep research iteration 5 → `DEEP_RESEARCH_CONCLUSION_PROMPT`

- **`format_context_text(docs, repo_url, commit_hash, repo_type)`** — Formats retrieved chunks for the LLM:
  ```
  ## File Path: src/auth/handler.py
  **Source:** [View in repository](https://dev.azure.com/org/proj/_git/repo?path=/src/auth/handler.py&version=GCabc123)
  **Language:** Python | **Section:** function | **Functions:** authenticate, validate_token
  
  [chunk content]
  ```

- **`format_conversation_history(messages)`** — Formats prior dialog turns as XML:
  ```xml
  <previous_messages>
  <message role="user">How does auth work?</message>
  <message role="assistant">The auth module...</message>
  </previous_messages>
  ```

- **`get_language_info(code)`** — Resolves language code (`en`, `zh`, `ja`) to display name

- **`_sanitize_for_content_filter(text)`** — Redacts connection strings, API keys, and credentials from context before sending to Azure OpenAI

### Request Model (`models.py`)

`ChatCompletionRequest` encapsulates everything needed for a chat turn:

| Field | Type | Purpose |
|-------|------|---------|
| `repo` | `RepoInfo` | Repository URL, owner, type, branch |
| `messages` | `List[ChatMessage]` | Conversation history |
| `provider` | `str` | Always `"azure"` |
| `model` | `str` | Ignored (model from `infra.json`) |
| `deepResearch` | `bool` | Enable 5-iteration deep research |
| `language` | `str` | Response language code |
| `excludedDirs/Files` | `List[str]` | File filter overrides |
| `includedDirs/Files` | `List[str]` | Inclusion-mode filters |
| `isWikiChat` | `bool` | Whether this is wiki-specific Ask |
| `filePaths` | `List[str]` | Files to prioritize in retrieval |

## Frontend Integration

- **WebSocket:** `src/utils/websocketClient.ts` connects to `ws://localhost:8001/ws/chat` (local) or `wss://hostname/ws/chat` (cloud via nginx proxy)
- **HTTP:** `src/app/api/chat/stream/route.ts` proxies to `/chat/completions/stream`
- Both transports fail gracefully when backend is absent — the frontend works as a read-only wiki viewer without chat

## Dependencies

- **Invokes:** `embedder/` (RAG retrieval), `repository/file_content` (remote file reads), `promptstore/` (system prompts)
- **Invoked by:** `app.py` (registered as FastAPI WebSocket + HTTP routes)
