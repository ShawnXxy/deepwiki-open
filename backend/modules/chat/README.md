# Chat Module

Ask/Chat Q&A backend for DeepWiki.

## Responsibility

Provides real-time code Q&A via two transport protocols:
- **WebSocket** (`ws_handler.py`) — Primary, used in local/Docker development
- **HTTP streaming** (`http_handler.py`) — Fallback for cloud environments

Supports two chat modes:
- **Simple chat** — Direct Q&A with single LLM response
- **Deep research** — Multi-turn investigation (5 iterations with plan → updates → conclusion)

## Files

| File | Purpose |
|------|---------|
| `ws_handler.py` | WebSocket streaming handler with keepalive support |
| `http_handler.py` | HTTP `StreamingResponse` fallback for cloud |
| `service.py` | `build_system_prompt()`, `format_context_text()`, `format_conversation_history()` |
| `models.py` | `ChatCompletionRequest`, `ChatMessage` — request models |

## Dependencies

- **Invokes:** `embedder/` (RAG retrieval), `repository/file_content` (read files), `promptstore/` (system prompts)
- **Invoked by:** `app.py` (registered as FastAPI routes)

## Frontend Integration

- WebSocket: `src/utils/websocketClient.ts` connects to `ws://localhost:8001/ws/chat`
- HTTP: `src/app/api/chat/stream/route.ts` proxies to `/chat/completions/stream`
- Both fail gracefully when backend is absent (frontend works without chat)
