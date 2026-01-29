"""
Chat Module

This module provides chat and deep research functionality including:
- HTTP streaming chat endpoints
- WebSocket chat handlers
- System prompt building

Exports:
    - ChatMessage: Chat message model
    - ChatCompletionRequest: Chat completion request model
    - chat_completions_stream: HTTP streaming endpoint
    - handle_websocket_chat: WebSocket handler
    - build_system_prompt: System prompt builder
"""

from backend.modules.chat.models import ChatMessage, ChatCompletionRequest
from backend.modules.chat.http_handler import chat_completions_stream
from backend.modules.chat.ws_handler import handle_websocket_chat, prepare_retriever_with_keepalive
from backend.modules.chat.service import build_system_prompt

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "chat_completions_stream",
    "handle_websocket_chat",
    "prepare_retriever_with_keepalive",
    "build_system_prompt",
]
