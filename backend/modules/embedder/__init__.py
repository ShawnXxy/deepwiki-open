"""
RAG (Retrieval Augmented Generation) Module

This module provides RAG functionality including:
- Memory management for conversation history
- Document retrieval and embedding
- Answer generation with context

Exports:
    - RAG: Main RAG component class
    - Memory: Conversation memory management
    - RAGAnswer: Answer dataclass
    - DatabaseManager: Document database management
    - Utility functions: safe_read_file, count_tokens, etc.
"""

from backend.modules.embedder.memory import Memory, DialogTurn, UserQuery, AssistantResponse, CustomConversation
from backend.modules.embedder.response import RAGAnswer
from backend.modules.embedder.retriever import RAG
from backend.modules.embedder.indexer import DatabaseManager
from backend.modules.embedder.tokenizer import safe_read_file, count_tokens, MAX_EMBEDDING_TOKENS, MAX_INPUT_TOKENS
from backend.modules.embedder.document import (
    read_all_documents,
    prepare_data_pipeline,
    prepare_embed_only_pipeline,
    transform_documents_and_save_to_db,
    transform_documents_and_save_as_json,
)
from backend.modules.embedder.code_splitter import (
    split_and_enrich_documents,
    split_code_at_boundaries,
    extract_code_elements,
)

# Aliases for backward compatibility
transform = transform_documents_and_save_to_db
transform_documents_from_api = transform_documents_and_save_as_json

__all__ = [
    # Classes
    "RAG",
    "Memory",
    "DialogTurn",
    "UserQuery",
    "AssistantResponse",
    "CustomConversation",
    "RAGAnswer",
    "DatabaseManager",
    # Functions
    "safe_read_file",
    "count_tokens",
    "read_all_documents",
    "prepare_data_pipeline",
    "prepare_embed_only_pipeline",
    "transform_documents_and_save_to_db",
    "transform_documents_and_save_as_json",
    "transform",
    "transform_documents_from_api",
    "split_and_enrich_documents",
    "split_code_at_boundaries",
    "extract_code_elements",
    # Constants
    "MAX_EMBEDDING_TOKENS",
    "MAX_INPUT_TOKENS",
]
