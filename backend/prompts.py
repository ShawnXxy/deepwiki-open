"""
Module containing all prompts used in the DeepWiki project.

DEPRECATED: This module is kept for backward compatibility.
Please import from backend.promptstore instead.

Example:
    from backend.promptstore import RAG_SYSTEM_PROMPT, SIMPLE_CHAT_SYSTEM_PROMPT
"""

# Re-export all prompts from promptstore for backward compatibility
from backend.promptstore import (
    RAG_SYSTEM_PROMPT,
    RAG_TEMPLATE,
    SIMPLE_CHAT_SYSTEM_PROMPT,
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    WIKI_STRUCTURE_PROMPT,
    WIKI_STRUCTURE_CONCISE_PROMPT,
    WIKI_PAGE_CONTENT_PROMPT,
)

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RAG_TEMPLATE",
    "SIMPLE_CHAT_SYSTEM_PROMPT",
    "DEEP_RESEARCH_FIRST_ITERATION_PROMPT",
    "DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT",
    "DEEP_RESEARCH_FINAL_ITERATION_PROMPT",
    "WIKI_STRUCTURE_PROMPT",
    "WIKI_STRUCTURE_CONCISE_PROMPT",
    "WIKI_PAGE_CONTENT_PROMPT",
]
