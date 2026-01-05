"""
Prompt Store - Centralized prompt management for DeepWiki.

This module organizes all prompts used in the application into separate files
for better maintainability and reusability.

Usage:
    from backend.promptstore import RAG_SYSTEM_PROMPT, WIKI_STRUCTURE_PROMPT
    
    # Or import specific modules
    from backend.promptstore.rag import RAG_SYSTEM_PROMPT, RAG_TEMPLATE
    from backend.promptstore.wiki_structure import WIKI_STRUCTURE_PROMPT
"""

# RAG prompts
from backend.promptstore.rag import (
    RAG_SYSTEM_PROMPT,
    RAG_TEMPLATE,
)

# Chat prompts
from backend.promptstore.simple_chat import SIMPLE_CHAT_SYSTEM_PROMPT

# Deep research prompts
from backend.promptstore.deep_research import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
)

# Wiki generation prompts
from backend.promptstore.wiki_structure import (
    WIKI_STRUCTURE_PROMPT,
    WIKI_STRUCTURE_CONCISE_PROMPT,
)

from backend.promptstore.wiki_page import WIKI_PAGE_CONTENT_PROMPT

# Chat system prompt builder
from backend.promptstore.chat_system import build_chat_system_prompt

__all__ = [
    # RAG
    "RAG_SYSTEM_PROMPT",
    "RAG_TEMPLATE",
    # Chat
    "SIMPLE_CHAT_SYSTEM_PROMPT",
    # Deep Research
    "DEEP_RESEARCH_FIRST_ITERATION_PROMPT",
    "DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT",
    "DEEP_RESEARCH_FINAL_ITERATION_PROMPT",
    # Wiki
    "WIKI_STRUCTURE_PROMPT",
    "WIKI_STRUCTURE_CONCISE_PROMPT",
    "WIKI_PAGE_CONTENT_PROMPT",
    # Chat system prompt builder
    "build_chat_system_prompt",
]
