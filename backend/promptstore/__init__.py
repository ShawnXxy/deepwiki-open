"""
Prompt Store — LLM prompt templates and builders for DeepWiki.

Files:
    chat_system.py      — System prompt builder (selects template by chat mode)
    deep_research.py    — Multi-turn research templates (first/intermediate/final)
    simple_chat.py      — Single-turn Q&A system prompt template
    rag.py              — RAG system prompt + Jinja2 context template
    wiki_page.py        — Page content template + builder (format_file_paths_list, format_page_catalog)
    wiki_structure.py   — Structure templates + builder (build_wiki_structure_prompt, file_tree_dirs_only)

Usage:
    from backend.promptstore import RAG_SYSTEM_PROMPT, WIKI_STRUCTURE_PROMPT
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
    build_wiki_structure_prompt,
    file_tree_dirs_only,
    LANGUAGE_DISPLAY_NAMES,
)

from backend.promptstore.wiki_page import WIKI_PAGE_CONTENT_PROMPT
from backend.promptstore.wiki_page import build_wiki_page_prompt
from backend.promptstore.wiki_page import format_page_catalog

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
