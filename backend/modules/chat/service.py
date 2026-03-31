"""
Chat service utilities.

Provides shared logic for building system prompts and formatting conversations.
"""

import logging

from backend.config import configs
from backend.promptstore import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT,
)
from backend.promptstore import build_chat_system_prompt

logger = logging.getLogger(__name__)


def build_system_prompt(
    is_deep_research: bool,
    research_iteration: int,
    repo_type: str,
    repo_url: str,
    repo_name: str,
    language_name: str
) -> str:
    """
    Build the system prompt based on chat mode and parameters.

    Args:
        is_deep_research: Whether this is a deep research request
        research_iteration: Current iteration number for deep research
        repo_type: Type of repository (github, gitlab, etc.)
        repo_url: URL of the repository
        repo_name: Name of the repository
        language_name: Display name of the language

    Returns:
        str: The formatted system prompt
    """
    if is_deep_research:
        is_first_iteration = research_iteration == 1
        is_final_iteration = research_iteration >= 5

        if is_first_iteration:
            return DEEP_RESEARCH_FIRST_ITERATION_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
            )
        elif is_final_iteration:
            return DEEP_RESEARCH_FINAL_ITERATION_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
            )
        else:
            return DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name,
                research_iteration=research_iteration
            )
    else:
        return SIMPLE_CHAT_SYSTEM_PROMPT.format(
            repo_type=repo_type,
            repo_url=repo_url,
            repo_name=repo_name,
            language_name=language_name
        )


def get_language_info(language_code: str = None) -> tuple:
    """
    Get language code and display name from config.

    Args:
        language_code: Optional language code (defaults to config default)

    Returns:
        Tuple of (language_code, language_name)
    """
    lang_config = configs["lang_config"]
    code = language_code or lang_config["default"]
    supported = lang_config["supported_languages"]
    name = supported.get(code, "English")
    return code, name


def format_conversation_history(memory_dict: dict) -> str:
    """
    Format conversation history from memory into XML-like structure.

    Args:
        memory_dict: Dictionary of dialog turns from Memory

    Returns:
        str: Formatted conversation history
    """
    conversation_history = ""
    for turn_id, turn in memory_dict.items():
        if (not isinstance(turn_id, int) and
                hasattr(turn, 'user_query') and
                hasattr(turn, 'assistant_response')):
            conversation_history += (
                f"<turn>\n<user>{turn.user_query.query_str}</user>\n"
                f"<assistant>{turn.assistant_response.response_str}"
                f"</assistant>\n</turn>\n"
            )
    return conversation_history


def _sanitize_for_content_filter(text: str) -> str:
    """Strip patterns from code that commonly trigger Azure content filters.

    Replaces sensitive-looking content (credentials, security rules,
    IP/firewall patterns) with safe placeholders so the LLM call
    is less likely to be truncated by the content management policy.
    """
    import re
    # Password / secret / key assignments
    text = re.sub(
        r'(["\']?(?:password|secret|api_key|token|credential|auth_token'
        r'|private_key|client_secret)["\']?\s*[:=]\s*)["\'][^"\']{4,}["\']',
        r'\1"<REDACTED>"',
        text, flags=re.IGNORECASE
    )
    # Connection strings
    text = re.sub(
        r'((?:Server|Data Source|Host)=[^;\n]{10,})',
        '<CONNECTION_STRING_REDACTED>',
        text, flags=re.IGNORECASE
    )
    return text


def format_context_text(
    retrieved_documents,
    repo_url: str = "",
    commit_hash: str = "",
    repo_type: str = "github",
) -> str:
    """
    Format retrieved documents into context text for LLM consumption.

    When enriched chunks are available (from code-aware splitting),
    includes structural metadata like section type, function names,
    and class names. Falls back to basic format for legacy chunks.

    When repo_url and commit_hash are provided, appends a commit-pinned
    source URL per chunk so the LLM can produce accurate citations.

    Args:
        retrieved_documents: Documents retrieved from RAG
        repo_url: Repository URL for generating source links
        commit_hash: Commit SHA for pinning source links
        repo_type: Repository type (github, azuredevops, etc.)

    Returns:
        str: Formatted context text with structural headers
    """
    if not retrieved_documents or not retrieved_documents[0].documents:
        return ""

    documents = retrieved_documents[0].documents
    logger.info(f"Retrieved {len(documents)} documents")

    # Group documents by file path
    docs_by_file = {}
    for doc in documents:
        file_path = doc.meta_data.get('file_path', 'unknown')
        if file_path not in docs_by_file:
            docs_by_file[file_path] = []
        docs_by_file[file_path].append(doc)

    # Format context text with enriched headers
    context_parts = []
    for file_path, docs in docs_by_file.items():
        # Build file-level header
        header = f"## File Path: {file_path}"

        # Collect structural info across chunks from this file
        all_functions = []
        all_classes = []
        file_type = None

        for doc in docs:
            meta = doc.meta_data
            if not file_type:
                file_type = meta.get('type', '')
            all_functions.extend(meta.get('functions', []))
            all_classes.extend(meta.get('classes', []))

        # Add structural summary under header
        summary_parts = []
        if file_type:
            summary_parts.append(f"Type: {file_type}")
        unique_classes = list(dict.fromkeys(all_classes))
        unique_functions = list(dict.fromkeys(all_functions))
        if unique_classes:
            summary_parts.append(
                f"Classes: {', '.join(unique_classes[:8])}"
            )
        if unique_functions:
            summary_parts.append(
                f"Functions: {', '.join(unique_functions[:12])}"
            )
        if summary_parts:
            header += f"\n({' | '.join(summary_parts)})"
        header += "\n"

        # Derive display text: strip enrichment header from doc.text.
        # New chunks store _header_len (offset of raw code in .text).
        # Legacy chunks (loaded from old JSON) store raw_chunk_text.
        chunk_texts = []
        for doc in docs:
            header_len = doc.meta_data.get('_header_len')
            if header_len is not None:
                raw_text = doc.text[header_len:]
            else:
                raw_text = doc.meta_data.get('raw_chunk_text', doc.text)
            section_type = doc.meta_data.get('section_type', '')
            start_line = doc.meta_data.get('start_line')
            end_line = doc.meta_data.get('end_line')

            # Add section marker if available
            if section_type and section_type != 'code':
                chunk_header = f"### [{section_type}]"
                if start_line is not None and end_line is not None:
                    chunk_header += (
                        f" (lines {start_line + 1}-{end_line + 1})"
                    )
                chunk_texts.append(f"{chunk_header}\n{raw_text}")
            else:
                chunk_texts.append(raw_text)

            # Append per-chunk source URL for LLM citation
            if repo_url and start_line is not None:
                from backend.utils.url_builder import build_source_url
                sl = start_line + 1  # 0-based → 1-based
                el = (end_line + 1) if end_line is not None else sl
                src_url = build_source_url(
                    repo_url, file_path, commit_hash,
                    repo_type, sl, el,
                )
                chunk_texts.append(
                    f"Source: [{file_path} L{sl}-L{el}]({src_url})"
                )

        content = "\n\n".join(chunk_texts)
        context_parts.append(f"{header}\n{content}")

    full_context = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
    return _sanitize_for_content_filter(full_context)
