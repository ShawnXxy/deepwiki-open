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


def format_context_text(retrieved_documents) -> str:
    """
    Format retrieved documents into context text.

    Args:
        retrieved_documents: Documents retrieved from RAG

    Returns:
        str: Formatted context text
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

    # Format context text
    context_parts = []
    for file_path, docs in docs_by_file.items():
        header = f"## File Path: {file_path}\n\n"
        content = "\n\n".join([doc.text for doc in docs])
        context_parts.append(f"{header}{content}")

    return "\n\n" + "-" * 10 + "\n\n".join(context_parts)
