"""
LLM-enhanced chunk processing for improved embedding quality.

Uses Azure OpenAI LLM calls to enhance code chunks with:
1. Complete code constructs with descriptions (partial snippets -> full blocks)
2. Key external reference extraction (up to 10 objects with descriptions)

This is the Phase 2 enhancement layer that sits between boundary-aware
splitting (code_splitter.py) and embedding (document.py pipeline).

Design decisions:
- Processes only code chunks (skips documentation)
- Gracefully skips chunks that hit content filters (returns None)
- Preserves original raw text in metadata for fallback
- Uses neighboring chunk context for better LLM understanding
- Token budget management prevents exceeding model limits

Reference: handling_embedder_ref.md (adapted for our AzureAIClient)
"""

import logging
from typing import List, Optional

from adalflow.core.types import Document, ModelType

from backend.modules.rag.utils import count_tokens

logger = logging.getLogger(__name__)

# Default token limits for LLM enhancement calls
DEFAULT_MAX_OUTPUT_ENHANCED = 4096
DEFAULT_MAX_OUTPUT_KEY_OBJECTS = 512


def _is_reasoning_model(deployment: str) -> bool:
    """Check if deployment is an o-series reasoning model (o1, o3, o4, etc.)."""
    return (
        deployment.startswith("o")
        and len(deployment) > 1
        and deployment[1].isdigit()
    )


def _build_api_kwargs(
    deployment: str,
    messages: list,
    max_output_tokens: int,
    temperature: float = 0.0,
) -> dict:
    """Build api_kwargs with correct token parameter for the model."""
    kwargs = {
        "model": deployment,
        "messages": messages,
    }
    if _is_reasoning_model(deployment):
        # o-series models require max_completion_tokens, not max_tokens
        # and only support temperature=1.0
        kwargs["max_completion_tokens"] = max_output_tokens
        kwargs["temperature"] = 1.0
    else:
        kwargs["max_tokens"] = max_output_tokens
        kwargs["temperature"] = temperature
    return kwargs


def llm_enhance_chunks(
    chunks: List[Document],
    client,
    deployment: str,
    max_context_window: int = 128000,
    max_output_enhanced: int = DEFAULT_MAX_OUTPUT_ENHANCED,
    max_output_key_objects: int = DEFAULT_MAX_OUTPUT_KEY_OBJECTS,
    repo_description: str = "",
    progress_callback: callable = None,
) -> List[Document]:
    """
    Enhance code chunks using LLM calls for better embedding quality.

    For each code chunk, makes two LLM calls:
    1. Enhanced context: Completes partial code into full logical blocks
       with a brief description of how it fits into the codebase
    2. Key objects: Extracts up to 10 key references (classes, functions,
       modules) with name and description in JSON format

    Documentation chunks are passed through unchanged.
    Chunks that fail content filters are excluded (returns None -> filtered).

    Args:
        chunks: List of Document objects from split_and_enrich_documents()
        client: AzureAIClient instance
        deployment: Azure OpenAI deployment name (e.g., 'o4-mini')
        max_context_window: Maximum token window for the deployment
        max_output_enhanced: Max output tokens for enhanced context call
        max_output_key_objects: Max output tokens for key objects call
        repo_description: Brief description of the repository
        progress_callback: Optional callback(processed, total) for progress

    Returns:
        List of enhanced Document objects (failed chunks excluded)
    """
    enhanced = []
    total = len(chunks)
    skipped = 0

    for i, chunk in enumerate(chunks):
        section_type = chunk.meta_data.get('section_type', '')

        # Only enhance code chunks, pass docs through unchanged
        if section_type == 'documentation':
            enhanced.append(chunk)
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        result = _enhance_single_chunk(
            chunk,
            client=client,
            deployment=deployment,
            max_context_window=max_context_window,
            max_output_enhanced=max_output_enhanced,
            max_output_key_objects=max_output_key_objects,
            repo_description=repo_description,
        )

        if result is not None:
            enhanced.append(result)
        else:
            skipped += 1
            fp = chunk.meta_data.get('file_path', 'unknown')
            ci = chunk.meta_data.get('chunk_index', '?')
            logger.warning(
                f"[Enhance] Skipped chunk {ci} from {fp} "
                f"(content filter or too long)"
            )

        if progress_callback:
            progress_callback(i + 1, total)

    logger.info(
        f"[Enhance] Enhanced {len(enhanced)}/{total} chunks "
        f"({skipped} skipped)"
    )
    return enhanced


def _enhance_single_chunk(
    doc: Document,
    client,
    deployment: str,
    max_context_window: int,
    max_output_enhanced: int,
    max_output_key_objects: int,
    repo_description: str,
) -> Optional[Document]:
    """
    Enhance a single code chunk with LLM-generated context.

    Makes two LLM calls:
    1. Enhanced context — completes partial code, adds description
    2. Key objects — extracts key references as JSON

    Args:
        doc: Document with raw_chunk_text and neighbor context
        client: AzureAIClient instance
        deployment: Model deployment name
        max_context_window: Max tokens for the model
        max_output_enhanced: Max output for enhanced context
        max_output_key_objects: Max output for key objects
        repo_description: Repository description for context

    Returns:
        Enhanced Document, or None if skipped (content filter/too long)
    """
    raw_text = doc.meta_data.get('raw_chunk_text', doc.text)
    previous_chunks = doc.meta_data.get('previous_chunks', [])
    next_chunks = doc.meta_data.get('next_chunks', [])
    file_path = doc.meta_data.get('file_path', 'unknown')
    chunk_index = doc.meta_data.get('chunk_index', 0)

    # Build content and context strings
    content = "SNIPPET:\n\n" + raw_text
    context = _build_context_string(previous_chunks, raw_text, next_chunks)

    # Token budget: reserve space for prompts + output
    max_input_length = max_context_window - max_output_enhanced * 2
    content_tokens = count_tokens(content)

    if content_tokens > max_input_length:
        logger.warning(
            f"[Enhance] Chunk {chunk_index} from {file_path} too long "
            f"({content_tokens} tokens > {max_input_length} limit). "
            f"Skipping enhancement."
        )
        return doc  # Return unenhanced rather than None

    # Trim context to fit within budget
    context = _trim_context_to_fit(
        context, previous_chunks, raw_text, next_chunks,
        max_context_length=max_input_length - content_tokens,
    )

    # ---- Step 1: Enhanced context ----
    enhanced_content = _call_enhanced_context(
        client, deployment, content, context,
        repo_description, max_output_enhanced,
    )
    if enhanced_content is None:
        # Content filter hit — return unenhanced chunk rather than
        # dropping it.  Same approach as the "too long" case above.
        # (ref: handling_embedder_ref.md — skip gracefully)
        return doc

    # ---- Step 2: Key objects extraction ----
    key_objects = _call_key_objects(
        client, deployment, enhanced_content,
        max_output_key_objects,
    )
    # key_objects can be None (content filter) — that's OK, we still
    # keep the enhanced content

    # Update document with enhanced data.
    # Re-apply Phase 1 enrichment header so the embedding still
    # captures file identity, language, section type and line range.
    original_enriched = doc.text  # Phase 1 enriched text
    header_end = original_enriched.find('\n\n')
    if header_end > 0:
        enrichment_header = original_enriched[:header_end]
        doc.text = enrichment_header + '\n\n' + enhanced_content
    else:
        doc.text = enhanced_content

    doc.meta_data['raw_content'] = raw_text
    doc.meta_data['key_external_objects'] = key_objects or ""
    doc.meta_data['llm_enhanced'] = True

    logger.debug(
        f"[Enhance] Chunk {chunk_index} from {file_path} enhanced "
        f"({count_tokens(doc.text)} tokens)"
    )
    return doc


# ============================================================================
# LLM Call Helpers
# ============================================================================

def _call_enhanced_context(
    client,
    deployment: str,
    content: str,
    context: str,
    repo_description: str,
    max_output_tokens: int,
) -> Optional[str]:
    """
    LLM call to enhance a code snippet into a complete logical block.

    Asks the model to:
    - Complete partial code into a full function/class/block
    - Add a brief description of the code and its role
    - Return description FIRST, enhanced snippet SECOND

    Args:
        client: AzureAIClient instance
        deployment: Model deployment name
        content: "SNIPPET:\\n\\n<code>"
        context: "CONTEXT:\\n\\n<surrounding code>"
        repo_description: Brief repo description
        max_output_tokens: Max output tokens

    Returns:
        Enhanced content string, or None if content filter hit
    """
    system_prompt = (
        "Your task is to enhance the code snippet provided under "
        "'SNIPPET' to have a clearly defined code construct or block "
        "around it, such as a function or class. "
        + (
            "This code is part of a git repository with the following "
            f"description: {repo_description}. "
            if repo_description else ""
        )
        + "For this, use the 'CONTEXT' input. "
        "MAKE SURE you only enhance the 'SNIPPET' based on the "
        "'CONTEXT' - DO NOT INVENT new code. If you cannot have a "
        "clearly defined code construct or block, please leave the "
        "code as is. "
        "In addition, provide a brief description of the enhanced "
        "code snippet and how it fits into the overall codebase. "
        "Return the description FIRST, and the enhanced snippet SECOND."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
        {"role": "user", "content": content},
    ]

    try:
        response = client.call(
            api_kwargs=_build_api_kwargs(
                deployment, messages, max_output_tokens,
            ),
            model_type=ModelType.LLM,
        )
        return _extract_response_text(response)
    except Exception as e:
        if _is_content_filter_error(e):
            logger.warning(
                f"[Enhance] Content filter on enhanced context: {e}"
            )
            return None
        raise


def _call_key_objects(
    client,
    deployment: str,
    enhanced_content: str,
    max_output_tokens: int,
) -> Optional[str]:
    """
    LLM call to extract key references from enhanced code.

    Asks the model to identify up to 10 key references (objects,
    classes, functions, modules) with name and description in JSON.

    Args:
        client: AzureAIClient instance
        deployment: Model deployment name
        enhanced_content: The enhanced code from step 1
        max_output_tokens: Max output tokens

    Returns:
        JSON string of key objects, or None if content filter hit
    """
    system_prompt = (
        "Identify up to 10 key references (objects, classes, "
        "functions, modules) from the following SNIPPET. "
        "For each reference, briefly describe its purpose or "
        "relationship within the code. "
        "Output in JSON format as an array of objects with 'name' "
        "and 'description' fields. "
        "DO NOT use common language code constructs - focus on "
        "references that appear to be from this codebase."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "SNIPPET:\n\n" + enhanced_content},
    ]

    try:
        response = client.call(
            api_kwargs=_build_api_kwargs(
                deployment, messages, max_output_tokens,
            ),
            model_type=ModelType.LLM,
        )
        return _extract_response_text(response)
    except Exception as e:
        if _is_content_filter_error(e):
            logger.warning(
                f"[Enhance] Content filter on key objects: {e}"
            )
            return None
        raise


# ============================================================================
# Context Management
# ============================================================================

def _build_context_string(
    previous_chunks: List[str],
    current_text: str,
    next_chunks: List[str],
) -> str:
    """Build the CONTEXT string from neighboring chunks."""
    parts = previous_chunks + [current_text] + next_chunks
    return "CONTEXT:\n\n" + " ".join(parts)


def _trim_context_to_fit(
    context: str,
    previous_chunks: List[str],
    current_text: str,
    next_chunks: List[str],
    max_context_length: int,
) -> str:
    """
    Trim context by alternately removing outer chunks until it fits.

    Alternates between removing the first previous chunk and the last
    next chunk, keeping context closest to the current snippet.

    Args:
        context: Current context string
        previous_chunks: Mutable list of previous chunk texts
        current_text: Current chunk raw text
        next_chunks: Mutable list of next chunk texts
        max_context_length: Maximum tokens for context

    Returns:
        Trimmed context string that fits within the budget
    """
    # Work with copies to avoid mutating the originals
    prev = list(previous_chunks)
    nxt = list(next_chunks)
    remove_previous = True

    while count_tokens(context) > max_context_length and (prev or nxt):
        if remove_previous and prev:
            prev = prev[1:]  # Remove oldest previous
        elif nxt:
            nxt = nxt[:-1]   # Remove farthest next
        remove_previous = not remove_previous
        context = _build_context_string(prev, current_text, nxt)

    # If still too long after removing all neighbors, use no context
    if count_tokens(context) > max_context_length:
        context = "CONTEXT:\n\nNone"

    return context


# ============================================================================
# Response Parsing
# ============================================================================

def _extract_response_text(response) -> str:
    """
    Extract text content from Azure OpenAI chat completion response.

    Handles both raw API response objects and adalflow-wrapped responses.

    Args:
        response: Chat completion response from AzureAIClient.call()

    Returns:
        The text content of the response
    """
    # Raw OpenAI API response
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content

    # String response (already extracted)
    if isinstance(response, str):
        return response

    # Fallback
    return str(response)


def _is_content_filter_error(error: Exception) -> bool:
    """
    Check if an exception is a content filter error.

    Matches both Azure-specific content filter errors and
    generic chat completion failures.

    Args:
        error: The caught exception

    Returns:
        True if this is a content filter error that should be skipped
    """
    error_type = type(error).__name__
    error_str = str(error).lower()

    return (
        "content_filter" in error_str
        or "contentfilter" in error_str
        or "ChatCompletionFailed" in error_type
        or "content_policy" in error_str
    )
