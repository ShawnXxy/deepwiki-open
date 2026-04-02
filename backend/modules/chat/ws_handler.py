"""
WebSocket handler for chat completions.

Provides the WebSocket /ws/chat endpoint with keepalive support.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from urllib.parse import unquote

from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect

from backend.config import (
    get_model_config,
    configs,
    get_azure_deployment_name,
    get_azure_ai_client,
)
from backend.modules.embedder import RAG
from backend.modules.embedder.tokenizer import count_tokens
from backend.modules.repository.file_content import get_file_content
from backend.modules.chat.models import ChatCompletionRequest
from backend.modules.chat.service import (
    format_conversation_history,
    format_context_text,
    get_language_info,
)
from backend.promptstore import build_chat_system_prompt
from backend.promptstore.wiki_structure import (
    WIKI_STRUCTURE_PROMPT,
    WIKI_STRUCTURE_CONCISE_PROMPT,
    build_wiki_structure_prompt,
    file_tree_dirs_only as _file_tree_dirs_only,
    LANGUAGE_DISPLAY_NAMES,
)

# Thread pool for running blocking operations (like embedding)
_executor = ThreadPoolExecutor(max_workers=4)

logger = logging.getLogger(__name__)


async def _safe_send(websocket: WebSocket, text: str) -> bool:
    """Send text over WebSocket, returning False if the client disconnected.

    Catches all disconnect-related exceptions so callers can simply
    check the return value and stop streaming when it returns False.
    """
    try:
        await websocket.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError, ConnectionError, Exception) as exc:
        logger.debug(f"Client disconnected during send: {type(exc).__name__}")
        return False


async def _safe_close(websocket: WebSocket) -> None:
    """Close the WebSocket gracefully, ignoring errors if already closed."""
    try:
        await websocket.close()
    except Exception:
        pass


async def prepare_retriever_with_keepalive(
    websocket: WebSocket,
    request_rag: RAG,
    repo_url: str,
    repo_type: str,
    token: Optional[str],
    branch: Optional[str],
    excluded_dirs: Optional[List[str]],
    excluded_files: Optional[List[str]],
    included_dirs: Optional[List[str]],
    included_files: Optional[List[str]],
    force_reprocess: bool = False
) -> bool:
    """
    Run prepare_retriever in a thread pool while sending keepalive pings to the WebSocket.
    This prevents connection timeout during long-running embedding operations.
    
    Args:
        force_reprocess: If True, ignore existing pkl/vectors and create fresh JSON vectors.
    
    Returns True if successful, False if an error occurred.
    """
    loop = asyncio.get_event_loop()
    
    # Track completion and errors
    completed = asyncio.Event()
    error_message = None
    
    def run_prepare():
        nonlocal error_message
        try:
            request_rag.prepare_retriever(
                repo_url,
                repo_type,
                token,
                branch,
                excluded_dirs,
                excluded_files,
                included_dirs,
                included_files,
                force_reprocess=force_reprocess
            )
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Full traceback in prepare_retriever:\n{full_traceback}")
            error_message = str(e)
        finally:
            loop.call_soon_threadsafe(completed.set)
    
    # Start the blocking operation in a thread
    future = loop.run_in_executor(_executor, run_prepare)
    
    # Send keepalive pings every 30 seconds while waiting
    PING_INTERVAL = 30
    ping_count = 0
    
    while not completed.is_set():
        try:
            # Wait for either completion or ping interval
            await asyncio.wait_for(completed.wait(), timeout=PING_INTERVAL)
        except asyncio.TimeoutError:
            # Send a keepalive ping
            try:
                ping_count += 1
                logger.debug(f"Sending WebSocket keepalive ping #{ping_count}")
                # Send an empty comment as keepalive (will be ignored by frontend)
                await websocket.send_text(f"<!-- keepalive {ping_count} -->")
            except Exception as e:
                logger.warning(f"Failed to send keepalive ping: {e}")
                # Connection might be closed, but let's continue waiting for embedding
    
    # Wait for the future to complete (should already be done)
    await asyncio.wrap_future(future)
    
    if error_message:
        logger.error(f"Error in prepare_retriever: {error_message}")
        raise Exception(error_message)
    
    return True


async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions using Azure OpenAI.
    """
    await websocket.accept()

    try:
        # Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)
        
        # Debug: Log token status
        logger.debug(f"Request parsed - token: {'[PROVIDED]' if request.token else '[NONE]'}, type: {request.type}")

        # Handle wiki structure generation requests specially
        # These use promptstore templates instead of frontend-provided prompts
        if request.wiki_structure_request:
            # Extract owner/repo from repo_url
            repo_url = request.repo_url
            # Parse owner/repo from URL like https://github.com/owner/repo
            parts = repo_url.rstrip('/').split('/')
            owner = parts[-2] if len(parts) >= 2 else 'unknown'
            repo = parts[-1] if len(parts) >= 1 else 'unknown'

            # Build prompt from promptstore template
            prompt_content = build_wiki_structure_prompt(
                file_tree=request.file_tree or '',
                readme=request.readme or '',
                owner=owner,
                repo=repo,
                language=request.language or 'en',
                comprehensive=request.comprehensive or True,
            )

            # Replace the message content with the built prompt
            if request.messages and len(request.messages) > 0:
                request.messages[-1].content = prompt_content
                logger.info(f"Wiki structure request: using promptstore template "
                            f"for {owner}/{repo}")

        # Check if request contains very large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, False)
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(
                        f"Request exceeds recommended token limit ({tokens} > 8000)"
                    )
                    input_too_large = True

        # Create a new RAG instance for this request (always Azure)
        try:
            request_rag = RAG(provider="azure", model=request.model)

            # Extract custom file filter parameters
            excluded_dirs = None
            excluded_files = None
            included_dirs = None
            included_files = None

            if request.excluded_dirs:
                excluded_dirs = [
                    unquote(dir_path)
                    for dir_path in request.excluded_dirs.split('\n')
                    if dir_path.strip()
                ]
                logger.info(f"Using custom excluded directories: {excluded_dirs}")
            if request.excluded_files:
                excluded_files = [
                    unquote(file_pattern)
                    for file_pattern in request.excluded_files.split('\n')
                    if file_pattern.strip()
                ]
                logger.info(f"Using custom excluded files: {excluded_files}")
            if request.included_dirs:
                included_dirs = [
                    unquote(dir_path)
                    for dir_path in request.included_dirs.split('\n')
                    if dir_path.strip()
                ]
                logger.info(f"Using custom included directories: {included_dirs}")
            if request.included_files:
                included_files = [
                    unquote(file_pattern)
                    for file_pattern in request.included_files.split('\n')
                    if file_pattern.strip()
                ]
                logger.info(f"Using custom included files: {included_files}")

            # Use async version with keepalive to prevent timeout during long embedding operations
            await prepare_retriever_with_keepalive(
                websocket,
                request_rag,
                request.repo_url,
                request.type,
                request.token,
                request.branch,
                excluded_dirs,
                excluded_files,
                included_dirs,
                included_files,
                force_reprocess=request.force_reprocess or False
            )
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                await websocket.send_text(
                    "Error: No valid document embeddings found. "
                    "This may be due to embedding size inconsistencies or "
                    "API errors during document processing."
                )
                await websocket.close()
                return
            else:
                import traceback
                logger.error(
                    f"ValueError preparing retriever: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                await websocket.send_text(f"Error preparing retriever: {str(e)}")
                await websocket.close()
                return
        except Exception as e:
            import traceback
            logger.error(
                f"Error preparing retriever: {str(e)}\n{traceback.format_exc()}"
            )
            # Check for specific embedding-related errors
            if "All embeddings should be of the same size" in str(e):
                error_msg = (
                    "Error: Inconsistent embedding sizes detected. "
                    "Some documents may have failed to embed properly."
                )
                await websocket.send_text(error_msg)
            else:
                error_msg = (
                    "Configuration error: Please check your Azure OpenAI "
                    "configuration and restart the application."
                )
                await websocket.send_text(error_msg)
            await websocket.close()
            return

        # Validate request
        if not request.messages or len(request.messages) == 0:
            await websocket.send_text("Error: No messages provided")
            await websocket.close()
            return

        last_message = request.messages[-1]
        if last_message.role != "user":
            await websocket.send_text("Error: Last message must be from the user")
            await websocket.close()
            return

        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]

                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    request_rag.memory.add_dialog_turn(
                        user_query=user_msg.content,
                        assistant_response=assistant_msg.content
                    )

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                if msg == request.messages[-1]:
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        if is_deep_research:
            research_iteration = sum(
                1 for msg in request.messages if msg.role == 'assistant'
            ) + 1
            logger.info(f"Deep Research request - iteration {research_iteration}")

            # Check if this is a continuation request
            if ("continue" in last_message.content.lower() and
                    "research" in last_message.content.lower()):
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace(
                            "[DEEP RESEARCH]", ""
                        ).strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    last_message.content = original_topic
                    logger.info(f"Using original topic: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # Get repository information (needed before RAG for citation URLs)
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
        repo_type = request.type

        # Get commit hash early — needed for citation URLs in context
        commit_hash = ""
        if request.wiki_page_request and request.page_title:
            try:
                from backend.modules.repository.git_ops import (
                    get_head_commit_hash
                )
                from backend.paths import get_adalflow_root_path
                from backend.modules.embedder.indexer import DatabaseManager
                dm = DatabaseManager()
                repo_name_for_path = dm._extract_repo_name_from_url(
                    request.repo_url, request.type
                )
                local_repo_path = os.path.join(
                    get_adalflow_root_path(), "repos",
                    repo_name_for_path
                )
                if os.path.isdir(local_repo_path):
                    commit_hash = get_head_commit_hash(
                        local_repo_path
                    )
                    if commit_hash:
                        logger.info(
                            f"Commit hash: {commit_hash[:8]}"
                        )
            except Exception as e:
                logger.debug(
                    f"Could not get commit hash: {e}"
                )

        # Only retrieve documents if input is not too large
        context_text = ""
        retrieved_documents = None

        if not input_too_large:
            try:
                rag_query = query
                if request.filePath:
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(
                        f"Modified RAG query for file: {request.filePath}"
                    )

                try:
                    # Use file-path-aware retrieval for wiki page generation
                    if (request.wiki_page_request
                            and request.page_file_paths):
                        from backend.config import get_embedder_config_obj
                        wiki_top_k = get_embedder_config_obj(
                        ).retriever.top_k_wiki
                        retrieved_documents = (
                            request_rag.call_with_file_filter(
                                query=request.page_title or rag_query,
                                file_paths=request.page_file_paths,
                                top_k=wiki_top_k,
                                language=request.language,
                            )
                        )
                    else:
                        retrieved_documents = request_rag(
                            rag_query, language=request.language
                        )
                    context_text = format_context_text(
                        retrieved_documents,
                        repo_url=repo_url if request.wiki_page_request else "",
                        commit_hash=commit_hash,
                        repo_type=repo_type,
                    )
                    if not context_text:
                        logger.warning("No documents retrieved from RAG")
                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get language information
        language_code, language_name = get_language_info(request.language)

        # --- Wiki page generation: build prompt server-side ---
        # When wiki_page_request=True, the backend constructs the full
        # prompt from promptstore templates, injecting server-side data
        # (commit hash, page catalog, context) that the frontend can't.
        if request.wiki_page_request and request.page_title:
            from backend.promptstore.wiki_page import (
                build_wiki_page_prompt
            )

            # Build page catalog from the request
            page_catalog = ""
            if request.page_related_pages:
                # Use related pages as a minimal catalog
                page_catalog = "\n".join([
                    f"- {pid}" for pid in request.page_related_pages
                    if pid != (request.page_id or "")
                ])

            # Build prompt from backend promptstore
            wiki_prompt = build_wiki_page_prompt(
                page_title=request.page_title,
                page_id=request.page_id or "",
                file_paths=request.page_file_paths or [],
                context_text=context_text,
                repo_url=request.repo_url,
                commit_hash=commit_hash,
                page_catalog=page_catalog if page_catalog else None,
                language_name=language_name,
                repo_type=repo_type,
            )

            # Use the backend-built prompt, prepend /no_think
            prompt = f"/no_think {wiki_prompt}"
            logger.info(
                f"Wiki page prompt built server-side for: "
                f"{request.page_title} "
                f"(commit={commit_hash[:8] if commit_hash else 'none'}, "
                f"context={len(context_text)} chars)"
            )
        else:
            # --- Standard chat/Q&A prompt assembly ---

            # Create system prompt based on research mode
            system_prompt = build_chat_system_prompt(
                is_deep_research, research_iteration, repo_type,
                repo_url, repo_name, language_name
            )

            # Fetch file content if provided
            file_content = ""
            if request.filePath:
                try:
                    file_content = get_file_content(
                        request.repo_url, request.filePath,
                        request.type, request.token
                    )
                    logger.info(
                        f"Retrieved content for file: "
                        f"{request.filePath}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error retrieving file content: {str(e)}"
                    )

            # Format conversation history
            conversation_history = format_conversation_history(
                request_rag.memory()
            )

            # Build the prompt
            prompt = f"/no_think {system_prompt}\n\n"

            if conversation_history:
                prompt += (
                    f"<conversation_history>\n"
                    f"{conversation_history}"
                    f"</conversation_history>\n\n"
                )

            if file_content:
                prompt += (
                    f"<currentFileContent path=\"{request.filePath}\">\n"
                    f"{file_content}\n</currentFileContent>\n\n"
                )

            CONTEXT_START = "<START_OF_CONTEXT>"
            CONTEXT_END = "<END_OF_CONTEXT>"
            if context_text.strip():
                prompt += (
                    f"{CONTEXT_START}\n{context_text}\n"
                    f"{CONTEXT_END}\n\n"
                )
            else:
                logger.info("No context available from RAG")
                prompt += (
                    "<note>Answering without retrieval "
                    "augmentation.</note>\n\n"
                )

            prompt += (
                f"<query>\n{query}\n</query>\n\nAssistant: "
            )

        # Select model based on task: reasoning for deep research, chat for Q&A
        task = 'reasoning' if is_deep_research else 'chat'
        logger.info(f"Using Azure OpenAI task={task} (deep_research={is_deep_research})")

        # Get deployment name for the task
        deployment_name = get_azure_deployment_name(task=task)

        # Get config for the deployment name (includes initialize_kwargs)
        model_config = get_model_config("azure", deployment_name, task=task)
        deployment_config = model_config["model_kwargs"]
        logger.info(f"Azure deployment_config: {deployment_config}")

        # Use shared Azure AI client instance (singleton)
        model = get_azure_ai_client(task=task)

        # Check if this is an o-series reasoning model (o1, o3, o4, etc.)
        is_reasoning_model = deployment_name.startswith("o") and len(deployment_name) > 1 and deployment_name[1].isdigit()

        # Get temperature from deployment config
        temperature = deployment_config.get("temperature", 1.0)

        model_kwargs = {
            "model": deployment_name,
            "stream": True,
            "temperature": temperature,
        }
        # Only add top_p if it exists (reasoning models don't support it)
        if "top_p" in deployment_config:
            model_kwargs["top_p"] = deployment_config["top_p"]

        # Add max_completion_tokens for o-series reasoning models (o1-mini, o4-mini, etc.)
        # These models require max_completion_tokens instead of max_tokens
        # Default to 16384 to allow for comprehensive wiki structure generation
        if is_reasoning_model:
            model_kwargs["max_completion_tokens"] = deployment_config.get(
                "max_completion_tokens", 16384
            )
            logger.info(f"Reasoning model {deployment_name}: max_completion_tokens={model_kwargs['max_completion_tokens']}")

        # Debug: Log model_kwargs before conversion
        debug_model_kwargs = {k: v for k, v in model_kwargs.items() if k != 'messages'}
        logger.debug(f"model_kwargs before conversion: {debug_model_kwargs}")

        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )

        # Debug: Log api_kwargs after conversion (exclude messages for brevity)
        debug_api_kwargs = {k: v for k, v in api_kwargs.items() if k != 'messages'}
        debug_api_kwargs['messages_count'] = len(api_kwargs.get('messages', []))
        logger.debug(f"api_kwargs after conversion: {debug_api_kwargs}")

        # Process Azure response
        try:
            import time as _time
            _stream_start = _time.time()
            logger.info("Making Azure AI API call")

            # Send commit hash metadata before streaming (wiki pages only)
            if request.wiki_page_request and commit_hash:
                if not await _safe_send(
                    websocket,
                    f"<!-- meta:commit_hash={commit_hash} -->"
                ):
                    return

            # Start a keepalive task that pings every 15s while waiting
            # for the LLM to respond. Reasoning models (o4-mini) can
            # take 30-130s before the first chunk — without pings the
            # browser closes the idle WebSocket on inactive tabs.
            _llm_keepalive_active = True
            _llm_ping_count = 0

            async def _llm_keepalive():
                nonlocal _llm_ping_count
                while _llm_keepalive_active:
                    await asyncio.sleep(15)
                    if not _llm_keepalive_active:
                        break
                    _llm_ping_count += 1
                    logger.debug(
                        f"Sending LLM keepalive ping #{_llm_ping_count}"
                    )
                    if not await _safe_send(
                        websocket,
                        f"<!-- llm-keepalive {_llm_ping_count} -->"
                    ):
                        break

            keepalive_task = asyncio.create_task(_llm_keepalive())

            try:
                response = await model.acall(
                    api_kwargs=api_kwargs, model_type=ModelType.LLM
                )
            finally:
                # Stop keepalive once we have the response iterator
                _llm_keepalive_active = False
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass
                if _llm_ping_count > 0:
                    logger.info(
                        f"LLM responded after {_llm_ping_count} "
                        "keepalive pings"
                    )
            # Handle streaming response from Azure AI
            total_text = ""
            chunk_count = 0
            finish_reason = None
            content_filter_category = None
            _client_connected = True
            async for chunk in response:
                if not _client_connected:
                    break
                chunk_count += 1
                choices = getattr(chunk, "choices", [])
                if len(choices) > 0:
                    choice = choices[0]
                    # Track finish_reason
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason
                    # Capture content filter annotations for debugging
                    cfr = getattr(choice, "content_filter_results", None)
                    if cfr:
                        for cat in ('profanity', 'hate', 'sexual',
                                    'violence', 'self_harm'):
                            entry = getattr(cfr, cat, None)
                            if entry and getattr(entry, 'filtered', False):
                                content_filter_category = cat
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        text = getattr(delta, "content", None)
                        if text is not None:
                            total_text += text
                            if not await _safe_send(websocket, text):
                                _client_connected = False
                                break
            _stream_elapsed = _time.time() - _stream_start
            logger.info(
                f"Streaming complete: {chunk_count} chunks, "
                f"{len(total_text)} chars, "
                f"finish_reason={finish_reason}, "
                f"elapsed={_stream_elapsed:.1f}s"
            )

            # If client already disconnected during streaming,
            # skip post-stream sends and just clean up.
            if not _client_connected:
                logger.info(
                    "Client disconnected during streaming "
                    f"({chunk_count} chunks, {len(total_text)} chars sent "
                    "before disconnect)"
                )
                await _safe_close(websocket)
                return

            # Content filter finish reason — signal the frontend so it
            # can handle truncated content (e.g. repair XML, use partial).
            # (ref: handling_embedder_ref.md — log warning, skip & continue)
            if finish_reason == "content_filter":
                filter_detail = (
                    f"category={content_filter_category}"
                    if content_filter_category else "category=unknown"
                )
                logger.warning(
                    "Response truncated by content filter "
                    f"({len(total_text)} chars received, "
                    f"{filter_detail}). "
                    "Sending WARNING marker to frontend."
                )

                # For wiki structure requests, retry with a
                # directory-only file tree so the LLM doesn't
                # reference file names that trigger the filter.
                if (request.wiki_structure_request
                        and not getattr(request, '_retry_attempted',
                                        False)):
                    logger.info(
                        "Retrying wiki structure with "
                        "directory-only file tree..."
                    )
                    request._retry_attempted = True
                    dir_tree = _file_tree_dirs_only(
                        request.file_tree or ''
                    )
                    request.file_tree = dir_tree
                    request.readme = (
                        '(README omitted for content safety)'
                    )
                    parts = (request.repo_url or '').rstrip('/').split('/')
                    r_owner = parts[-2] if len(parts) >= 2 else 'unknown'
                    r_repo = parts[-1] if len(parts) >= 1 else 'unknown'
                    retry_prompt = build_wiki_structure_prompt(
                        file_tree=dir_tree,
                        readme='(README omitted for content safety)',
                        owner=r_owner,
                        repo=r_repo,
                        language=request.language or 'en',
                        comprehensive=request.comprehensive or True,
                    )
                    request.messages[-1].content = retry_prompt
                    retry_kwargs = model.convert_inputs_to_api_kwargs(
                        input=f"/no_think {retry_prompt}",
                        model_kwargs=model_kwargs,
                        model_type=ModelType.LLM,
                    )
                    retry_response = await model.acall(
                        api_kwargs=retry_kwargs,
                        model_type=ModelType.LLM,
                    )
                    async for rchunk in retry_response:
                        rchoices = getattr(rchunk, "choices", [])
                        if rchoices:
                            rdelta = getattr(
                                rchoices[0], "delta", None
                            )
                            if rdelta:
                                rtext = getattr(
                                    rdelta, "content", None
                                )
                                if rtext:
                                    if not await _safe_send(
                                        websocket, rtext
                                    ):
                                        break
                    await _safe_close(websocket)
                    return  # Skip the WARNING marker

                await _safe_send(
                    websocket, "\n[CONTENT_FILTER_WARNING]"
                )

            await _safe_close(websocket)
        except Exception as e_azure:
            # Extract APIM request ID for debugging
            from backend.clients.azureai_client import (
                _extract_request_id, _mask_secrets,
            )
            import traceback
            req_id = _extract_request_id(e_azure)
            is_connection_error = 'Connection' in type(e_azure).__name__
            req_label = 'N/A (connection failed)' if is_connection_error else req_id
            logger.error(_mask_secrets(
                f"Error with Azure AI API (req_id={req_label}): "
                f"{type(e_azure).__name__}: {str(e_azure)}"
            ))
            if is_connection_error:
                logger.error(_mask_secrets(
                    f"Connection error details — "
                    f"exception_type={type(e_azure).__name__}, "
                    f"cause={type(e_azure.__cause__).__name__ if e_azure.__cause__ else 'None'}, "
                    f"cause_detail={str(e_azure.__cause__) if e_azure.__cause__ else 'N/A'}"
                ))
                logger.debug(_mask_secrets(
                    f"Full traceback:\n{traceback.format_exc()}"
                ))
            error_message = str(e_azure).lower()

            # Check for content filter errors — signal frontend
            # (ref: handling_embedder_ref.md pattern)
            if ("content_filter" in error_message
                    or "content_management_policy" in error_message
                    or "ChatCompletionFailed" in type(e_azure).__name__):
                logger.warning(
                    f"Content filter triggered: {e_azure}. "
                    "Sending WARNING marker to frontend."
                )
                await _safe_send(
                    websocket, "\n[CONTENT_FILTER_WARNING]"
                )
                await _safe_close(websocket)
            # Check for token limit errors
            elif ("maximum context length" in error_message or
                    "token limit" in error_message or
                    "too many tokens" in error_message):
                logger.warning("Token limit exceeded, retrying without context")
                await _handle_fallback_request(
                    websocket, model, model_kwargs, system_prompt,
                    conversation_history, request, file_content, query
                )
            else:
                error_msg = (
                    f"\nError with Azure AI API (request_id: {req_label}): "
                    f"{str(e_azure)}\n\n"
                    "Please check your AZURE_OPENAI_API_KEY, "
                    "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_VERSION."
                )
                await _safe_send(websocket, error_msg)
                await _safe_close(websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        import traceback
        logger.error(
            f"Error in WebSocket handler: {str(e)}\n"
            f"{traceback.format_exc()}"
        )
        await _safe_send(websocket, f"Error: {str(e)}")
        await _safe_close(websocket)


async def _handle_fallback_request(
    websocket, model, model_kwargs, system_prompt,
    conversation_history, request, file_content, query
):
    """Handle fallback request when token limit is exceeded."""
    try:
        simplified_prompt = f"/no_think {system_prompt}\n\n"
        if conversation_history:
            simplified_prompt += (
                f"<conversation_history>\n{conversation_history}"
                f"</conversation_history>\n\n"
            )

        if request.filePath and file_content:
            simplified_prompt += (
                f"<currentFileContent path=\"{request.filePath}\">\n"
                f"{file_content}\n</currentFileContent>\n\n"
            )

        simplified_prompt += (
            "<note>Answering without retrieval augmentation "
            "due to input size constraints.</note>\n\n"
        )
        simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
            input=simplified_prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )

        logger.info("Making fallback Azure AI API call")
        fallback_response = await model.acall(
            api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM
        )

        finish_reason = None
        async for chunk in fallback_response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                choice = choices[0]
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        if not await _safe_send(websocket, text):
                            break

        if finish_reason == "content_filter":
            logger.warning(
                "Fallback response truncated by content filter. "
                "Sending WARNING marker to frontend."
            )
            await _safe_send(websocket, "\n[CONTENT_FILTER_WARNING]")

        await _safe_close(websocket)
    except Exception as e_fallback:
        logger.error(f"Error with Azure AI API fallback: {str(e_fallback)}")
        error_message = str(e_fallback).lower()

        # Content filter — signal frontend
        if ("content_filter" in error_message
                or "content_management_policy" in error_message
                or "ChatCompletionFailed" in type(e_fallback).__name__):
            logger.warning(
                f"Content filter triggered in fallback: {e_fallback}. "
                "Sending WARNING marker to frontend."
            )
            await _safe_send(
                websocket, "\n[CONTENT_FILTER_WARNING]"
            )
            await _safe_close(websocket)
        else:
            error_msg = (
                f"\nError with Azure AI API fallback: {str(e_fallback)}\n\n"
                "Please check your Azure OpenAI configuration."
            )
            await _safe_send(websocket, error_msg)
            await _safe_close(websocket)
