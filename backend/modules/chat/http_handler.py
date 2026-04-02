"""
HTTP handler for streaming chat completions.

Provides the POST /chat/completions/stream endpoint.
"""

import logging
from urllib.parse import unquote

from adalflow.core.types import ModelType
from fastapi.responses import StreamingResponse

from backend.config import (
    get_model_config, configs, get_azure_deployment_name,
    get_azure_ai_client
)
from backend.modules.embedder import RAG
from backend.modules.embedder.tokenizer import count_tokens
from backend.modules.repository.file_content import get_file_content
from backend.modules.chat.models import ChatCompletionRequest
from backend.modules.chat.service import (
    build_system_prompt,
    get_language_info,
    format_conversation_history,
    format_context_text,
)

logger = logging.getLogger(__name__)


async def chat_completions_stream(request: ChatCompletionRequest):
    """Stream a chat completion response using Azure OpenAI."""
    try:
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

        # Create a new RAG instance (always Azure)
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

            request_rag.prepare_retriever(
                request.repo_url,
                request.type,
                request.token,
                request.branch,
                excluded_dirs,
                excluded_files,
                included_dirs,
                included_files
            )
            logger.info(f"Retriever prepared for {request.repo_url}")
        except ValueError as e:
            if "No valid documents with embeddings found" in str(e):
                logger.error(f"No valid embeddings found: {str(e)}")
                return StreamingResponse(
                    content=_error_generator(
                        "Error: No valid document embeddings found."
                    ),
                    media_type="text/event-stream"
                )
            else:
                import traceback
                logger.error(
                    f"ValueError preparing retriever: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                return StreamingResponse(
                    content=_error_generator(
                        f"Error preparing retriever: {str(e)}"
                    ),
                    media_type="text/event-stream"
                )
        except Exception as e:
            import traceback
            logger.error(
                f"Error preparing retriever: {str(e)}\n{traceback.format_exc()}"
            )
            return StreamingResponse(
                content=_error_generator(f"Error preparing retriever: {str(e)}"),
                media_type="text/event-stream"
            )

        # Validate request
        if not request.messages or len(request.messages) == 0:
            return StreamingResponse(
                content=_error_generator("Error: No messages provided"),
                media_type="text/event-stream"
            )

        last_message = request.messages[-1]
        if last_message.role != "user":
            return StreamingResponse(
                content=_error_generator("Error: Last message must be from the user"),
                media_type="text/event-stream"
            )

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

            if ("continue" in last_message.content.lower() and
                    "research" in last_message.content.lower()):
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        break

                if original_topic:
                    last_message.content = original_topic

        # Get the query from the last message
        query = last_message.content

        # Only retrieve documents if input is not too large
        context_text = ""
        if not input_too_large:
            try:
                rag_query = query
                if request.filePath:
                    rag_query = f"Contexts related to {request.filePath}"
                    logger.info(f"Modified RAG query for file: {request.filePath}")

                try:
                    retrieved_documents = request_rag(
                        rag_query, language=request.language
                    )
                    context_text = format_context_text(retrieved_documents)
                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")

            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
        repo_type = request.type

        # Get language information
        language_code, language_name = get_language_info(request.language)

        # Create system prompt based on research mode
        system_prompt = build_system_prompt(
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
                logger.info(f"Retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")

        # Format conversation history
        conversation_history = format_conversation_history(request_rag.memory())

        # Build the prompt
        prompt = f"/no_think {system_prompt}\n\n"

        if conversation_history:
            prompt += (
                f"<conversation_history>\n{conversation_history}"
                f"</conversation_history>\n\n"
            )

        if file_content:
            prompt += (
                f"<currentFileContent path=\"{request.filePath}\">\n"
                f"{file_content}\n</currentFileContent>\n\n"
            )

        if context_text.strip():
            prompt += f"<START_OF_CONTEXT>\n{context_text}\n<END_OF_CONTEXT>\n\n"
        else:
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        # Select model based on task: reasoning for deep research, chat for Q&A
        task = 'reasoning' if is_deep_research else 'chat'
        logger.info(f"Using Azure OpenAI task={task} (deep_research={is_deep_research})")

        # Use shared Azure AI client instance (singleton)
        model = get_azure_ai_client(task=task)
        deployment_name = get_azure_deployment_name(task=task)
        logger.info(f"Using Azure deployment: {deployment_name}")

        deployment_config = get_model_config("azure", deployment_name, task=task)["model_kwargs"]
        temperature = deployment_config.get("temperature", 1.0)

        model_kwargs = {
            "model": deployment_name,
            "stream": True,
            "temperature": temperature,
        }
        if "top_p" in deployment_config:
            model_kwargs["top_p"] = deployment_config["top_p"]

        logger.info(f"Azure model_kwargs: {model_kwargs}")

        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )

        # Return streaming response
        return StreamingResponse(
            content=_azure_stream_generator(model, api_kwargs),
            media_type="text/event-stream"
        )

    except Exception as e:
        import traceback
        logger.error(
            f"Error in chat_completions_stream: {str(e)}\n{traceback.format_exc()}"
        )
        return StreamingResponse(
            content=_error_generator(f"Error: {str(e)}"),
            media_type="text/event-stream"
        )


async def _azure_stream_generator(model, api_kwargs):
    """Generate streaming response from Azure OpenAI."""
    try:
        logger.info("Making Azure AI API call")
        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        total_chars = 0
        finish_reason = None
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                choice = choices[0]
                # Track finish_reason
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        total_chars += len(text)
                        yield text

        # Content filter finish_reason — signal the frontend
        # (ref: handling_embedder_ref.md — log warning, skip & continue)
        if finish_reason == "content_filter":
            logger.warning(
                "Response truncated by content filter "
                f"({total_chars} chars received). "
                "Sending WARNING marker to frontend."
            )
            yield "\n[CONTENT_FILTER_WARNING]"
    except Exception as e:
        error_message = str(e).lower()
        # Content filter — signal frontend (ref: handling_embedder_ref.md)
        if ("content_filter" in error_message
                or "content_management_policy" in error_message
                or "ChatCompletionFailed" in type(e).__name__):
            logger.warning(
                f"Content filter triggered: {e}. "
                "Sending WARNING marker to frontend."
            )
            yield "\n[CONTENT_FILTER_WARNING]"
        else:
            logger.error(f"Error with Azure AI API: {str(e)}")
            yield f"\nError with Azure AI API: {str(e)}\n"


async def _error_generator(message: str):
    """Generate an error message."""
    yield message
