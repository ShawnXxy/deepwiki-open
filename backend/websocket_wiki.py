"""
WebSocket handler for wiki chat completions.
This module handles WebSocket connections for chat completions using Azure OpenAI.
"""

import logging
from typing import List, Optional
from urllib.parse import unquote

from adalflow.core.types import ModelType
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.config import (
    get_model_config,
    configs,
    get_azure_deployment_name,
)
from backend.data_pipeline import count_tokens, get_file_content
from backend.azureai_client import AzureAIClient
from backend.rag import RAG

# Configure logging
from backend.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Model for a chat message."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatCompletionRequest(BaseModel):
    """Model for requesting a chat completion."""
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(
        None, description="Optional path to a file in the repository"
    )
    token: Optional[str] = Field(
        None, description="Personal access token for private repositories"
    )
    type: Optional[str] = Field(
        "github", description="Type of repository (e.g., 'github', 'gitlab')"
    )
    branch: Optional[str] = Field(
        None, description="Specific branch to clone/process"
    )

    # Model parameters (provider is ignored, always uses Azure)
    provider: str = Field(
        "azure", description="Model provider (always Azure OpenAI)"
    )
    model: Optional[str] = Field(
        None, description="Model name for Azure OpenAI deployment"
    )

    language: Optional[str] = Field(
        "en", description="Language for content generation"
    )
    excluded_dirs: Optional[str] = Field(
        None, description="Comma-separated list of directories to exclude"
    )
    excluded_files: Optional[str] = Field(
        None, description="Comma-separated list of file patterns to exclude"
    )
    included_dirs: Optional[str] = Field(
        None, description="Comma-separated list of directories to include exclusively"
    )
    included_files: Optional[str] = Field(
        None, description="Comma-separated list of file patterns to include"
    )


async def handle_websocket_chat(websocket: WebSocket):
    """
    Handle WebSocket connection for chat completions using Azure OpenAI.
    """
    await websocket.accept()

    try:
        # Receive and parse the request data
        request_data = await websocket.receive_json()
        request = ChatCompletionRequest(**request_data)

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

        # Only retrieve documents if input is not too large
        context_text = ""
        retrieved_documents = None

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

                    if (retrieved_documents and
                            retrieved_documents[0].documents):
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

                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("No documents retrieved from RAG")
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
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # Create system prompt based on research mode
        system_prompt = _build_system_prompt(
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
        conversation_history = ""
        for turn_id, turn in request_rag.memory().items():
            if (not isinstance(turn_id, int) and
                    hasattr(turn, 'user_query') and
                    hasattr(turn, 'assistant_response')):
                conversation_history += (
                    f"<turn>\n<user>{turn.user_query.query_str}</user>\n"
                    f"<assistant>{turn.assistant_response.response_str}"
                    f"</assistant>\n</turn>\n"
                )

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

        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        logger.info(f"Using Azure OpenAI with model: {request.model}")

        # Get deployment name for Azure
        deployment_name = get_azure_deployment_name(request.model)
        logger.info(f"Using Azure deployment: {deployment_name}")

        # Get config for the deployment name (includes initialize_kwargs)
        model_config = get_model_config("azure", deployment_name)
        deployment_config = model_config["model_kwargs"]
        logger.info(f"Azure deployment_config: {deployment_config}")

        # Initialize Azure AI client with proper configuration
        initialize_kwargs = model_config.get("initialize_kwargs", {})
        model = AzureAIClient(**initialize_kwargs)

        # Get temperature from deployment config
        temperature = deployment_config.get("temperature", 1.0)
        logger.info(f"Azure temperature: {temperature}")

        model_kwargs = {
            "model": deployment_name,
            "stream": True,
            "temperature": temperature,
        }
        # Only add top_p if it exists (reasoning models don't support it)
        if "top_p" in deployment_config:
            model_kwargs["top_p"] = deployment_config["top_p"]

        logger.info(f"Azure model_kwargs: {model_kwargs}")

        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )

        # Process Azure response
        try:
            logger.info("Making Azure AI API call")
            response = await model.acall(
                api_kwargs=api_kwargs, model_type=ModelType.LLM
            )
            # Handle streaming response from Azure AI
            total_text = ""
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                choices = getattr(chunk, "choices", [])
                if len(choices) > 0:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        text = getattr(delta, "content", None)
                        if text is not None:
                            total_text += text
                            await websocket.send_text(text)
            logger.info(f"Streaming complete: {chunk_count} chunks, {len(total_text)} chars")
            await websocket.close()
        except Exception as e_azure:
            logger.error(f"Error with Azure AI API: {str(e_azure)}")
            error_message = str(e_azure)

            # Check for token limit errors
            if ("maximum context length" in error_message or
                    "token limit" in error_message or
                    "too many tokens" in error_message):
                logger.warning("Token limit exceeded, retrying without context")
                await _handle_fallback_request(
                    websocket, model, model_kwargs, system_prompt,
                    conversation_history, request, file_content, query
                )
            else:
                error_msg = (
                    f"\nError with Azure AI API: {str(e_azure)}\n\n"
                    "Please check your AZURE_OPENAI_API_KEY, "
                    "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_VERSION."
                )
                await websocket.send_text(error_msg)
                await websocket.close()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        try:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()
        except Exception:
            pass


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

        async for chunk in fallback_response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        await websocket.send_text(text)
        await websocket.close()
    except Exception as e_fallback:
        logger.error(f"Error with Azure AI API fallback: {str(e_fallback)}")
        error_msg = (
            f"\nError with Azure AI API fallback: {str(e_fallback)}\n\n"
            "Please check your Azure OpenAI configuration."
        )
        await websocket.send_text(error_msg)
        await websocket.close()


def _build_system_prompt(
    is_deep_research, research_iteration, repo_type,
    repo_url, repo_name, language_name
):
    """Build the system prompt based on context."""
    if is_deep_research:
        is_first_iteration = research_iteration == 1
        is_final_iteration = research_iteration >= 5

        if is_first_iteration:
            return f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are conducting a multi-turn Deep Research process to investigate the topic in the user's query.
IMPORTANT: You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the first iteration of a multi-turn research process
- Start your response with "## Research Plan"
- Outline your approach to investigating this specific topic
- If the topic is about a specific file or feature, focus ONLY on that
- End with "## Next Steps" indicating what you'll investigate next
- Do NOT provide a final conclusion yet
- NEVER respond with just "Continue the research"
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting
- Cite specific files and code sections when relevant
</style>"""
        elif is_final_iteration:
            return f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are in the final iteration of a Deep Research process.
IMPORTANT: You MUST respond in {language_name} language.
</role>

<guidelines>
- This is the final iteration of the research process
- CAREFULLY review the entire conversation history
- Synthesize ALL findings into a comprehensive conclusion
- Start with "## Final Conclusion"
- Include specific code references and implementation details
- NEVER respond with "Continue the research"
</guidelines>

<style>
- Be concise but thorough
- Use markdown formatting
- Cite specific files and code sections
- End with actionable insights when appropriate
</style>"""
        else:
            return f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You are currently in iteration {research_iteration} of a Deep Research process.
IMPORTANT: You MUST respond in {language_name} language.
</role>

<guidelines>
- CAREFULLY review the conversation history
- Your response MUST build on previous research iterations
- Identify gaps or areas that need further exploration
- Start your response with "## Research Update {research_iteration}"
- Provide new insights not covered in previous iterations
- NEVER respond with just "Continue the research"
</guidelines>

<style>
- Be concise but thorough
- Focus on providing new information
- Use markdown formatting
- Cite specific files and code sections
</style>"""
    else:
        return f"""<role>
You are an expert code analyst examining the {repo_type} repository: {repo_url} ({repo_name}).
You provide direct, concise, and accurate information about code repositories.
You NEVER start responses with markdown headers or code fences.
IMPORTANT: You MUST respond in {language_name} language.
</role>

<guidelines>
- Answer the user's question directly without ANY preamble or filler phrases
- DO NOT include any rationale, explanation, or extra comments
- Strictly base answers ONLY on existing code or documents
- DO NOT speculate or invent citations
- DO NOT start with preambles like "Okay, here's a breakdown"
- DO NOT start with markdown headers like "## Analysis of..."
- DO NOT start with ```markdown code fences
- DO NOT end your response with ``` closing fences
- JUST START with the direct answer to the question
- Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
- For code analysis, organize your response with clear sections
</guidelines>

<style>
- Use concise, direct language
- Prioritize accuracy over verbosity
- When showing code, include line numbers and file paths when relevant
- Use markdown formatting to improve readability
</style>"""
