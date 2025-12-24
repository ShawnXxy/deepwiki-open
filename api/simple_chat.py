"""
Simple Chat API module for streaming chat completions using Azure OpenAI.
"""

import logging
from typing import List, Optional
from urllib.parse import unquote

from adalflow.core.types import ModelType
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, get_azure_deployment_name
from api.data_pipeline import count_tokens, get_file_content
from api.azureai_client import AzureAIClient
from api.rag import RAG
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT
)

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Simple Chat API",
    description="Simplified API for streaming chat completions using Azure OpenAI"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "github", description="Type of repository"
    )
    branch: Optional[str] = Field(
        None, description="Specific branch to clone/process"
    )

    # Model parameters (provider ignored, always Azure)
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
        None, description="Comma-separated directories to exclude"
    )
    excluded_files: Optional[str] = Field(
        None, description="Comma-separated file patterns to exclude"
    )
    included_dirs: Optional[str] = Field(
        None, description="Comma-separated directories to include exclusively"
    )
    included_files: Optional[str] = Field(
        None, description="Comma-separated file patterns to include"
    )


@app.post("/chat/completions/stream")
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

                    if retrieved_documents and retrieved_documents[0].documents:
                        documents = retrieved_documents[0].documents
                        logger.info(f"Retrieved {len(documents)} documents")

                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

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
        if is_deep_research:
            is_first_iteration = research_iteration == 1
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = DEEP_RESEARCH_FIRST_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name
                )
            elif is_final_iteration:
                system_prompt = DEEP_RESEARCH_FINAL_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name
                )
            else:
                system_prompt = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name,
                    research_iteration=research_iteration
                )
        else:
            system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
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

        if context_text.strip():
            prompt += f"<START_OF_CONTEXT>\n{context_text}\n<END_OF_CONTEXT>\n\n"
        else:
            logger.info("No context available from RAG")
            prompt += "<note>Answering without retrieval augmentation.</note>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        # Get Azure model configuration
        logger.info(f"Using Azure OpenAI with model: {request.model}")

        model = AzureAIClient()
        deployment_name = get_azure_deployment_name(request.model)
        logger.info(f"Using Azure deployment: {deployment_name}")

        deployment_config = get_model_config("azure", deployment_name)["model_kwargs"]
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
        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text is not None:
                        yield text
    except Exception as e:
        logger.error(f"Error with Azure AI API: {str(e)}")
        yield f"\nError with Azure AI API: {str(e)}\n"


async def _error_generator(message: str):
    """Generate an error message."""
    yield message
