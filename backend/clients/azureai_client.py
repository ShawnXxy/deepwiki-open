"""AzureOpenAI ModelClient integration."""

import os
import time
import re
import asyncio
from typing import (
    Dict,
    Sequence,
    Optional,
    List,
    Any,
    TypeVar,
    Callable,
    Generator,
    Union,
    Literal,
)

import logging

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

import sys

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])
# Importing all Azure packages together
azure_modules = safe_import(
    OptionalPackages.AZURE.value[0],  # List of package names
    OptionalPackages.AZURE.value[1],  # Error message
)
# Manually add each module to sys.modules to make them available globally as if imported normally
azure_module_names = OptionalPackages.AZURE.value[0]
for name, module in zip(azure_module_names, azure_modules):
    sys.modules[name] = module

# Use the modules as if they were imported normally
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# from azure.core.credentials import AccessToken
from openai import AzureOpenAI, AsyncAzureOpenAI, Stream
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    Embedding,
    CompletionUsage,
    GeneratorOutput,
)
from adalflow.components.model_client.utils import parse_embedding_response

log = logging.getLogger(__name__)
T = TypeVar("T")


def _extract_request_id(error_or_response) -> str:
    """Extract APIM request ID from an OpenAI error or response.

    Works with:
    - OpenAI API errors (APIStatusError subclasses) → e.response.headers
    - OpenAI response objects (ChatCompletion, etc.) → response._response.headers
    - Stream objects → stream.response.headers
    Returns 'unknown' if not found.
    """
    headers = None
    try:
        # Error objects: e.response.headers
        if hasattr(error_or_response, 'response') and error_or_response.response is not None:
            resp = error_or_response.response
            if hasattr(resp, 'headers'):
                headers = resp.headers
        # Response objects: response._response.headers
        elif hasattr(error_or_response, '_response'):
            if hasattr(error_or_response._response, 'headers'):
                headers = error_or_response._response.headers
    except Exception:
        pass

    if headers:
        return (headers.get('apim-request-id')
                or headers.get('x-ms-client-request-id')
                or headers.get('x-request-id')
                or 'unknown')
    return 'unknown'


def _mask_secrets(text: str) -> str:
    """Mask API keys, bearer tokens, and PATs in log messages.

    Prevents accidental credential exposure in exception tracebacks.
    Masks any string that looks like a credential (>20 chars of
    alphanumeric after 'Bearer ', 'api_key=', or similar patterns).
    """
    # Mask Bearer tokens: "Bearer ABCDEF..." → "Bearer ABCDEF***"
    text = re.sub(
        r"(Bearer\s+)([A-Za-z0-9+/=]{6})([A-Za-z0-9+/=]{10,})",
        r"\1\2***",
        text,
    )
    # Mask api_key values: "api_key='ABCDEF...'" → "api_key='ABCDEF***'"
    text = re.sub(
        r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9]{6})([A-Za-z0-9]{10,})",
        r"\1\2***",
        text,
    )
    # Mask raw long tokens (>30 chars of base64-ish after b')
    text = re.sub(
        r"(b['\"])([A-Za-z0-9+/=]{6})([A-Za-z0-9+/=]{20,})",
        r"\1\2***",
        text,
    )
    return text


__all__ = ["AzureAIClient"]

# TODO: this overlaps with openai client largely, might need to refactor to subclass openai client to simplify the code


# completion parsing functions and you can combine them into one singple chat completion parser
def get_first_message_content(completion: ChatCompletion) -> str:
    r"""When we only need the content of the first message.
    It is the default parser for chat completion."""
    return completion.choices[0].message.content


# def _get_chat_completion_usage(completion: ChatCompletion) -> OpenAICompletionUsage:
#     return completion.usage


def parse_stream_response(completion: ChatCompletionChunk) -> str:
    r"""Parse the response of the stream API."""
    return completion.choices[0].delta.content


def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    r"""Handle the streaming response."""
    for completion in generator:
        parsed_content = parse_stream_response(completion)
        yield parsed_content


def parse_azure_rate_limit_error(error_message: str) -> Optional[int]:
    """
    Parse Azure OpenAI rate limit error message to extract retry delay.
    
    Args:
        error_message: The error message from Azure OpenAI
        
    Returns:
        int: Number of seconds to wait, or None if not a rate limit error
    """
    # Pattern for "Please retry after X seconds"
    retry_pattern = r"Please retry after (\d+) seconds"
    match = re.search(retry_pattern, error_message)
    if match:
        return int(match.group(1))
    return None


def azure_openai_retry_with_delay(func):
    """
    Decorator to handle Azure OpenAI rate limiting with intelligent delays.
    
    This decorator:
    1. Catches RateLimitError exceptions
    2. Parses the error message to extract the required delay
    3. Waits the specified time before retrying
    4. Falls back to exponential backoff for other errors
    """
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                retry_count += 1
                error_message = str(e)
                req_id = _extract_request_id(e)
                
                # Try to parse the required delay from error message
                retry_delay = parse_azure_rate_limit_error(error_message)
                
                if retry_delay is not None and retry_count < max_retries:
                    log.warning(f"Azure OpenAI rate limit hit (req_id={req_id}). "
                                f"Waiting {retry_delay} seconds before retry "
                                f"({retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    # If we can't parse delay or max retries reached, re-raise
                    if retry_count >= max_retries:
                        log.error(f"Max retries ({max_retries}) reached "
                                  f"for rate limit error (req_id={req_id})")
                    else:
                        log.warning("Could not parse retry delay from "
                                    f"error message (req_id={req_id})")
                    raise
            except (APITimeoutError, InternalServerError,
                    UnprocessableEntityError, BadRequestError) as e:
                req_id = _extract_request_id(e)
                # Content filter errors are permanent — skip retry
                error_msg = str(e).lower()
                if isinstance(e, BadRequestError) and any(
                    kw in error_msg for kw in [
                        "content_filter",
                        "content management policy",
                        "content filtering",
                        "responsibleaipolicy",
                    ]
                ):
                    log.warning(
                        "Content filter error (non-retryable), "
                        f"raising immediately (req_id={req_id}): {e}"
                    )
                    raise
                # For other errors, use simple exponential backoff
                if retry_count < max_retries - 1:
                    retry_count += 1
                    delay = 2 ** retry_count
                    log.warning(f"API error: {type(e).__name__} (req_id={req_id}). "
                                f"Retrying in {delay} seconds "
                                f"({retry_count}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise
            except APIConnectionError as e:
                log.error(_mask_secrets(
                    f"Connection error to Azure OpenAI: {e}. "
                    f"Cause: {type(e.__cause__).__name__}: {e.__cause__}"
                    if e.__cause__ else f"Connection error: {e}"
                ))
                raise
        
        # This should not be reached, but just in case
        return func(*args, **kwargs)
    
    return wrapper


def azure_openai_async_retry_with_delay(func):
    """
    Async version of the Azure OpenAI retry decorator.
    """
    async def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                retry_count += 1
                error_message = str(e)
                req_id = _extract_request_id(e)
                
                # Try to parse the required delay from error message
                retry_delay = parse_azure_rate_limit_error(error_message)
                
                if retry_delay is not None and retry_count < max_retries:
                    log.warning(f"Azure OpenAI rate limit hit (req_id={req_id}). "
                                f"Waiting {retry_delay} seconds before retry "
                                f"({retry_count}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # If we can't parse delay or max retries reached, re-raise
                    if retry_count >= max_retries:
                        log.error(f"Max retries ({max_retries}) reached "
                                  f"for rate limit error (req_id={req_id})")
                    else:
                        log.warning("Could not parse retry delay from "
                                    f"error message (req_id={req_id})")
                    raise
            except (APITimeoutError, InternalServerError,
                    UnprocessableEntityError, BadRequestError) as e:
                req_id = _extract_request_id(e)
                # Content filter errors are permanent — skip retry
                error_msg = str(e).lower()
                if isinstance(e, BadRequestError) and any(
                    kw in error_msg for kw in [
                        "content_filter",
                        "content management policy",
                        "content filtering",
                        "responsibleaipolicy",
                    ]
                ):
                    log.warning(
                        "Content filter error (non-retryable), "
                        f"raising immediately (req_id={req_id}): {e}"
                    )
                    raise
                # For other errors, use simple exponential backoff
                if retry_count < max_retries - 1:
                    retry_count += 1
                    delay = 2 ** retry_count
                    log.warning(f"API error: {type(e).__name__} (req_id={req_id}). "
                                f"Retrying in {delay} seconds "
                                f"({retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
            except APIConnectionError as e:
                log.error(_mask_secrets(
                    f"Async connection error to Azure OpenAI: {e}. "
                    f"Cause: {type(e.__cause__).__name__}: {e.__cause__}"
                    if e.__cause__ else f"Async connection error: {e}"
                ))
                raise
        
        # This should not be reached, but just in case
        return await func(*args, **kwargs)
    
    return wrapper


class AzureAIClient(ModelClient):
    __doc__ = r"""
    A client wrapper for interacting with Azure OpenAI's API.

    This class provides support for both embedding and chat completion API calls.
    Users can use this class to simplify their interactions with Azure OpenAI models
    through the `Embedder` and `Generator` components.

    **Initialization:**

    You can initialize the `AzureAIClient` with either an API key or Azure Active Directory (AAD) token
    authentication. It is recommended to set environment variables for sensitive data like API keys.

    Args:
        api_key (Optional[str]): Azure OpenAI API key. Default is None.
        api_version (Optional[str]): API version to use. Default is None.
        azure_endpoint (Optional[str]): Azure OpenAI endpoint URL. Default is None.
        credential (Optional[DefaultAzureCredential]): Azure AD credential for token-based authentication. Default is None.
        chat_completion_parser (Callable[[Completion], Any]): Function to parse chat completions. Default is `get_first_message_content`.
        input_type (Literal["text", "messages"]): Format for input, either "text" or "messages". Default is "text".

    **Setup Instructions:**

    - **Using API Key:**
      Set up the following environment variables:
      ```bash
      export AZURE_OPENAI_API_KEY="your_api_key"
      export AZURE_OPENAI_ENDPOINT="your_endpoint"
      export AZURE_OPENAI_VERSION="your_version"
      ```

    - **Using Azure AD Token:**
      Ensure you have configured Azure AD credentials. The `DefaultAzureCredential` will automatically use your configured credentials.

    **Example Usage:**

    .. code-block:: python

        from azure.identity import DefaultAzureCredential
        from your_module import AzureAIClient  # Adjust import based on your module name

        # Initialize with API key
        client = AzureAIClient(
            api_key="your_api_key",
            api_version="2023-05-15",
            azure_endpoint="https://your-endpoint.openai.azure.com/"
        )

        # Or initialize with Azure AD token
        client = AzureAIClient(
            api_version="2023-05-15",
            azure_endpoint="https://your-endpoint.openai.azure.com/",
            credential=DefaultAzureCredential()
        )

        # Example call to the chat completion API
        api_kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
            "stream": True
        }
        response = client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)

        for chunk in response:
            print(chunk)


    **Notes:**
    - Ensure that the API key or credentials are correctly set up and accessible to avoid authentication errors.
    - Use `chat_completion_parser` to define how to extract and handle the chat completion responses.
    - The `input_type` parameter determines how input is formatted for the API call.

    **References:**
    - [Azure OpenAI API Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)
    - [OpenAI API Documentation](https://platform.openai.com/docs/guides/text-generation)
    """

    def __init__(
        self,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        managed_identity_client_id: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
    ):
        r"""Initializes the Azure OpenAI client with Managed Identity (MSI) authentication.

        Args:
            api_version: Azure OpenAI API version.
            azure_endpoint: Azure OpenAI endpoint.
            managed_identity_client_id: The client ID of the managed identity to use.
            chat_completion_parser: Function to parse chat completions.
            input_type: Input format, either "text" or "messages".

        """
        super().__init__()

        # added api_type azure for azure Ai
        self.api_type = "azure"
        self._apiversion = api_version
        self._azure_endpoint = azure_endpoint
        self._managed_identity_client_id = managed_identity_client_id
        self.sync_client = self.init_sync_client()
        self.async_client = None  # only initialize if the async call is called
        self.chat_completion_parser = (
            chat_completion_parser or get_first_message_content
        )
        self._input_type = input_type

    def _get_credential(self) -> DefaultAzureCredential:
        """
        Get Azure credential with fallback chain.
        
        Fallback order:
        1. MSI with explicit client_id (Azure Container Apps)
        2. DefaultAzureCredential (includes MSI, Azure CLI, VS Code, etc.)
        """
        client_id = self._managed_identity_client_id
        if client_id:
            log.info(f"🔐 [Auth] Using Managed Identity with "
                     f"client_id: {client_id[:8]}...")
            return DefaultAzureCredential(managed_identity_client_id=client_id)
        else:
            log.info("🔐 [Auth] Using DefaultAzureCredential "
                     "(MSI → Azure CLI → VS Code → Environment)")
            return DefaultAzureCredential()

    def init_sync_client(self):
        """
        Initialize sync Azure OpenAI client.
        
        Authentication fallback chain:
        1. API Key from environment (AZURE_OPENAI_API_KEY) - Local Docker
        2. MSI with client_id (Azure Container Apps)
        3. DefaultAzureCredential (Local terminal with Azure CLI)
        """
        azure_endpoint = self._azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = self._apiversion or os.getenv("AZURE_OPENAI_VERSION")
        api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip() or None
        
        log.info("🔧 [AzureOpenAI] Initializing sync client...")
        log.info(f"   Endpoint: {azure_endpoint}")
        log.info(f"   API Version: {api_version}")
        
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
        if not api_version:
            raise ValueError("AZURE_OPENAI_VERSION must be set")

        # Authentication chain: API Key → MSI/DefaultAzureCredential
        if api_key:
            masked = api_key[:6] + '***' if len(api_key) > 6 else '***'
            log.info(f"🔑 [AzureOpenAI] Auth method: API Key ({masked})")
            return AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            log.info("🔐 [AzureOpenAI] Auth method: Azure Identity "
                     "(MSI/CLI fallback)")
            credential = self._get_credential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            return AzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )

    def init_async_client(self):
        """
        Initialize async Azure OpenAI client.
        
        Authentication fallback chain:
        1. API Key from environment (AZURE_OPENAI_API_KEY) - Local Docker
        2. MSI with client_id (Azure Container Apps)
        3. DefaultAzureCredential (Local terminal with Azure CLI)
        """
        azure_endpoint = self._azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = self._apiversion or os.getenv("AZURE_OPENAI_VERSION")
        api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip() or None
        
        log.info("🔧 [AzureOpenAI] Initializing async client...")
        
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
        if not api_version:
            raise ValueError("AZURE_OPENAI_VERSION must be set")

        # Authentication chain: API Key → MSI/DefaultAzureCredential
        if api_key:
            masked = api_key[:6] + '***' if len(api_key) > 6 else '***'
            log.info(f"🔑 [AzureOpenAI Async] Auth method: API Key ({masked})")
            return AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            log.info("🔐 [AzureOpenAI Async] Auth method: Azure Identity")
            credential = self._get_credential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            return AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )

    # def _parse_chat_completion(self, completion: ChatCompletion) -> "GeneratorOutput":
    #     # TODO: raw output it is better to save the whole completion as a source of truth instead of just the message
    #     try:
    #         data = self.chat_completion_parser(completion)
    #         usage = self.track_completion_usage(completion)
    #         return GeneratorOutput(
    #             data=data, error=None, raw_response=str(data), usage=usage
    #         )
    #     except Exception as e:
    #         log.error(f"Error parsing the completion: {e}")
    #         return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """Parse the completion, and put it into the raw_response."""
        log.debug(f"completion type: {type(completion).__name__}, parser: {self.chat_completion_parser.__name__ if self.chat_completion_parser else None}")
        try:
            data = self.chat_completion_parser(completion)
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=data, usage=usage
            )
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        if isinstance(completion, ChatCompletion):
            usage: CompletionUsage = CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            return usage
        else:
            raise NotImplementedError(
                "streaming completion usage tracking is not implemented"
            )

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        r"""Parse the embedding response to a structure AdalFlow components can understand.

        Should be called in ``Embedder``.
        """
        try:
            log.debug(f"Parsing embedding response type: {type(response)}")
            # Manual parsing to ensure compatibility with OpenAI v1+ response objects
            if hasattr(response, 'data'):
                embeddings = []
                for idx, item in enumerate(response.data):
                    # Extract the raw embedding vector
                    if hasattr(item, 'embedding'):
                        embedding_vector = item.embedding
                        embedding_index = getattr(item, 'index', idx)
                    elif isinstance(item, dict) and 'embedding' in item:
                        embedding_vector = item['embedding']
                        embedding_index = item.get('index', idx)
                    elif isinstance(item, list):
                        # Item itself is the embedding vector (list of floats)
                        embedding_vector = item
                        embedding_index = idx
                    else:
                        log.warning(f"Unknown embedding item type at index {idx}: {type(item)}")
                        continue
                    
                    # Wrap in Embedding dataclass as expected by adalflow
                    embeddings.append(Embedding(embedding=embedding_vector, index=embedding_index))
                
                log.debug(f"Extracted {len(embeddings)} embeddings. First embedding length: {len(embeddings[0].embedding) if embeddings else 0}")
                return EmbedderOutput(data=embeddings, error=None, raw_response=response)
            
            # Fallback to adalflow's parser if it's not a standard object
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Specify the API input type and output api_kwargs that will be used in _call and _acall methods.
        Convert the Component's standard input, and system_input(chat model) and model_kwargs into API-specific format
        """

        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            # convert input to input
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text")
            final_model_kwargs["input"] = input
        elif model_type == ModelType.LLM:
            # convert input to messages
            messages: List[Dict[str, str]] = []

            if self._input_type == "messages":
                system_start_tag = "<START_OF_SYSTEM_PROMPT>"
                system_end_tag = "<END_OF_SYSTEM_PROMPT>"
                user_start_tag = "<START_OF_USER_PROMPT>"
                user_end_tag = "<END_OF_USER_PROMPT>"
                pattern = f"{system_start_tag}(.*?){system_end_tag}{user_start_tag}(.*?){user_end_tag}"
                # Compile the regular expression
                regex = re.compile(pattern)
                # Match the pattern
                match = regex.search(input)
                system_prompt, input_str = None, None

                if match:
                    system_prompt = match.group(1)
                    input_str = match.group(2)

                else:
                    print("No match found.")
                if system_prompt and input_str:
                    messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": input_str})
            if len(messages) == 0:
                messages.append({"role": "system", "content": input})
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    @azure_openai_retry_with_delay
    def call(self, api_kwargs: Dict = {},
             model_type: ModelType = ModelType.UNDEFINED):
        """
        kwargs is the combined input and model_kwargs.  Support streaming call.
        """
        # Log api_kwargs summary without full message/input content
        try:
            debug_kwargs = {}
            for k, v in api_kwargs.items():
                if k == 'messages':
                    debug_kwargs[k] = f"[{len(v)} messages]"
                elif k == 'input':
                    if isinstance(v, list):
                        debug_kwargs[k] = f"[{len(v)} texts]"
                    else:
                        debug_kwargs[k] = f"[{len(str(v))} chars]"
                else:
                    debug_kwargs[k] = v
            log.debug(f"api_kwargs: {debug_kwargs}")
        except Exception as e:
            log.debug(f"api_kwargs logging failed: {str(e)}")
        
        if model_type == ModelType.EMBEDDER:
            try:
                result = self.sync_client.embeddings.create(**api_kwargs)
                # Use response object's built-in ID for logging
                log.debug("Embedding call succeeded")
                return result
            except Exception as e:
                req_id = _extract_request_id(e)
                log.critical(
                    _mask_secrets(f"CRITICAL: Azure Embedding Failed (req_id={req_id}): {e}")
                )
                raise e
        elif model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                self.chat_completion_parser = handle_streaming_response
                return self.sync_client.chat.completions.create(**api_kwargs)
            result = self.sync_client.chat.completions.create(**api_kwargs)
            # ChatCompletion.id is the OpenAI request ID (e.g. "chatcmpl-...")
            completion_id = getattr(result, 'id', 'unknown')
            log.debug(f"LLM call succeeded (completion_id={completion_id})")
            return result
        else:
            raise ValueError(f"model_type {model_type} is not supported")

