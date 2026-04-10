"""
Azure OpenAI Embedding Client Factory.

Creates and configures the embedding client with token-safety guardrails.
Wraps adalflow.Embedder in SafeEmbedder which auto-splits oversized texts.

Usage:
    from backend.clients.embedding_client import get_embedder
    embedder = get_embedder()  # Singleton, configured from infra.json
    result = embedder(input=["some text"])
"""

import logging
import tiktoken

import adalflow as adal

from backend.config import configs

logger = logging.getLogger(__name__)

# Cached embedder instance (singleton)
_embedder: adal.Embedder = None

# Maximum token limit for Azure OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192
SAFE_EMBEDDING_TOKENS = 7500  # Safe threshold below the limit


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int = SAFE_EMBEDDING_TOKENS) -> str:
    """
    Truncate text to fit within the token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens allowed
        
    Returns:
        Truncated text that fits within the limit
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Truncate and decode back
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception as e:
        logger.warning(f"Error truncating text: {e}, using character-based truncation")
        # Fallback: estimate ~4 chars per token
        max_chars = max_tokens * 4
        return text[:max_chars]


def split_into_chunks(text: str, max_tokens: int = SAFE_EMBEDDING_TOKENS, overlap: int = 100) -> list:
    """
    Split text into overlapping chunks that fit within token limit.

    This preserves all information instead of truncating.

    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(encoding.decode(chunk_tokens))

            # Move start forward, accounting for overlap
            start = end - overlap if end < len(tokens) else end

            # Prevent infinite loop
            if start >= end:
                break

        return chunks
    except Exception as e:
        logger.warning(f"Error splitting text: {e}, using single truncated chunk")
        return [truncate_to_token_limit(text, max_tokens)]


class SafeEmbedder:
    """
    Wrapper around adalflow Embedder that validates token counts
    and splits oversized texts into chunks to preserve all information.

    Unlike truncation, splitting ensures no data is lost - each chunk
    gets its own embedding, and retrieval can match any chunk.
    
    This wrapper is fully compatible with adalflow's BatchEmbedder which
    requires both __call__ and call() methods.
    """

    def __init__(self, embedder: adal.Embedder):
        self.embedder = embedder
        self.batch_size = getattr(embedder, 'batch_size', 10)

    def _prepare_safe_inputs(self, input, model_kwargs=None):
        """
        Prepare inputs by splitting oversized texts into chunks.
        
        Returns:
            Tuple of (safe_inputs, model_kwargs)
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Handle single string input
        if isinstance(input, str):
            input = [input]

        # Validate and split texts that exceed token limit
        safe_inputs = []
        chunk_map = []  # Track which original text each chunk belongs to

        for i, text in enumerate(input):
            token_count = count_tokens(text)
            if token_count > SAFE_EMBEDDING_TOKENS:
                # Split into chunks instead of truncating
                chunks = split_into_chunks(text, SAFE_EMBEDDING_TOKENS, overlap=200)
                logger.debug(
                    f"Text {i} has {token_count} tokens, split into {len(chunks)} chunks"
                )
                for chunk in chunks:
                    safe_inputs.append(chunk)
                    chunk_map.append(i)
            else:
                safe_inputs.append(text)
                chunk_map.append(i)

        return safe_inputs, model_kwargs

    def __call__(self, input, model_kwargs=None):
        """
        Embed input text(s) with automatic chunking for oversized inputs.

        Oversized texts are split into overlapping chunks, preserving all content.
        Each chunk is embedded separately.
        """
        # Delegate to call() to avoid duplication
        return self.call(input, model_kwargs)

    def call(self, input, model_kwargs=None):
        """
        Embed input text(s) - required by adalflow's BatchEmbedder.
        
        This is the core implementation. Both __call__ and BatchEmbedder use this.
        """
        safe_inputs, model_kwargs = self._prepare_safe_inputs(input, model_kwargs)
        # Use the embedder's call method directly
        return self.embedder.call(input=safe_inputs, model_kwargs=model_kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying embedder for full compatibility."""
        # This is called when an attribute is not found on SafeEmbedder
        return getattr(self.embedder, name)


def get_embedder(embedder_type: str = None) -> SafeEmbedder:
    """
    Get an Azure OpenAI embedder instance (singleton pattern).

    Returns a SafeEmbedder that automatically splits oversized inputs
    into chunks to prevent token limit errors while preserving accuracy.

    Args:
        embedder_type: Ignored, kept for backward compatibility. Always uses Azure.

    Returns:
        SafeEmbedder: Configured embedder wrapper for Azure OpenAI
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    # Always use the default embedder config (Azure)
    embedder_config = configs["embedder"]

    # Use the shared Azure AI client
    from backend.config import get_azure_ai_client
    model_client = get_azure_ai_client()

    # Create embedder with basic parameters
    embedder_kwargs = {
        "model_client": model_client,
        "model_kwargs": embedder_config["model_kwargs"]
    }

    base_embedder = adal.Embedder(**embedder_kwargs)

    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        base_embedder.batch_size = embedder_config["batch_size"]

    # Wrap with SafeEmbedder for automatic token validation
    _embedder = SafeEmbedder(base_embedder)

    return _embedder
