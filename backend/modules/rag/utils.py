"""
Utility functions for RAG module.

Provides file reading and token counting utilities used across the RAG pipeline.
"""

import logging
import tiktoken

logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192

# Alias for backward compatibility (used in rag.py)
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit


def safe_read_file(file_path: str) -> str:
    """
    Safely read a file with automatic encoding detection.
    Tries multiple encodings to handle files with different encodings (UTF-8, UTF-16, etc.)

    Args:
        file_path (str): Path to the file to read.

    Returns:
        str: The file content as a string.

    Raises:
        UnicodeDecodeError: If no encoding can decode the file.
    """
    # List of encodings to try, in order of preference
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # For non-encoding errors, raise immediately
            raise e

    # If all encodings fail, raise an error
    raise UnicodeDecodeError(
        'all-encodings', b'', 0, 1,
        f"Could not decode file {file_path} with any supported encoding"
    )


def count_tokens(text: str, embedder_type: str = None, is_ollama_embedder: bool = None) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.
        embedder_type (str, optional): Kept for backward compatibility, ignored.
        is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Use OpenAI embedding model encoding for Azure OpenAI
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4
