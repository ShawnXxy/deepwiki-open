"""
Embedder module for Azure OpenAI embeddings.
"""

import adalflow as adal

from backend.config import configs

# Cached embedder instance (singleton)
_embedder: adal.Embedder = None


def get_embedder(embedder_type: str = None) -> adal.Embedder:
    """
    Get an Azure OpenAI embedder instance (singleton pattern).

    Args:
        embedder_type: Ignored, kept for backward compatibility. Always uses Azure.

    Returns:
        adal.Embedder: Configured embedder for Azure OpenAI
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

    _embedder = adal.Embedder(**embedder_kwargs)

    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        _embedder.batch_size = embedder_config["batch_size"]

    return _embedder
