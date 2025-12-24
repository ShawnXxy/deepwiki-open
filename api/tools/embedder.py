"""
Embedder module for Azure OpenAI embeddings.
"""

import adalflow as adal

from api.config import configs
from api.azureai_client import AzureAIClient


def get_embedder(embedder_type: str = None) -> adal.Embedder:
    """
    Get an Azure OpenAI embedder instance.

    Args:
        embedder_type: Ignored, kept for backward compatibility. Always uses Azure.

    Returns:
        adal.Embedder: Configured embedder for Azure OpenAI
    """
    # Always use the default embedder config (Azure)
    embedder_config = configs["embedder"]

    # Initialize model client
    model_client_class = embedder_config.get("model_client")
    if not model_client_class:
        # Fallback to client_class if model_client is not set
        client_class_name = embedder_config.get("client_class")
        if client_class_name == "AzureAIClient":
            model_client_class = AzureAIClient
        else:
            raise ValueError(f"Unknown client class: {client_class_name}")

    # Initialize model client with proper configuration
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()

    # Create embedder with basic parameters
    embedder_kwargs = {
        "model_client": model_client,
        "model_kwargs": embedder_config["model_kwargs"]
    }

    embedder = adal.Embedder(**embedder_kwargs)

    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]

    return embedder
