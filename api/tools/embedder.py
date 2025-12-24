import adalflow as adal

from api.config import configs, is_azure_openai_configured, get_azure_openai_embedding_config
from api.azureai_client import AzureAIClient


def get_embedder(embedder_type: str = None) -> adal.Embedder:
    # Determine which config to use
    if embedder_type == 'ollama':
        embedder_config = configs.get("embedder_ollama", configs["embedder"])
    elif embedder_type == 'google':
        embedder_config = configs.get("embedder_google", configs["embedder"])
    elif embedder_type == 'bedrock':
        embedder_config = configs.get("embedder_bedrock", configs["embedder"])
    else:
        # Default to 'embedder' for openai, azure, or unspecified
        embedder_config = configs["embedder"]

    # --- Initialize Embedder ---
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
    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}
    
    embedder = adal.Embedder(**embedder_kwargs)
    
    # Set batch_size as an attribute if available (not a constructor parameter)
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder
