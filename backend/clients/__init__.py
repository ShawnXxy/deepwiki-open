# Backend clients module
# Contains client implementations for external services

from backend.clients.azureai_client import (
    AzureAIClient,
    AzureBatchEmbedder,
    AzureToEmbeddings
)
from backend.clients.blob_client import (
    AzureBlobStorageClient,
    get_blob_storage_client,
    is_blob_storage_configured,
    get_storage_mode
)
from backend.clients.storage import storage, StorageClient
from backend.clients.vector_storage import VectorStorage, get_vector_storage

# Note: backend.clients.embedder is NOT imported here to avoid circular
# dependency (embedder imports backend.config which imports backend.clients).
# Use: from backend.clients.embedder import get_embedder

__all__ = [
    "AzureAIClient",
    "AzureBatchEmbedder",
    "AzureToEmbeddings",
    "AzureBlobStorageClient",
    "get_blob_storage_client",
    "is_blob_storage_configured",
    "get_storage_mode",
    "storage",
    "StorageClient",
    "VectorStorage",
    "get_vector_storage",
]
