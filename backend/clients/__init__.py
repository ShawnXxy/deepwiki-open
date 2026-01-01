# Backend clients module
# Contains client implementations for external services

from backend.clients.azureai_client import AzureAIClient, AzureBatchEmbedder, AzureToEmbeddings
from backend.clients.blob_client import (
    AzureBlobStorageClient,
    get_blob_storage_client,
    is_blob_storage_configured
)

__all__ = [
    "AzureAIClient",
    "AzureBatchEmbedder",
    "AzureToEmbeddings",
    "AzureBlobStorageClient",
    "get_blob_storage_client",
    "is_blob_storage_configured",
]
