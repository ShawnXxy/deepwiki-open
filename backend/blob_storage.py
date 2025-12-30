"""
Azure Blob Storage client for DeepWiki.
Handles storage of repository databases and embeddings using MSI authentication.
"""

import io
import os
import pickle
import logging
from typing import Optional, Any

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

from backend.config import get_infra_config, get_managed_identity_client_id

logger = logging.getLogger(__name__)

# Singleton blob client instance
_blob_client: Optional["AzureBlobStorageClient"] = None


class AzureBlobStorageClient:
    """
    Azure Blob Storage client using MSI authentication.
    Stores and retrieves pickled database objects.
    """

    def __init__(self, account_name: str, container_name: str, managed_identity_client_id: Optional[str] = None):
        """
        Initialize the Azure Blob Storage client.

        Args:
            account_name: Azure Storage account name
            container_name: Blob container name
            managed_identity_client_id: Optional MSI client ID for authentication
        """
        self.account_name = account_name
        self.container_name = container_name
        self.account_url = f"https://{account_name}.blob.core.windows.net"
        
        # Create credential with MSI
        if managed_identity_client_id:
            logger.debug(f"Using Managed Identity with client_id: {managed_identity_client_id[:8]}...")
            self.credential = DefaultAzureCredential(
                managed_identity_client_id=managed_identity_client_id
            )
        else:
            logger.debug("Using DefaultAzureCredential without explicit client_id")
            self.credential = DefaultAzureCredential()
        
        # Create blob service client
        self.blob_service_client = BlobServiceClient(
            account_url=self.account_url,
            credential=self.credential
        )
        
        # Ensure container exists
        self._ensure_container_exists()
        
        logger.debug(f"Azure Blob Storage client initialized for account: {account_name}, container: {container_name}")

    def _ensure_container_exists(self) -> None:
        """Create the container if it doesn't exist."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            logger.warning(f"Could not verify/create container (may already exist): {e}")

    def get_container_client(self) -> ContainerClient:
        """Get the container client."""
        return self.blob_service_client.get_container_client(self.container_name)

    def save_pickle(self, blob_name: str, obj: Any) -> bool:
        """
        Save a Python object as a pickled blob.

        Args:
            blob_name: Name of the blob (e.g., "databases/owner_repo.pkl")
            obj: Python object to pickle and save

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Pickle the object to bytes
            data = pickle.dumps(obj)
            
            # Upload to blob storage
            blob_client = self.get_container_client().get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True)
            
            logger.debug(f"Saved pickle to blob: {blob_name} ({len(data)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle to blob {blob_name}: {e}")
            return False

    def load_pickle(self, blob_name: str) -> Optional[Any]:
        """
        Load a pickled object from blob storage.

        Args:
            blob_name: Name of the blob to load

        Returns:
            The unpickled Python object, or None if not found/error
        """
        try:
            blob_client = self.get_container_client().get_blob_client(blob_name)
            
            if not blob_client.exists():
                logger.debug(f"Blob does not exist: {blob_name}")
                return None
            
            # Download and unpickle
            data = blob_client.download_blob().readall()
            obj = pickle.loads(data)
            
            logger.debug(f"Loaded pickle from blob: {blob_name} ({len(data)} bytes)")
            return obj
        except Exception as e:
            logger.error(f"Failed to load pickle from blob {blob_name}: {e}")
            return None

    def exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists.

        Args:
            blob_name: Name of the blob to check

        Returns:
            bool: True if exists, False otherwise
        """
        try:
            blob_client = self.get_container_client().get_blob_client(blob_name)
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Failed to check blob existence {blob_name}: {e}")
            return False

    def delete(self, blob_name: str) -> bool:
        """
        Delete a blob.

        Args:
            blob_name: Name of the blob to delete

        Returns:
            bool: True if deleted (or didn't exist), False on error
        """
        try:
            blob_client = self.get_container_client().get_blob_client(blob_name)
            if blob_client.exists():
                blob_client.delete_blob()
                logger.debug(f"Deleted blob: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete blob {blob_name}: {e}")
            return False

    def upload_text(self, blob_name: str, content: str) -> bool:
        """
        Upload text content to blob storage.

        Args:
            blob_name: Name of the blob (e.g., "wikicache/file.json")
            content: Text content to upload

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            blob_client = self.get_container_client().get_blob_client(blob_name)
            blob_client.upload_blob(content.encode('utf-8'), overwrite=True)
            logger.debug(f"Uploaded text to blob: {blob_name} ({len(content)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to upload text to blob {blob_name}: {e}")
            return False

    def download_text(self, blob_name: str) -> Optional[str]:
        """
        Download text content from blob storage.

        Args:
            blob_name: Name of the blob to download

        Returns:
            Text content as string, or None if not found/error
        """
        try:
            blob_client = self.get_container_client().get_blob_client(blob_name)
            
            if not blob_client.exists():
                logger.debug(f"Blob does not exist: {blob_name}")
                return None
            
            data = blob_client.download_blob().readall()
            content = data.decode('utf-8')
            logger.debug(f"Downloaded text from blob: {blob_name} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Failed to download text from blob {blob_name}: {e}")
            return None

    def list_blobs(self, prefix: str = "") -> list:
        """
        List blobs with an optional prefix.

        Args:
            prefix: Optional prefix to filter blobs (e.g., "databases/")

        Returns:
            List of blob names
        """
        try:
            container_client = self.get_container_client()
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list blobs with prefix {prefix}: {e}")
            return []

    def list_blobs_with_metadata(self, prefix: str = "") -> list:
        """
        List blobs with metadata (including last modified time).

        Args:
            prefix: Optional prefix to filter blobs (e.g., "wikicache/")

        Returns:
            List of dicts with 'name' and 'last_modified' (timestamp in ms)
        """
        try:
            container_client = self.get_container_client()
            blobs = container_client.list_blobs(name_starts_with=prefix)
            result = []
            for blob in blobs:
                last_modified_ms = int(blob.last_modified.timestamp() * 1000) if blob.last_modified else 0
                result.append({
                    "name": blob.name,
                    "last_modified": last_modified_ms
                })
            return result
        except Exception as e:
            logger.error(f"Failed to list blobs with metadata, prefix {prefix}: {e}")
            return []

    def upload_directory(self, local_dir: str, blob_prefix: str) -> bool:
        """
        Upload an entire directory to blob storage.

        Args:
            local_dir: Local directory path to upload
            blob_prefix: Prefix for blob names (e.g., "repos/my_repo/")

        Returns:
            bool: True if all files uploaded successfully
        """
        import os
        
        try:
            if not os.path.exists(local_dir):
                logger.error(f"Directory does not exist: {local_dir}")
                return False
            
            uploaded_count = 0
            failed_count = 0
            
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    # Calculate relative path from local_dir
                    rel_path = os.path.relpath(local_path, local_dir)
                    # Convert Windows path separators to forward slashes
                    rel_path = rel_path.replace("\\", "/")
                    blob_name = f"{blob_prefix.rstrip('/')}/{rel_path}"
                    
                    try:
                        with open(local_path, 'rb') as f:
                            data = f.read()
                        blob_client = self.get_container_client().get_blob_client(blob_name)
                        blob_client.upload_blob(data, overwrite=True)
                        uploaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to upload {local_path}: {e}")
                        failed_count += 1
            
            logger.info(f"Uploaded directory {local_dir} to blob prefix {blob_prefix}: {uploaded_count} files, {failed_count} failed")
            return failed_count == 0
        except Exception as e:
            logger.error(f"Failed to upload directory {local_dir}: {e}")
            return False

    def download_directory(self, blob_prefix: str, local_dir: str) -> bool:
        """
        Download all blobs with a prefix to a local directory.

        Args:
            blob_prefix: Prefix for blob names (e.g., "repos/my_repo/")
            local_dir: Local directory to download to

        Returns:
            bool: True if all files downloaded successfully
        """
        import os
        
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            blobs = self.list_blobs(blob_prefix)
            if not blobs:
                logger.info(f"No blobs found with prefix: {blob_prefix}")
                return False
            
            downloaded_count = 0
            failed_count = 0
            
            for blob_name in blobs:
                # Calculate relative path from prefix
                rel_path = blob_name[len(blob_prefix):].lstrip('/')
                local_path = os.path.join(local_dir, rel_path)
                
                # Create parent directories if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                try:
                    blob_client = self.get_container_client().get_blob_client(blob_name)
                    data = blob_client.download_blob().readall()
                    with open(local_path, 'wb') as f:
                        f.write(data)
                    downloaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to download {blob_name}: {e}")
                    failed_count += 1
            
            logger.info(f"Downloaded {downloaded_count} files from blob prefix {blob_prefix} to {local_dir}, {failed_count} failed")
            return failed_count == 0 and downloaded_count > 0
        except Exception as e:
            logger.error(f"Failed to download directory from {blob_prefix}: {e}")
            return False

    def directory_exists(self, blob_prefix: str) -> bool:
        """
        Check if any blobs exist with the given prefix.

        Args:
            blob_prefix: Prefix to check (e.g., "repos/my_repo/")

        Returns:
            bool: True if at least one blob exists with the prefix
        """
        try:
            container_client = self.get_container_client()
            blobs = container_client.list_blobs(name_starts_with=blob_prefix)
            # Check if at least one blob exists
            for _ in blobs:
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check directory existence {blob_prefix}: {e}")
            return False


def get_blob_storage_client() -> Optional[AzureBlobStorageClient]:
    """
    Get the singleton Azure Blob Storage client.
    Initializes from infra.json configuration.

    Returns:
        AzureBlobStorageClient instance or None if not configured
    """
    global _blob_client
    
    if _blob_client is not None:
        return _blob_client
    
    try:
        infra = get_infra_config()
        blob_config = infra.get("azure_blob_storage", {})
        
        account_name = blob_config.get("account_name")
        container_name = blob_config.get("container_name", "deepwiki-data")
        
        if not account_name:
            logger.warning("Azure Blob Storage not configured in infra.json")
            return None
        
        msi_client_id = get_managed_identity_client_id()
        
        _blob_client = AzureBlobStorageClient(
            account_name=account_name,
            container_name=container_name,
            managed_identity_client_id=msi_client_id
        )
        
        return _blob_client
    except Exception as e:
        logger.error(f"Failed to initialize Azure Blob Storage client: {e}")
        return None


def is_blob_storage_configured() -> bool:
    """
    Check if Azure Blob Storage is enabled and configured in infra.json.
    
    The 'enabled' property controls whether to use blob storage:
        - enabled: true (default) -> use Azure Blob Storage
        - enabled: false -> use local storage

    Returns:
        bool: True if enabled and configured, False otherwise
    """
    infra = get_infra_config()
    blob_config = infra.get("azure_blob_storage", {})
    
    # Check if enabled (defaults to True if not specified)
    enabled = blob_config.get("enabled", True)
    if not enabled:
        logger.debug("Azure Blob Storage is disabled in config")
        return False
    
    # Check if account_name is configured
    return bool(blob_config.get("account_name"))
