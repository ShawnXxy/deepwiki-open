"""
Azure Blob Storage client for DeepWiki.
Handles storage of repository databases and embeddings.

Authentication chain:
1. MSI with explicit client_id (Azure Container Apps)
2. DefaultAzureCredential fallback (includes Azure CLI, VS Code, etc.)

Storage decision:
- If blob storage is enabled in config → use Azure Blob
- If blob storage is disabled → use local storage (~/.adalflow/)
"""

import io
import os
import logging
from typing import Optional, Any, Tuple

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ClientAuthenticationError

from backend.config import get_infra_config, get_managed_identity_client_id

logger = logging.getLogger(__name__)

# Singleton blob client instance
_blob_client: Optional["AzureBlobStorageClient"] = None
# Track if blob storage initialization was attempted and failed
_blob_init_failed: bool = False
_blob_init_error: Optional[str] = None


class AzureBlobStorageClient:
    """
    Azure Blob Storage client with MSI/DefaultAzureCredential authentication.
    
    Authentication chain:
    1. MSI with explicit client_id (for Azure Container Apps)
    2. DefaultAzureCredential (MSI → Azure CLI → VS Code → Environment)
    """

    def __init__(
        self,
        account_name: str,
        container_name: str,
        managed_identity_client_id: Optional[str] = None
    ):
        """
        Initialize the Azure Blob Storage client.

        Args:
            account_name: Azure Storage account name
            container_name: Blob container name
            managed_identity_client_id: Optional MSI client ID for authentication
        
        Raises:
            ClientAuthenticationError: If authentication fails
            Exception: If connection to blob storage fails
        """
        self.account_name = account_name
        self.container_name = container_name
        self.account_url = f"https://{account_name}.blob.core.windows.net"
        
        logger.info(f"🔧 [BlobStorage] Initializing client...")
        logger.info(f"   Account: {account_name}")
        logger.info(f"   Container: {container_name}")
        
        # Create credential with auth chain
        if managed_identity_client_id:
            logger.info(f"🔐 [BlobStorage] Auth method: MSI with "
                        f"client_id: {managed_identity_client_id[:8]}...")
            self.credential = DefaultAzureCredential(
                managed_identity_client_id=managed_identity_client_id
            )
        else:
            logger.info("🔐 [BlobStorage] Auth method: DefaultAzureCredential "
                        "(MSI → Azure CLI → VS Code → Environment)")
            self.credential = DefaultAzureCredential()
        
        # Create blob service client with larger connection pool.
        # Default urllib3 pool size (10) causes "Connection pool is full"
        # warnings during concurrent embedding uploads.
        from azure.core.pipeline.transport import RequestsTransport
        import requests

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50, pool_maxsize=50,
        )
        session.mount("https://", adapter)

        self.blob_service_client = BlobServiceClient(
            account_url=self.account_url,
            credential=self.credential,
            transport=RequestsTransport(session=session),
        )
        
        # Validate connection by ensuring container exists
        self._ensure_container_exists()
        
        logger.info(f"✅ [BlobStorage] Client initialized successfully")

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
            
        Raises:
            Exception: If listing fails, to allow caller to handle appropriately
        """
        logger.debug(f"[BlobStorage] list_blobs_with_metadata called with prefix: {prefix}")
        container_client = self.get_container_client()
        logger.debug(f"[BlobStorage] Got container client, listing blobs...")
        blobs = container_client.list_blobs(name_starts_with=prefix)
        result = []
        for blob in blobs:
            last_modified_ms = int(blob.last_modified.timestamp() * 1000) if blob.last_modified else 0
            result.append({
                "name": blob.name,
                "last_modified": last_modified_ms
            })
            logger.debug(f"[BlobStorage] Found blob: {blob.name}")
        logger.debug(f"[BlobStorage] Total blobs found: {len(result)}")
        return result

    def upload_directory(self, local_dir: str, blob_prefix: str) -> bool:
        """
        Upload an entire directory to blob storage with full sync.

        Uploads all local files and deletes any blob files under the prefix
        that no longer exist locally (stale file cleanup).

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
            failed_files: list = []  # collect first few failures for summary
            uploaded_blobs = set()
            
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    # Calculate relative path from local_dir
                    rel_path = os.path.relpath(local_path, local_dir)
                    # Convert Windows path separators to forward slashes
                    rel_path = rel_path.replace("\\", "/")
                    blob_name = f"{blob_prefix.rstrip('/')}/{rel_path}"
                    uploaded_blobs.add(blob_name)
                    
                    try:
                        with open(local_path, 'rb') as f:
                            data = f.read()
                        blob_client = self.get_container_client().get_blob_client(blob_name)
                        blob_client.upload_blob(data, overwrite=True)
                        uploaded_count += 1
                    except Exception as e:
                        if len(failed_files) < 3:
                            failed_files.append(f"{local_path}: {e}")
                        failed_count += 1
            
            if failed_count > 0:
                logger.warning(
                    f"Failed to upload {failed_count} files "
                    f"(first: {failed_files[0]})"
                )

            # Delete stale blobs that no longer exist locally
            existing_blobs = set(self.list_blobs(blob_prefix))
            stale_blobs = existing_blobs - uploaded_blobs
            if stale_blobs:
                deleted_count = 0
                stale_failures = 0
                for blob_name in stale_blobs:
                    try:
                        blob_client = self.get_container_client().get_blob_client(blob_name)
                        blob_client.delete_blob()
                        deleted_count += 1
                    except Exception as e:
                        stale_failures += 1
                if stale_failures > 0:
                    logger.warning(
                        f"Failed to delete {stale_failures}/{len(stale_blobs)} "
                        f"stale blobs from {blob_prefix}"
                    )
                logger.info(f"Deleted {deleted_count} stale blobs from {blob_prefix}")

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
            failed_blobs: list = []  # collect first few failures for summary
            
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
                    if len(failed_blobs) < 3:
                        failed_blobs.append(f"{blob_name}: {e}")
                    failed_count += 1
            
            if failed_count > 0:
                logger.warning(
                    f"Failed to download {failed_count}/{len(blobs)} blobs "
                    f"(first: {failed_blobs[0]})"
                )

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

    def delete_directory(self, blob_prefix: str) -> bool:
        """
        Delete all blobs with the given prefix (simulating directory deletion).
        
        Azure Blob Storage uses a flat namespace - there are no actual directories,
        just blob names that may contain "/" characters. This method simulates
        directory deletion by listing all blobs with the given prefix and deleting
        each one individually.
        
        Purpose:
            This method is primarily used by the VectorStorage class to clean up
            JSON chunk files when regenerating embeddings for a repository. Since
            the new JSON-based vector storage saves each chunk as a separate file
            under a directory-like structure (e.g., "vectors/repo_name_branch/"),
            this method provides a way to remove all those files at once.
        
        Use Cases:
            1. Regenerating embeddings: When embedding model changes or you need
               to re-embed a repository, delete old vectors first
            2. Repository cleanup: Remove vector data for deleted/archived repos
            3. Testing: Clear test data between test runs
            4. Storage management: Free up blob storage space
        
        Example Usage:
            ```python
            blob_client = get_blob_storage_client()
            
            # Delete all JSON chunks for a repository's vectors
            # This removes files like:
            #   vectors/owner_repo_main/src/backend/main_chunk_000.json
            #   vectors/owner_repo_main/src/backend/main_chunk_001.json
            #   vectors/owner_repo_main/README_chunk_000.json
            blob_client.delete_directory("vectors/owner_repo_main/")
            
            # Called internally by VectorStorage.delete():
            vector_storage = get_vector_storage()
            vector_storage.delete("owner_repo", "main")
            ```
        
        Note:
            - Returns True even if no blobs exist (idempotent operation)
            - Logs warnings for individual blob deletion failures but continues
            - Returns False only if ANY blob fails to delete

        Args:
            blob_prefix: Prefix to delete (e.g., "vectors/my_repo/"). Should end
                        with "/" to avoid accidentally matching unintended blobs
                        (e.g., "vectors/my_repo" might match "vectors/my_repo2/")

        Returns:
            bool: True if all blobs deleted successfully (or none existed),
                  False if any deletion failed
        """
        try:
            blobs = self.list_blobs(blob_prefix)
            if not blobs:
                logger.debug(f"No blobs to delete with prefix: {blob_prefix}")
                return True
            
            deleted_count = 0
            failed_count = 0
            
            for blob_name in blobs:
                try:
                    blob_client = self.get_container_client().get_blob_client(blob_name)
                    blob_client.delete_blob()
                    deleted_count += 1
                except Exception:
                    failed_count += 1
            
            if failed_count > 0:
                logger.warning(
                    f"Failed to delete {failed_count}/{len(blobs)} blobs "
                    f"with prefix {blob_prefix}"
                )

            logger.info(f"Deleted {deleted_count} blobs with prefix {blob_prefix}, {failed_count} failed")
            return failed_count == 0
        except Exception as e:
            logger.error(f"Failed to delete directory {blob_prefix}: {e}")
            return False


def get_blob_storage_client() -> Optional[AzureBlobStorageClient]:
    """
    Get the singleton Azure Blob Storage client.
    
    This function checks if blob storage is enabled before initializing.
    If disabled or initialization fails, returns None.

    Returns:
        AzureBlobStorageClient instance or None if disabled/failed
        
    Note:
        Callers should use is_blob_storage_configured() first to determine
        whether to use blob or local storage.
    """
    global _blob_client, _blob_init_failed, _blob_init_error
    
    # Return cached client if available
    if _blob_client is not None:
        return _blob_client
    
    # Don't retry if initialization already failed
    if _blob_init_failed:
        logger.debug(f"📦 [BlobStorage] Skipping - previous init failed: "
                     f"{_blob_init_error}")
        return None
    
    # Check if blob storage is enabled
    if not is_blob_storage_configured():
        logger.info("📦 [BlobStorage] Disabled in config → using local storage")
        return None
    
    try:
        infra = get_infra_config()
        blob_config = infra.azure_blob_storage
        
        account_name = blob_config.account_name
        container_name = blob_config.container_name
        
        if not account_name:
            logger.warning("⚠️ [BlobStorage] No account_name in config")
            _blob_init_failed = True
            _blob_init_error = "No account_name configured"
            return None
        
        msi_client_id = get_managed_identity_client_id()
        
        logger.info(f"📦 [BlobStorage] Initializing connection to "
                    f"{account_name}/{container_name}...")
        
        _blob_client = AzureBlobStorageClient(
            account_name=account_name,
            container_name=container_name,
            managed_identity_client_id=msi_client_id
        )
        
        logger.info("✅ [BlobStorage] Ready for use")
        return _blob_client
        
    except ClientAuthenticationError as e:
        _blob_init_failed = True
        _blob_init_error = f"Authentication failed: {e}"
        logger.error(f"❌ [BlobStorage] Auth failed - check MSI/credentials: {e}")
        raise ConnectionError(
            f"Blob storage auth failed. "
            f"Ensure MSI has 'Storage Blob Data Contributor' role. "
            f"Error: {e}"
        )
    except Exception as e:
        _blob_init_failed = True
        _blob_init_error = str(e)
        logger.error(f"❌ [BlobStorage] Init failed: {e}")
        raise ConnectionError(f"Blob storage connection failed: {e}")


def is_blob_storage_configured() -> bool:
    """
    Check if Azure Blob Storage is enabled in infra.json config.
    
    This is the PRIMARY check to determine storage mode:
        - True  → Use Azure Blob Storage
        - False → Use local storage (~/.adalflow/)
    
    The 'enabled' flag in config controls this behavior:
        - enabled: true  → blob storage mode
        - enabled: false → local storage mode (for Docker/testing)

    Returns:
        bool: True if blob storage should be used, False for local storage
    """
    infra = get_infra_config()
    blob_config = infra.azure_blob_storage
    
    # Check if enabled
    enabled = blob_config.enabled
    if not enabled:
        logger.debug("📦 [Storage] Mode: LOCAL (blob disabled in config)")
        return False
    
    # Check if account_name is configured
    has_account = bool(blob_config.account_name)
    if has_account:
        logger.debug("📦 [Storage] Mode: BLOB (enabled and configured)")
    else:
        logger.debug("📦 [Storage] Mode: LOCAL (no account_name)")
    return has_account


def get_storage_mode() -> Tuple[str, Optional[str]]:
    """
    Get current storage mode and reason.
    
    Returns:
        Tuple of (mode, reason) where:
        - mode: "blob" or "local"
        - reason: Explanation for the mode
    """
    infra = get_infra_config()
    blob_config = infra.azure_blob_storage
    
    enabled = blob_config.enabled
    if not enabled:
        return ("local", "Blob storage disabled in config")
    
    account_name = blob_config.account_name
    if not account_name:
        return ("local", "No blob account_name configured")
    
    if _blob_init_failed:
        return ("local", f"Blob init failed: {_blob_init_error}")
    
    return ("blob", f"Using {account_name}")
