"""
Unified Storage Abstraction for DeepWiki.

This module provides a consistent interface for storage operations,
automatically choosing between Azure Blob Storage and local storage
based on configuration.

Usage:
    from backend.clients.storage import storage
    
    # Check storage mode
    mode = storage.get_mode()  # Returns "blob" or "local"
    
    # Check existence
    if storage.exists("path/to/data"):
        ...
"""

import os
import logging
from typing import Optional, Any, List, Dict
from pathlib import Path

from backend.paths import get_adalflow_root_path

from backend.clients.blob_client import (
    get_blob_storage_client,
    is_blob_storage_configured,
    get_storage_mode
)

logger = logging.getLogger(__name__)


class StorageClient:
    """
    Unified storage client that abstracts blob vs local storage.
    
    Automatically uses:
    - Azure Blob Storage when enabled in config
    - Local storage (~/.adalflow/) when blob is disabled
    """
    
    def __init__(self):
        self._mode = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Initialize storage on first use."""
        if self._initialized:
            return
        
        mode, reason = get_storage_mode()
        self._mode = mode
        self._initialized = True
        
        logger.info(f"📦 [Storage] Initialized in {mode.upper()} mode: {reason}")
    
    def get_mode(self) -> str:
        """
        Get current storage mode.
        
        Returns:
            "blob" or "local"
        """
        self._ensure_initialized()
        return self._mode
    
    def is_blob_mode(self) -> bool:
        """Check if using blob storage."""
        return self.get_mode() == "blob"
    
    def _get_local_path(self, relative_path: str) -> str:
        """Convert relative path to local filesystem path."""
        base_path = get_adalflow_root_path()
        return os.path.join(base_path, relative_path)
    
    def exists(self, path: str) -> bool:
        """
        Check if a path exists in storage.
        
        Args:
            path: Relative path to check
            
        Returns:
            bool: True if exists
        """
        self._ensure_initialized()
        
        if self.is_blob_mode():
            blob_client = get_blob_storage_client()
            if blob_client:
                return blob_client.exists(path)
            return False
        else:
            local_path = self._get_local_path(path)
            return os.path.exists(local_path)
    
    def delete(self, path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            path: Relative path to delete
            
        Returns:
            bool: True if deleted successfully
        """
        self._ensure_initialized()
        
        if self.is_blob_mode():
            blob_client = get_blob_storage_client()
            if blob_client:
                return blob_client.delete(path)
            return False
        else:
            local_path = self._get_local_path(path)
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                    logger.debug(f"📦 [Storage] Deleted local: {local_path}")
                return True
            except Exception as e:
                logger.error(f"📦 [Storage] Delete failed: {e}")
                return False
    
    def save_text(self, path: str, content: str) -> bool:
        """
        Save text content to storage.
        
        Args:
            path: Relative path for the file
            content: Text content to save
            
        Returns:
            bool: True if successful
        """
        self._ensure_initialized()
        
        if self.is_blob_mode():
            blob_client = get_blob_storage_client()
            if blob_client:
                return blob_client.upload_text(path, content)
            return False
        else:
            local_path = self._get_local_path(path)
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"📦 [Storage] Saved text to local: {local_path}")
                return True
            except Exception as e:
                logger.error(f"📦 [Storage] Local text save failed: {e}")
                return False
    
    def load_text(self, path: str) -> Optional[str]:
        """
        Load text content from storage.
        
        Args:
            path: Relative path for the file
            
        Returns:
            Text content or None if not found
        """
        self._ensure_initialized()
        
        if self.is_blob_mode():
            blob_client = get_blob_storage_client()
            if blob_client:
                return blob_client.download_text(path)
            return None
        else:
            local_path = self._get_local_path(path)
            if not os.path.exists(local_path):
                return None
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"📦 [Storage] Local text load failed: {e}")
                return None
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files with optional prefix.
        
        Args:
            prefix: Path prefix to filter
            
        Returns:
            List of file paths
        """
        self._ensure_initialized()
        
        if self.is_blob_mode():
            blob_client = get_blob_storage_client()
            if blob_client:
                return blob_client.list_blobs(prefix)
            return []
        else:
            local_path = self._get_local_path(prefix)
            if not os.path.exists(local_path):
                return []
            try:
                result = []
                for root, _, files in os.walk(local_path):
                    for f in files:
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(
                            full_path,
                            self._get_local_path("")
                        )
                        result.append(rel_path.replace("\\", "/"))
                return result
            except Exception as e:
                logger.error(f"📦 [Storage] Local list failed: {e}")
                return []
    
    def directory_exists(self, prefix: str) -> bool:
        """
        Check if any files exist with the given prefix.
        
        Args:
            prefix: Path prefix to check
            
        Returns:
            bool: True if any files exist
        """
        self._ensure_initialized()
        
        if self.is_blob_mode():
            blob_client = get_blob_storage_client()
            if blob_client:
                return blob_client.directory_exists(prefix)
            return False
        else:
            local_path = self._get_local_path(prefix)
            if not os.path.exists(local_path):
                return False
            # Check if directory has any files
            try:
                for _ in os.scandir(local_path):
                    return True
                return False
            except Exception:
                return False


# Singleton instance
storage = StorageClient()
