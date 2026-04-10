"""
JSON-based Vector Storage for DeepWiki.

This module provides memory-efficient storage of embeddings as JSON files,
organized by source code file structure. Each chunk is stored as a separate
JSON file with suffix _001, _002, etc.

Storage structure:
    vectors/{owner}_{repo}_{branch}/
        └── src/backend/main_001.json  (chunk 1 for main.py)
        └── src/backend/main_002.json  (chunk 2 for main.py)
        └── src/backend/main_003.json  (chunk 3 for main.py)
        └── src/utils/helper_001.json  (chunk 1 for helper.py)
        └── README_001.json            (chunk 1 for README.md)
        ...

JSON format per chunk file:
    {
        "file_path": "src/backend/main.py",
        "chunk_index": 0,
        "total_chunks": 5,
        "text": "chunk text content",
        "vector": [0.123, 0.456, ...],
        "meta_data": {
            "file_path": "src/backend/main.py",
            "type": "py",
            "url": "https://github.com/owner/repo/blob/branch/src/backend/main.py",
            "raw_chunk_text": "...",
            ...
        }
    }

This module supports both local and Azure Blob storage backends.
"""

import os
import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple

try:
    import orjson as _json_fast
    _USE_ORJSON = True
except ImportError:
    _json_fast = None  # type: ignore
    _USE_ORJSON = False

from adalflow.core.types import Document
from backend.paths import get_adalflow_root_path

from backend.clients.blob_client import (
    get_blob_storage_client,
    is_blob_storage_configured,
)

logger = logging.getLogger(__name__)


class VectorStorage:
    """
    JSON-based vector storage with one JSON file per chunk.
    
    Key features:
    - One JSON file per chunk (named with _001, _002, etc. suffix)
    - Memory-efficient: loads files on demand
    - Organized by source file structure
    - Supports both local and blob storage
    - Backward compatible with existing pkl databases
    """
    
    VECTORS_DIR = "vectors"
    
    def __init__(self):
        self._root_path = get_adalflow_root_path()
    
    def _get_vectors_base_path(self, repo_name: str, branch: str) -> str:
        """Get the base path for vectors storage."""
        branch_suffix = branch.strip() if branch and branch.strip() else 'main'
        return f"{self.VECTORS_DIR}/{repo_name}_{branch_suffix}"
    
    def _get_local_vectors_path(self, repo_name: str, branch: str) -> str:
        """Get local filesystem path for vectors."""
        return os.path.join(self._root_path, self._get_vectors_base_path(repo_name, branch))
    
    def _sanitize_path_for_storage(self, file_path: str) -> str:
        """
        Sanitize source file path for use in storage.
        Replaces path separators and special characters.
        """
        # Normalize path separators
        sanitized = file_path.replace("\\", "/")
        # Replace characters that might cause issues in filenames
        sanitized = re.sub(r'[<>:"|?*]', '_', sanitized)
        return sanitized
    
    def _get_json_filename(self, source_file: str, chunk_index: int) -> str:
        """
        Generate JSON filename for a chunk of a source file.
        
        Example: src/backend/main.py, chunk 0 -> src/backend/main_001.json
        Example: src/backend/main.py, chunk 5 -> src/backend/main_006.json
        """
        sanitized = self._sanitize_path_for_storage(source_file)
        # Remove extension and add chunk suffix
        base_name = os.path.splitext(sanitized)[0]
        # Use 1-based indexing for display (001, 002, etc.)
        return f"{base_name}_{chunk_index + 1:03d}.json"
    
    def _parse_chunk_filename(self, filename: str) -> Tuple[str, int]:
        """
        Parse a chunk filename to extract source file base and chunk index.
        
        Example: src/backend/main_001.json -> ("src/backend/main", 0)
        
        Returns:
            Tuple of (base_name, chunk_index) or (filename, -1) if not parseable
        """
        if not filename.endswith('.json'):
            return filename, -1
        
        base = filename[:-5]  # Remove .json
        # Match pattern: base_NNN where NNN is 3 digits
        match = re.match(r'^(.+)_(\d{3})$', base)
        if match:
            return match.group(1), int(match.group(2)) - 1  # Convert to 0-based
        return base, -1
    
    def _chunk_to_dict(self, doc: Document, chunk_index: int, total_chunks: int,
                       source_file: str) -> Dict[str, Any]:
        """Convert a Document chunk to a dictionary for JSON serialization."""
        meta_data = dict(doc.meta_data) if doc.meta_data else {}
        meta_data["chunk_index"] = chunk_index
        meta_data["total_chunks"] = total_chunks
        
        # Convert vector to list if needed
        vector = doc.vector
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        elif not isinstance(vector, list):
            vector = list(vector) if vector else []
        
        return {
            "file_path": source_file,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "text": doc.text,
            "vector": vector,
            "meta_data": meta_data,
        }
    
    def _dict_to_document(self, data: Dict[str, Any]) -> Document:
        """Convert a dictionary back to a Document object."""
        # Support both new format (flat) and legacy format (nested chunks)
        meta_data = data.get("meta_data", {})
        
        # For new format, add chunk info to meta_data if not present
        if "chunk_index" not in meta_data and "chunk_index" in data:
            meta_data["chunk_index"] = data["chunk_index"]
        if "total_chunks" not in meta_data and "total_chunks" in data:
            meta_data["total_chunks"] = data["total_chunks"]
        
        doc = Document(
            text=data.get("text", ""),
            meta_data=meta_data,
        )
        doc.vector = data.get("vector", [])
        return doc
    
    def exists(self, repo_name: str, branch: str) -> bool:
        """
        Check if vectors exist for the given repository.
        
        Args:
            repo_name: Repository name (owner_repo format)
            branch: Branch name
            
        Returns:
            True if vectors directory exists and contains files
        """
        vectors_path = self._get_vectors_base_path(repo_name, branch)
        
        if is_blob_storage_configured():
            try:
                blob_client = get_blob_storage_client()
                if blob_client:
                    # Check if directory exists in blob
                    return blob_client.directory_exists(vectors_path + "/")
            except Exception as e:
                logger.error(f"Error checking blob vectors existence: {e}")
                return False
        else:
            # Local storage
            local_path = self._get_local_vectors_path(repo_name, branch)
            if os.path.exists(local_path) and os.path.isdir(local_path):
                # Check if directory contains any JSON files
                for root, _, files in os.walk(local_path):
                    for f in files:
                        if f.endswith('.json'):
                            return True
            return False
    
    def save_documents(
        self,
        documents: List[Document],
        repo_name: str,
        branch: str,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Save transformed documents as JSON files (one per chunk).
        
        Each chunk is saved as a separate JSON file with suffix _001, _002, etc.
        
        Args:
            documents: List of Document objects with embeddings
            repo_name: Repository name (owner_repo format)
            branch: Branch name
            progress_callback: Optional callback(saved, total) for progress
            
        Returns:
            True if all files saved successfully
        """
        if not documents:
            logger.warning("[Vec] No documents to save")
            return False
        
        vectors_path = self._get_vectors_base_path(repo_name, branch)
        logger.info(f"[Vec] Saving {len(documents)} chunks to {vectors_path}")
        
        # Group documents by source file to get total_chunks for each file
        docs_by_file: Dict[str, List[Tuple[int, Document]]] = {}
        for i, doc in enumerate(documents):
            file_path = (
                doc.meta_data.get("file_path", "unknown")
                if doc.meta_data else "unknown"
            )
            if file_path not in docs_by_file:
                docs_by_file[file_path] = []
            docs_by_file[file_path].append((i, doc))
        
        logger.info(f"[Vec] {len(docs_by_file)} source files, {len(documents)} total chunks")
        
        try:
            if is_blob_storage_configured():
                return self._save_to_blob(
                    docs_by_file, vectors_path, len(documents), progress_callback
                )
            else:
                return self._save_to_local(
                    docs_by_file, vectors_path, len(documents), progress_callback
                )
        except Exception as e:
            logger.error(f"[Vec] Error saving vectors: {e}")
            return False
    
    def _save_to_local(
        self,
        docs_by_file: Dict[str, List[Tuple[int, Document]]],
        vectors_path: str,
        total_chunks: int,
        progress_callback: Optional[callable]
    ) -> bool:
        """Save documents to local filesystem (one JSON per chunk)."""
        local_base = os.path.join(self._root_path, vectors_path)
        os.makedirs(local_base, exist_ok=True)
        
        chunks_saved = 0
        save_errors: list = []  # accumulate errors for summary
        
        for source_file, doc_list in docs_by_file.items():
            file_total_chunks = len(doc_list)
            
            for chunk_idx, (_, doc) in enumerate(doc_list):
                json_filename = self._get_json_filename(source_file, chunk_idx)
                json_path = os.path.join(local_base, json_filename)
                
                # Ensure directory exists
                json_dir = os.path.dirname(json_path)
                if json_dir:
                    os.makedirs(json_dir, exist_ok=True)
                
                # Create chunk data
                chunk_data = self._chunk_to_dict(
                    doc, chunk_idx, file_total_chunks, source_file
                )
                
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk_data, f, ensure_ascii=False)
                    chunks_saved += 1
                    
                    if progress_callback:
                        progress_callback(chunks_saved, total_chunks)
                        
                except Exception as e:
                    if len(save_errors) < 3:
                        save_errors.append(f"{json_path}: {e}")
                    else:
                        save_errors.append(None)  # count only
        
        if save_errors:
            real_errors = [e for e in save_errors if e is not None]
            logger.error(
                f"[Vec] Failed to save {len(save_errors)} chunks locally "
                f"(first: {real_errors[0] if real_errors else 'N/A'})"
            )
            return False
        
        logger.info(f"[Vec] Saved {chunks_saved} chunk files to local storage")
        return True
    
    def _save_to_blob(
        self,
        docs_by_file: Dict[str, List[Tuple[int, Document]]],
        vectors_path: str,
        total_chunks: int,
        progress_callback: Optional[callable]
    ) -> bool:
        """Save documents to Azure Blob Storage (one JSON per chunk)."""
        blob_client = get_blob_storage_client()
        if not blob_client:
            logger.error("[Vec] Blob client not available")
            return False
        
        chunks_saved = 0
        save_errors: list = []  # accumulate errors for summary
        
        for source_file, doc_list in docs_by_file.items():
            file_total_chunks = len(doc_list)
            
            for chunk_idx, (_, doc) in enumerate(doc_list):
                json_filename = self._get_json_filename(source_file, chunk_idx)
                blob_path = f"{vectors_path}/{json_filename}"
                
                # Create chunk data
                chunk_data = self._chunk_to_dict(
                    doc, chunk_idx, file_total_chunks, source_file
                )
                
                json_content = json.dumps(chunk_data, ensure_ascii=False)
                
                try:
                    if not blob_client.upload_text(blob_path, json_content):
                        if len(save_errors) < 3:
                            save_errors.append(f"upload failed: {blob_path}")
                        else:
                            save_errors.append(None)
                        continue
                    
                    chunks_saved += 1
                    
                    if progress_callback:
                        progress_callback(chunks_saved, total_chunks)
                        
                except Exception as e:
                    if len(save_errors) < 3:
                        save_errors.append(f"{blob_path}: {e}")
                    else:
                        save_errors.append(None)
        
        if save_errors:
            real_errors = [e for e in save_errors if e is not None]
            logger.error(
                f"[Vec] Failed to save {len(save_errors)} chunks to blob "
                f"(first: {real_errors[0] if real_errors else 'N/A'})"
            )
            return False
        
        logger.info(f"[Vec] Saved {chunks_saved} chunk files to blob storage")
        return True
    
    def load_documents(
        self,
        repo_name: str,
        branch: str,
        progress_callback: Optional[callable] = None
    ) -> List[Document]:
        """
        Load all documents from vector storage.
        
        Args:
            repo_name: Repository name (owner_repo format)
            branch: Branch name
            progress_callback: Optional callback(loaded, total) for progress
            
        Returns:
            List of Document objects with embeddings
        """
        vectors_path = self._get_vectors_base_path(repo_name, branch)
        
        try:
            if is_blob_storage_configured():
                return self._load_from_blob(vectors_path, progress_callback)
            else:
                return self._load_from_local(vectors_path, progress_callback)
        except Exception as e:
            logger.error(f"[Vec] Error loading vectors: {e}")
            return []
    
    @staticmethod
    def _read_json_file(json_path: str) -> Optional[Dict[str, Any]]:
        """Read and parse a single JSON chunk file. Returns None on error."""
        try:
            if _USE_ORJSON:
                with open(json_path, 'rb') as f:
                    return _json_fast.loads(f.read())
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            # Errors are counted by the caller; per-file logging is too noisy
            return None

    def _load_from_local(
        self,
        vectors_path: str,
        progress_callback: Optional[callable]
    ) -> List[Document]:
        """Load documents from local filesystem using parallel I/O."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        local_base = os.path.join(self._root_path, vectors_path)

        if not os.path.exists(local_base):
            logger.info(f"[Vec] Vectors directory does not exist: {local_base}")
            return []

        # Find all JSON files
        json_files = []
        for root, _, files in os.walk(local_base):
            for f in files:
                if f.endswith('.json'):
                    json_files.append(os.path.join(root, f))

        if not json_files:
            logger.info("[Vec] No JSON files found")
            return []

        total = len(json_files)
        logger.info(f"[Vec] Found {total} chunk files, loading with parallel I/O...")
        load_start = time.time()

        # Parallel read: I/O-bound, so many workers help
        workers = min(32, max(8, total // 200))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw_results = list(pool.map(self._read_json_file, json_files))

        documents = []
        for chunk_data in raw_results:
            if chunk_data is not None:
                documents.append(self._dict_to_document(chunk_data))

        failed = total - len(documents)
        elapsed = time.time() - load_start
        if failed > 0:
            logger.warning(
                f"[Vec] {failed}/{total} chunk files failed to load from local storage"
            )
        logger.info(
            f"[Vec] Loaded {len(documents)}/{total} chunks from local storage "
            f"in {elapsed:.1f}s ({workers} workers)"
        )
        return documents
    
    def _load_from_blob(
        self,
        vectors_path: str,
        progress_callback: Optional[callable]
    ) -> List[Document]:
        """Load documents from Azure Blob Storage using parallel downloads."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        blob_client = get_blob_storage_client()
        if not blob_client:
            logger.error("[Vec] Blob client not available")
            return []

        # List all blobs in the vectors path
        try:
            blobs = blob_client.list_blobs(vectors_path + "/")
            json_blobs = [b for b in blobs if b.endswith('.json')]
        except Exception as e:
            logger.error(f"[Vec] Failed to list blobs: {e}")
            return []

        if not json_blobs:
            logger.info("[Vec] No JSON blobs found")
            return []

        total = len(json_blobs)
        logger.info(f"[Vec] Found {total} chunk files in blob, loading in parallel...")
        load_start = time.time()

        def _download_one(blob_name: str) -> Optional[Dict[str, Any]]:
            try:
                content = blob_client.download_text(blob_name)
                if content:
                    if _USE_ORJSON:
                        return _json_fast.loads(content.encode('utf-8') if isinstance(content, str) else content)
                    return json.loads(content)
            except Exception:
                # Errors are counted by the caller; per-blob logging is too noisy
                pass
            return None

        # Parallel download: network-bound, more workers help
        workers = min(64, max(8, total // 100))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw_results = list(pool.map(_download_one, json_blobs))

        documents = []
        for chunk_data in raw_results:
            if chunk_data is not None:
                documents.append(self._dict_to_document(chunk_data))

        failed = total - len(documents)
        elapsed = time.time() - load_start
        if failed > 0:
            logger.warning(
                f"[Vec] {failed}/{total} chunk files failed to load from blob storage"
            )
        logger.info(
            f"[Vec] Loaded {len(documents)}/{total} chunks from blob storage "
            f"in {elapsed:.1f}s ({workers} workers)"
        )
        return documents
    
    def list_files(self, repo_name: str, branch: str) -> set:
        """List all vector JSON file paths (relative to vectors base) for a repo.

        Used to snapshot existing files before reprocessing so orphans
        can be cleaned up afterward without deleting everything upfront.

        Args:
            repo_name: Repository name (owner_repo format)
            branch: Branch name

        Returns:
            Set of relative file paths (e.g. {"src/main_001.json", ...})
        """
        vectors_path = self._get_vectors_base_path(repo_name, branch)
        files: set = set()

        try:
            if is_blob_storage_configured():
                blob_client = get_blob_storage_client()
                if blob_client:
                    prefix = vectors_path + "/"
                    blobs = blob_client.list_blobs(prefix)
                    for b in blobs:
                        if b.endswith('.json'):
                            # Store path relative to vectors base
                            files.add(b[len(prefix):])
            else:
                local_base = self._get_local_vectors_path(repo_name, branch)
                if os.path.exists(local_base):
                    for root, _, filenames in os.walk(local_base):
                        for f in filenames:
                            if f.endswith('.json'):
                                rel = os.path.relpath(
                                    os.path.join(root, f), local_base
                                ).replace("\\", "/")
                                files.add(rel)
        except Exception as e:
            logger.warning(f"[Vec] Failed to list files: {e}")

        return files

    def delete_files(self, repo_name: str, branch: str, rel_paths: set) -> int:
        """Delete specific vector files by relative path.

        Used for orphan cleanup after incremental reprocessing.

        Args:
            repo_name: Repository name (owner_repo format)
            branch: Branch name
            rel_paths: Set of relative paths to delete

        Returns:
            Number of files successfully deleted
        """
        if not rel_paths:
            return 0

        vectors_path = self._get_vectors_base_path(repo_name, branch)
        deleted = 0

        try:
            if is_blob_storage_configured():
                blob_client = get_blob_storage_client()
                if blob_client:
                    for rp in rel_paths:
                        blob_path = f"{vectors_path}/{rp}"
                        try:
                            blob_client.delete_blob(blob_path)
                            deleted += 1
                        except Exception as e:
                            logger.warning(f"[Vec] Failed to delete blob {blob_path}: {e}")
            else:
                local_base = self._get_local_vectors_path(repo_name, branch)
                for rp in rel_paths:
                    full_path = os.path.join(local_base, rp.replace("/", os.sep))
                    try:
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            deleted += 1
                            # Remove empty parent dirs up to base
                            parent = os.path.dirname(full_path)
                            while parent != local_base:
                                if os.path.isdir(parent) and not os.listdir(parent):
                                    os.rmdir(parent)
                                    parent = os.path.dirname(parent)
                                else:
                                    break
                    except Exception as e:
                        logger.warning(f"[Vec] Failed to delete {full_path}: {e}")
        except Exception as e:
            logger.error(f"[Vec] Error during file deletion: {e}")

        logger.info(f"[Vec] Deleted {deleted}/{len(rel_paths)} orphan files")
        return deleted

    def delete(self, repo_name: str, branch: str) -> bool:
        """
        Delete all vectors for a repository.
        
        Args:
            repo_name: Repository name (owner_repo format)
            branch: Branch name
            
        Returns:
            True if deleted successfully
        """
        vectors_path = self._get_vectors_base_path(repo_name, branch)
        
        try:
            if is_blob_storage_configured():
                blob_client = get_blob_storage_client()
                if blob_client:
                    return blob_client.delete_directory(vectors_path + "/")
            else:
                local_path = self._get_local_vectors_path(repo_name, branch)
                if os.path.exists(local_path):
                    import shutil
                    shutil.rmtree(local_path)
                    logger.info(f"[Vec] Deleted vectors directory: {local_path}")
                return True
        except Exception as e:
            logger.error(f"[Vec] Failed to delete vectors: {e}")
            return False


# Singleton instance
_vector_storage: Optional[VectorStorage] = None


def get_vector_storage() -> VectorStorage:
    """Get the singleton VectorStorage instance."""
    global _vector_storage
    if _vector_storage is None:
        _vector_storage = VectorStorage()
    return _vector_storage
