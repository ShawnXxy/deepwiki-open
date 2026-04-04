"""
Database management for RAG document storage.

Provides the DatabaseManager class for managing document databases.
"""

import os
import logging
from typing import List

from adalflow.core.types import Document
from backend.paths import get_adalflow_root_path

from backend.config import configs
from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.clients.vector_storage import get_vector_storage
from backend.modules.embedder.document import transform_documents_and_save_as_json
from backend.modules.repository.git_ops import download_repo, detect_default_branch

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages the creation, loading, and persistence of document embeddings.
    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(
        self,
        repo_url_or_path: str,
        repo_type: str = None,
        access_token: str = None,
        branch: str = None,
        embedder_type: str = None,
        is_ollama_embedder: bool = None,
        excluded_dirs: List[str] = None,
        excluded_files: List[str] = None,
        included_dirs: List[str] = None,
        included_files: List[str] = None,
        force_reprocess: bool = False,
        repo_dir: str = None
    ) -> List[Document]:
        """
        Create a new database from the repository.

        Args:
            repo_url_or_path (str): The URL or local path of the repository
            repo_type (str): Type of repository (github, gitlab, bitbucket, azuredevops)
            access_token (str, optional): Access token for private repositories
            branch (str, optional): Branch name to clone/process
            embedder_type (str, optional): Kept for backward compatibility, ignored.
            is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively
            force_reprocess (bool): If True, ignore existing vectors and create fresh.
            repo_dir (str, optional): Pre-cloned repo directory. Skips clone when provided.

        Returns:
            List[Document]: List of Document objects
        """
        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token, branch,
                          force_reprocess=force_reprocess,
                          repo_dir=repo_dir)
        return self.prepare_db_index(
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
            force_reprocess=force_reprocess
        )

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        """Extract owner and repo name to create unique identifier.

        Delegates to _extract_owner_repo from code_processor for Azure DevOps
        URLs (handles both dev.azure.com and visualstudio.com formats).
        """
        from backend.processor.code_processor import _extract_owner_repo
        owner, repo = _extract_owner_repo(repo_url_or_path)
        return f"{owner}_{repo}"

    def _create_repo(
        self,
        repo_url_or_path: str,
        repo_type: str = None,
        access_token: str = None,
        branch: str = None,
        force_reprocess: bool = False,
        repo_dir: str = None
    ) -> None:
        """
        Download and prepare all paths.
        
        Path structure (consistent between local and blob):
        - Local root: ~/.adalflow/
        - Blob container: deepwiki-data (configured in infra.json)
        
        Storage paths:
        - Repos (local only): {root}/repos/{owner}_{repo_name}/
        - Vectors (local):    {root}/vectors/{owner}_{repo_name}_{branch}/
        - Vectors (blob):     vectors/{owner}_{repo_name}_{branch}/

        Args:
            repo_url_or_path (str): The URL or local path of the repository
            repo_type (str): Type of repository (github, gitlab, etc.)
            access_token (str, optional): Access token for private repos
            branch (str, optional): Branch name to clone/process (uses 'default' if not specified)
            force_reprocess (bool): If True, git pull latest changes for existing repo
            repo_dir (str, optional): Pre-cloned repo directory. When provided, skips
                clone/download and uses this path directly. The caller (e.g. step_clone)
                is responsible for ensuring the directory is up-to-date.
        """
        logger.info(f"Preparing repo storage for {repo_url_or_path}...")
        logger.debug(f"_create_repo params: repo_type={repo_type}, access_token={'[PROVIDED]' if access_token else '[NONE]'}, branch={branch}, force_reprocess={force_reprocess}")

        try:
            # Strip whitespace to handle URLs with leading/trailing spaces
            repo_url_or_path = repo_url_or_path.strip()
            
            root_path = get_adalflow_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # Extract the repository name from the URL
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type)
                logger.info(f"Extracted repo name: {repo_name}")

                # Include branch in clone dir so different branches don't clobber each other
                from backend.utils.filter import sanitize_branch_for_path
                branch_safe = sanitize_branch_for_path(branch or 'main')
                repo_dir_name = f"{repo_name}_{branch_safe}"

                save_repo_dir = repo_dir if repo_dir else os.path.join(root_path, "repos", repo_dir_name)
                blob_repo_path = f"repos/{repo_dir_name}/"

                if repo_dir:
                    # Pre-cloned by step_clone() — skip all clone/download logic
                    logger.info(f"Using pre-cloned repo at {repo_dir}")
                    # In blob mode, upload to blob for AML caching if not already there
                    if is_blob_storage_configured():
                        try:
                            blob_client = get_blob_storage_client()
                            if blob_client and not blob_client.directory_exists(blob_repo_path):
                                logger.info("Uploading pre-cloned repo to blob for caching")
                                blob_client.upload_directory(save_repo_dir, blob_repo_path)
                        except Exception as e:
                            logger.warning(f"Failed to upload repo to blob cache: {e}")
                elif is_blob_storage_configured():
                    # BLOB MODE: Check blob first, clone and upload to blob if not found
                    logger.info("Using Azure Blob Storage mode")
                    try:
                        blob_client = get_blob_storage_client()
                        if blob_client and blob_client.directory_exists(blob_repo_path):
                            logger.info(f"Repository found in Azure Blob Storage: {blob_repo_path}")
                            # Download from blob to local working directory
                            if not (os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)):
                                logger.info(f"Downloading repository from blob to {save_repo_dir}")
                                if not blob_client.download_directory(blob_repo_path, save_repo_dir):
                                    raise ConnectionError("Failed to download repository from blob")
                                logger.info("Repository downloaded from blob storage successfully")
                            elif force_reprocess:
                                # Pull latest changes and re-upload to blob
                                logger.info("Force reprocess: pulling latest changes")
                                try:
                                    download_repo(repo_url_or_path, save_repo_dir, repo_type,
                                                  access_token, branch, force_update=True)
                                    logger.info("Uploading updated repo to blob storage")
                                    blob_client.upload_directory(save_repo_dir, blob_repo_path)
                                except Exception as e:
                                    logger.warning(f"Git pull failed, using existing: {e}")
                            else:
                                logger.info("Repository already in local working directory")
                        else:
                            # Not in blob - clone fresh and upload to blob
                            logger.info("Repository not found in blob storage, cloning fresh")
                            download_repo(repo_url_or_path, save_repo_dir, repo_type,
                                          access_token, branch)
                            logger.info("Uploading cloned repository to blob storage")
                            if not blob_client.upload_directory(save_repo_dir, blob_repo_path):
                                logger.warning("Failed to upload repository to blob storage")
                    except ConnectionError:
                        raise
                    except Exception as e:
                        raise ConnectionError(f"Failed to access Azure Blob Storage: {e}") from e
                else:
                    # LOCAL MODE: Use download_repo with force_update flag
                    logger.info("Using local storage mode")
                    download_repo(repo_url_or_path, save_repo_dir, repo_type,
                                  access_token, branch, force_update=force_reprocess)
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path
                blob_repo_path = f"repos/{repo_name}/"

            # Normalize branch: detect actual branch if not provided
            if not branch or not branch.strip():
                if os.path.exists(save_repo_dir) and os.path.exists(os.path.join(save_repo_dir, ".git")):
                    try:
                        detected_branch = detect_default_branch(save_repo_dir)
                        logger.info(f"Detected default branch from repo: {detected_branch}")
                        branch = detected_branch
                    except Exception as e:
                        logger.warning(f"Could not detect default branch: {e}, using 'main'")
                        branch = 'main'
                else:
                    branch = 'main'
                    logger.info("Repo not cloned yet, defaulting to 'main' branch")
            
            branch_suffix = branch.strip() if branch and branch.strip() else 'main'
            
            os.makedirs(save_repo_dir, exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "blob_repo_path": blob_repo_path,
                "repo_name": repo_name,
                "branch_suffix": branch_suffix,
                "repo_type": repo_type,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.debug(f"Repo: {repo_name}, branch: {branch_suffix}, type: {repo_type}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(
        self,
        embedder_type: str = None,
        is_ollama_embedder: bool = None,
        excluded_dirs: List[str] = None,
        excluded_files: List[str] = None,
        included_dirs: List[str] = None,
        included_files: List[str] = None,
        force_reprocess: bool = False
    ) -> List[Document]:
        """
        Prepare the indexed database for the repository.
        
        Storage priority:
        1. If force_reprocess=True: Snapshot existing files, overwrite in-place,
           then delete orphans (zero-downtime reprocessing)
        2. Check for existing JSON vectors in "vectors/" - load if found
        3. If none exist, create new using JSON format in "vectors/"

        Zero-downtime reprocessing (force_reprocess=True):
            Instead of deleting all vectors upfront (which creates a window where
            the wiki has no embeddings), we:
            1. Snapshot the set of existing vector filenames
            2. Run embedding — new chunks overwrite existing files in-place
            3. After completion, delete orphan files (old files not in new set)
            This ensures the FAISS index is always populated during reprocessing.

        Args:
            embedder_type (str, optional): Kept for backward compatibility, ignored.
            is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively
            force_reprocess (bool): If True, ignore existing vectors and create fresh.

        Returns:
            List[Document]: List of Document objects
            
        Raises:
            ConnectionError: If Azure Blob Storage is configured but connection fails
        """
        repo_name = self.repo_paths.get("repo_name", "unknown")
        branch_suffix = self.repo_paths.get("branch_suffix", "main")
        vector_storage = get_vector_storage()
        old_files: set = set()  # Snapshot for orphan cleanup
        
        # ========================================================================
        # FORCE REPROCESS: Zero-downtime incremental overwrite + orphan cleanup
        # ========================================================================
        if force_reprocess:
            logger.info("[Vec] Force reprocess requested — zero-downtime mode")

            # Snapshot existing vector files BEFORE processing
            old_files = vector_storage.list_files(repo_name, branch_suffix)
            if old_files:
                logger.info(f"[Vec] Snapshot: {len(old_files)} existing vector files")
            else:
                logger.info("[Vec] No existing vectors (fresh run)")

            # Skip loading existing, go directly to creating new
        else:
            logger.info(f"Looking for existing embeddings for {repo_name} (branch: {branch_suffix})...")
        
            # ==================================================================
            # STEP 1: Check for existing vectors (only when not force_reprocess)
            # ==================================================================
            if self.repo_paths and is_blob_storage_configured():
                try:
                    blob_client = get_blob_storage_client()
                    if not blob_client:
                        error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
                        logger.error(error_msg)
                        raise ConnectionError(error_msg)
                    
                    if vector_storage.exists(repo_name, branch_suffix):
                        logger.info(f"[Vec] Found JSON vectors at: vectors/{repo_name}_{branch_suffix}/")
                        documents = vector_storage.load_documents(repo_name, branch_suffix)
                        if documents:
                            logger.info(f"[Vec] Successfully loaded {len(documents)} documents from JSON vector storage")
                            return documents
                        logger.info("[Vec] Vectors directory exists but empty/invalid, will create new")
                    else:
                        logger.info("[Vec] No existing vectors found, will create new")
                        
                except ConnectionError:
                    raise
                except Exception as e:
                    error_msg = f"Failed to connect to Azure Blob Storage: {e}"
                    logger.error(error_msg)
                    raise ConnectionError(error_msg) from e
            else:
                # Local storage mode
                if vector_storage.exists(repo_name, branch_suffix):
                    logger.info(f"[Vec] Found JSON vectors at: vectors/{repo_name}_{branch_suffix}/")
                    documents = vector_storage.load_documents(repo_name, branch_suffix)
                    if documents:
                        logger.info(f"[Vec] Successfully loaded {len(documents)} documents from JSON vector storage")
                        return documents

        # ========================================================================
        # STEP 2: Create new database using JSON format
        # ========================================================================
        logger.info("[Vec] Creating new embeddings with JSON vector storage...")

        # Fused read+split+embed: reads files in batches of 1000 to
        # avoid loading the entire repository into memory at once.
        # Returns (chunk_count, documents) — documents have vectors
        # attached, ready for FAISS construction without disk reload.
        chunk_count, transformed_docs = transform_documents_and_save_as_json(
            self.repo_paths["save_repo_dir"],
            repo_name,
            branch_suffix,
            repo_url=self.repo_url_or_path,
            repo_type=self.repo_paths.get("repo_type"),
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )

        if chunk_count == 0:
            logger.warning("No documents found to process")
            return []

        logger.info(
            f"[Vec] Embedded {chunk_count} chunks, "
            f"using returned docs for FAISS (no disk reload)"
        )

        # ====================================================================
        # STEP 3: Orphan cleanup (only during force_reprocess)
        # ====================================================================
        if force_reprocess and old_files:
            new_files = vector_storage.list_files(repo_name, branch_suffix)
            orphans = old_files - new_files
            if orphans:
                logger.info(
                    f"[Vec] Cleaning {len(orphans)} orphan files "
                    f"(old={len(old_files)}, new={len(new_files)})"
                )
                vector_storage.delete_files(repo_name, branch_suffix, orphans)
            else:
                logger.info("[Vec] No orphan files to clean up")

        return transformed_docs

    def prepare_retriever(
        self,
        repo_url_or_path: str,
        repo_type: str = None,
        access_token: str = None,
        branch: str = None
    ):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            type (str): Type of repository (github, gitlab, etc.)
            access_token (str, optional): Access token for private repos
            branch (str, optional): Branch name to clone/process

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_url_or_path, repo_type, access_token, branch)
