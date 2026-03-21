"""
Database management for RAG document storage.

Provides the DatabaseManager class for managing document databases.
"""

import os
import logging
from typing import List

from adalflow.core.types import Document
from adalflow.core.db import LocalDB
from backend.paths import get_adalflow_root_path

from backend.config import configs
from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.clients.vector_storage import get_vector_storage
from backend.modules.embedder.document import read_all_documents, transform_documents_and_save_as_json
from backend.modules.repository.git_ops import download_repo, detect_default_branch

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
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
        force_reprocess: bool = False
    ) -> List[Document]:
        """
        Create a new database from the repository.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            type (str): Type of repository (github, gitlab, bitbucket, azuredevops)
            access_token (str, optional): Access token for private repositories
            branch (str, optional): Branch name to clone/process
            embedder_type (str, optional): Kept for backward compatibility, ignored.
            is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively
            force_reprocess (bool): If True, ignore existing pkl/vectors and create fresh JSON vectors.
                                   Use this to migrate from pkl to vector-based storage.

        Returns:
            List[Document]: List of Document objects
        """
        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token, branch,
                          force_reprocess=force_reprocess)
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

    def _delete_legacy_pkl(self, repo_name: str, branch_suffix: str) -> None:
        """Delete legacy pkl database if it exists.

        Pkl databases are never served live, so deleting upfront is
        safe and doesn't affect zero-downtime guarantees.

        Args:
            repo_name: Repository name (owner_repo format)
            branch_suffix: Branch name
        """
        if is_blob_storage_configured():
            blob_db_path = self.repo_paths.get("blob_db_path") if self.repo_paths else None
            if blob_db_path:
                try:
                    blob_client = get_blob_storage_client()
                    if blob_client and blob_client.exists(blob_db_path):
                        logger.info(f"[Pkl] Deleting legacy pkl database: {blob_db_path}")
                        blob_client.delete_blob(blob_db_path)
                except Exception as e:
                    logger.warning(f"[Pkl] Failed to delete legacy pkl: {e}")
        else:
            if self.repo_paths and os.path.exists(self.repo_paths.get("save_db_file", "")):
                pkl_path = self.repo_paths["save_db_file"]
                logger.info(f"[Pkl] Deleting legacy pkl database: {pkl_path}")
                try:
                    os.remove(pkl_path)
                except Exception as e:
                    logger.warning(f"[Pkl] Failed to delete legacy pkl: {e}")

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
        """Extract owner and repo name to create unique identifier."""
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            # Bitbucket URL format: https://bitbucket.org/owner/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name

    def _create_repo(
        self,
        repo_url_or_path: str,
        repo_type: str = None,
        access_token: str = None,
        branch: str = None,
        force_reprocess: bool = False
    ) -> None:
        """
        Download and prepare all paths.
        
        Path structure (consistent between local and blob):
        - Local root: ~/.adalflow/
        - Blob container: deepwiki-data (configured in infra.json)
        
        Storage paths:
        - Repos (local only): {root}/repos/{owner}_{repo_name}/
        - Database (local):   {root}/databases/{owner}_{repo_name}_{branch}.pkl
        - Database (blob):    databases/{owner}_{repo_name}_{branch}.pkl

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            repo_type (str): Type of repository (github, gitlab, etc.)
            access_token (str, optional): Access token for private repos
            branch (str, optional): Branch name to clone/process (uses 'default' if not specified)
            force_reprocess (bool): If True, git pull latest changes for existing repo
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

                save_repo_dir = os.path.join(root_path, "repos", repo_name)
                blob_repo_path = f"repos/{repo_name}/"  # Blob prefix for repo files

                # Storage mode: blob OR local (no syncing between them)
                if is_blob_storage_configured():
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

            # Path consistency: local and blob use same relative structure
            # Local: ~/.adalflow/databases/{owner}_{repo}_{branch}.pkl
            # Blob:  databases/{owner}_{repo}_{branch}.pkl (same structure, different root)
            
            # Normalize branch: detect actual branch instead of using 'default'
            if not branch or not branch.strip():
                # Check if database already exists for main/master
                for candidate_branch in ['main', 'master']:
                    candidate_path = os.path.join(root_path, f"databases/{repo_name}_{candidate_branch}.pkl")
                    if os.path.exists(candidate_path):
                        logger.info(f"Found existing database for branch '{candidate_branch}', reusing it")
                        branch = candidate_branch
                        break
                
                # If no existing database found, try to detect from repo
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
                        # Default to main if repo doesn't exist yet
                        branch = 'main'
                        logger.info("Repo not cloned yet, defaulting to 'main' branch")
            
            branch_suffix = branch.strip() if branch and branch.strip() else 'main'
            db_relative_path = f"databases/{repo_name}_{branch_suffix}.pkl"
            save_db_file = os.path.join(root_path, db_relative_path)
            blob_db_path = db_relative_path  # Same relative path for blob
            
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
                "blob_db_path": blob_db_path,
                "blob_repo_path": blob_repo_path,  # Add blob repo path
                "repo_name": repo_name,  # Store for reference
                "branch_suffix": branch_suffix,  # Store branch suffix for vector storage
                "repo_type": repo_type,  # Store repo type for URL construction
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
        
        Storage priority (backward compatible):
        1. If force_reprocess=True: Snapshot existing files, overwrite in-place,
           then delete orphans (zero-downtime reprocessing)
        2. Check for existing pkl database in "databases/" - load if found (backward compat)
        3. Check for existing JSON vectors in "vectors/" - load if found (new format)
        4. If neither exists, create new using JSON format in "vectors/"

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
            force_reprocess (bool): If True, ignore existing pkl/vectors and create fresh JSON vectors.
                                   Use this to migrate from pkl to vector-based storage.

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

            # Delete legacy pkl database (always safe — pkl is not served live)
            self._delete_legacy_pkl(repo_name, branch_suffix)

            # Skip STEP 1 (loading existing), go directly to STEP 2 (create new)
            # New vectors will overwrite existing files in-place
        else:
            logger.info(f"Looking for existing embeddings for {repo_name} (branch: {branch_suffix})...")
        
            # ==================================================================
            # STEP 1: Check for existing storage (only when not force_reprocess)
            # ==================================================================
            if self.repo_paths and is_blob_storage_configured():
                blob_db_path = self.repo_paths.get("blob_db_path")
                if blob_db_path:
                    try:
                        blob_client = get_blob_storage_client()
                        if not blob_client:
                            error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
                            logger.error(error_msg)
                            raise ConnectionError(error_msg)
                        
                        # Check for legacy pkl database first
                        if blob_client.exists(blob_db_path):
                            logger.info(f"[Pkl] Found legacy pkl database in Azure Blob: {blob_db_path}")
                            self.db = blob_client.load_pickle(blob_db_path)
                            if self.db:
                                documents = self.db.get_transformed_data(key="split_and_embed")
                                if documents:
                                    logger.info(f"[Pkl] Successfully loaded {len(documents)} documents from legacy pkl database")
                                    return documents
                            logger.info("[Pkl] Legacy pkl exists but is empty/invalid, switching to check vectors...")
                        
                        # Check for new JSON vectors
                        if vector_storage.exists(repo_name, branch_suffix):
                            logger.info(f"[Vec] Found JSON vectors at: vectors/{repo_name}_{branch_suffix}/")
                            documents = vector_storage.load_documents(repo_name, branch_suffix)
                            if documents:
                                logger.info(f"[Vec] Successfully loaded {len(documents)} documents from JSON vector storage")
                                return documents
                            logger.info("[Vec] Vectors directory exists but empty/invalid, will create new")
                        else:
                            logger.info("[Vec] No existing vectors found, will create new with JSON format")
                            
                    except ConnectionError:
                        raise  # Re-raise connection errors
                    except Exception as e:
                        error_msg = f"Failed to connect to Azure Blob Storage: {e}"
                        logger.error(error_msg)
                        raise ConnectionError(error_msg) from e
            else:
                # Local storage mode (blob not configured)
                # Check for legacy pkl database first
                if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
                    logger.info(f"[Pkl] Found legacy pkl database at local: {self.repo_paths['save_db_file']}")
                    try:
                        self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                        documents = self.db.get_transformed_data(key="split_and_embed")
                        if documents:
                            logger.info(f"[Pkl] Successfully loaded {len(documents)} documents from legacy pkl database")
                            return documents
                    except Exception as e:
                        logger.error(f"[Pkl] Error loading legacy pkl database: {e}")
                        logger.info("[Pkl] Switching to check vectors...")
                
                # Check for new JSON vectors
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
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"],
            repo_url=self.repo_url_or_path,
            repo_type=self.repo_paths.get("repo_type"),
            branch=self.repo_paths.get("branch_suffix"),
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        logger.info(f"[Vec] Processing {len(documents)} documents...")
        
        # Use new JSON format for storage
        transformed_docs = transform_documents_and_save_as_json(
            documents,
            repo_name,
            branch_suffix
        )
        
        logger.info(f"[Vec] Total documents: {len(documents)}")
        logger.info(f"[Vec] Total transformed chunks: {len(transformed_docs)}")

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
