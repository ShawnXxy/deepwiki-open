"""
Database management for RAG document storage.

Provides the DatabaseManager class for managing document databases.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List

from adalflow.core.types import Document
from backend.paths import get_adalflow_root_path

from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.clients.vector_storage import get_vector_storage
from backend.modules.embedder.document import transform_documents_and_save_as_json
from backend.modules.embedder.delta import (
    compute_file_delta,
    summarize_for_log,
    CURRENT_MANIFEST_VERSION,
)
from backend.modules.repository.git_ops import (
    download_repo,
    detect_default_branch,
    get_head_commit_hash,
)

logger = logging.getLogger(__name__)


def _current_embedder_signature() -> Dict[str, object]:
    """Read deployment / dimensions of the embedder that will be used now."""
    try:
        from backend.config import get_embedder_config_obj
        cfg = get_embedder_config_obj()
        kw = cfg.embedder.model_kwargs
        return {
            "deployment": getattr(kw, "model", "") or "",
            "model_name": getattr(kw, "model", "") or "",
            "vector_dim": int(getattr(kw, "dimensions", 0) or 0),
        }
    except Exception as e:
        logger.warning(f"[Delta] Could not read embedder signature: {e}")
        return {"deployment": "", "model_name": "", "vector_dim": 0}


def _walk_candidate_files(
    repo_dir: str,
    excluded_dirs: List[str] = None,
    excluded_files: List[str] = None,
    included_dirs: List[str] = None,
    included_files: List[str] = None,
) -> List[tuple]:
    """Reproduce the file-walk filter logic from
    ``transform_documents_and_save_as_json`` without doing any I/O on file
    contents.

    Returns ``[(full_path, relative_path, ext, is_code), ...]`` matching the
    structure the embedder consumes.
    """
    from backend.config import get_file_filters_config, get_included_config
    from backend.types import FileFilter
    from backend.utils.filter import load_gitignore, is_gitignored

    if not repo_dir or not os.path.isdir(repo_dir):
        return []

    included = get_included_config()
    code_extensions = set(included["code"])
    doc_extensions = set(included["doc"])
    all_ext_set = code_extensions | doc_extensions

    has_dirs = included_dirs is not None and len(included_dirs) > 0
    has_files = included_files is not None and len(included_files) > 0
    if has_dirs or has_files:
        file_filter = FileFilter(
            included_dirs=set(included_dirs) if included_dirs else set(),
            included_patterns=set(included_files) if included_files else set(),
        )
    else:
        file_filters = get_file_filters_config()
        final_excluded_dirs = set(file_filters["excluded_dirs"])
        final_excluded_patterns = set(file_filters["excluded_files"])
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)
        if excluded_files is not None:
            final_excluded_patterns.update(excluded_files)
        file_filter = FileFilter(
            excluded_dirs=final_excluded_dirs,
            excluded_patterns=final_excluded_patterns,
        )

    gitignore_spec = load_gitignore(repo_dir)
    walk_excluded_dirs = set(get_file_filters_config()["excluded_dirs"])

    file_infos: List[tuple] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in walk_excluded_dirs]
        for fname in files:
            full_path = os.path.join(root, fname)
            relative_path = os.path.relpath(full_path, repo_dir)
            if is_gitignored(gitignore_spec, relative_path):
                continue
            if not file_filter.should_process_file(relative_path):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in all_ext_set:
                continue
            is_code = ext in code_extensions
            file_infos.append(
                (full_path, relative_path.replace("\\", "/"), ext, is_code)
            )
    return file_infos


def _build_manifest(
    prev_manifest,
    delta,
    chunks_by_source_new: Dict[str, List[str]],
    current_commit_hash: str,
    embedder_sig: Dict[str, object],
) -> Dict[str, object]:
    """Compose the new manifest from the previous manifest + delta result.

    Carries unchanged entries forward verbatim, replaces / adds entries for
    embedded files, and drops entries listed in ``delta.to_delete``.
    """
    prev_files: Dict[str, Dict[str, object]] = {}
    if prev_manifest:
        prev_files = dict(prev_manifest.get("files") or {})

    # Drop deleted / replaced entries.
    for src in delta.to_delete:
        prev_files.pop(src, None)

    # Add / replace entries for freshly embedded sources.
    for src, chunk_files in chunks_by_source_new.items():
        norm = src.replace("\\", "/")
        sha = delta.file_hashes.get(norm) or delta.file_hashes.get(src) or ""
        entry: Dict[str, object] = {
            "sha256": sha,
            "chunk_count": len(chunk_files),
            "chunk_files": sorted(chunk_files),
        }
        # Pick up file size opportunistically — best-effort, never blocks.
        for fi in delta.to_embed:
            if fi[1].replace("\\", "/") == norm:
                try:
                    entry["size"] = os.path.getsize(fi[0])
                except OSError:
                    pass
                break
        prev_files[norm] = entry

    # Update hashes for unchanged files when delta produced new ones (e.g.
    # the fast-path verify-hash code path).
    for rel, sha in (delta.file_hashes or {}).items():
        if rel in prev_files and not prev_files[rel].get("sha256"):
            prev_files[rel]["sha256"] = sha

    return {
        "version": CURRENT_MANIFEST_VERSION,
        "commit_hash": current_commit_hash or "",
        "embedded_at": datetime.now(timezone.utc).isoformat(),
        "embedder": dict(embedder_sig),
        "files": prev_files,
    }


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
            
            from backend.utils.filter import sanitize_branch_for_path
            branch_suffix = sanitize_branch_for_path(branch, default='main')
            
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
        """Prepare the indexed database for the repository.

        Uses an embedding manifest sidecar (``vectors/<owner>_<repo>_<branch>/
        _manifest.json``) to compute a precise file-level delta against the
        previous run. Only changed / added files are re-embedded; chunks for
        deleted or renamed files are removed. Unchanged files are reused
        in-place.

        Args:
            force_reprocess: When True, ignores any existing manifest and
                re-embeds every file (full rebuild). The previous chunks are
                left in place during embedding for zero-downtime, then
                orphans are cleaned up.
        """
        repo_name = self.repo_paths.get("repo_name", "unknown")
        branch_suffix = self.repo_paths.get("branch_suffix", "main")
        save_repo_dir = self.repo_paths.get("save_repo_dir")
        vector_storage = get_vector_storage()

        # ----- Read previous manifest (unless force_reprocess) -----
        prev_manifest = None
        if not force_reprocess:
            prev_manifest = vector_storage.read_manifest(repo_name, branch_suffix)
            if prev_manifest:
                logger.info(
                    f"[Vec] Found previous manifest: commit="
                    f"{str(prev_manifest.get('commit_hash') or '')[:7]}, "
                    f"files={len(prev_manifest.get('files') or {})}"
                )
            else:
                logger.info("[Vec] No previous manifest — cold start")
        else:
            logger.info("[Vec] force_reprocess=True — ignoring manifest")

        # ----- Resolve current commit hash for delta detection -----
        current_commit_hash = ""
        if save_repo_dir and os.path.isdir(os.path.join(save_repo_dir, ".git")):
            try:
                current_commit_hash = get_head_commit_hash(save_repo_dir)
            except Exception as e:
                logger.warning(f"[Delta] Could not read HEAD commit: {e}")

        # ----- Collect candidate files (lightweight walk only) -----
        # The same walk happens inside ``transform_documents_and_save_as_json``;
        # we replicate it here just to compute the delta. The cost is a
        # directory scan (no file reads, no embedding).
        candidate_file_infos = _walk_candidate_files(
            save_repo_dir,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )

        # ----- Compute delta -----
        embedder_sig = _current_embedder_signature()
        delta = compute_file_delta(
            repo_dir=save_repo_dir,
            file_infos=candidate_file_infos,
            prev_manifest=prev_manifest if not force_reprocess else None,
            current_embedder=embedder_sig,
            current_commit_hash=current_commit_hash,
            verify_hash=True,
        )
        logger.info(f"[Vec] Delta summary: {summarize_for_log(delta)}")

        # ----- Fast path: nothing to embed, nothing to delete -----
        if (
            prev_manifest
            and not delta.to_embed
            and not delta.to_delete
            and vector_storage.exists(repo_name, branch_suffix)
        ):
            logger.info(
                "[Vec] No changes detected — reusing existing embeddings"
            )
            documents = vector_storage.load_documents(repo_name, branch_suffix)
            if documents:
                logger.info(
                    f"[Vec] Loaded {len(documents)} cached documents from "
                    f"vector storage"
                )
                # Opportunistic manifest self-heal: if the fast-path verifier
                # produced sha256s for entries that were missing them in the
                # old manifest, persist the updated manifest now. Otherwise
                # the next run cannot do hash-based mismatch detection.
                prev_files_map = prev_manifest.get("files") or {}
                missing_hashes = [
                    rel for rel, sha in delta.file_hashes.items()
                    if rel in prev_files_map and not prev_files_map[rel].get("sha256")
                ]
                if missing_hashes:
                    try:
                        healed = _build_manifest(
                            prev_manifest=prev_manifest,
                            delta=delta,
                            chunks_by_source_new={},
                            current_commit_hash=current_commit_hash,
                            embedder_sig=embedder_sig,
                        )
                        vector_storage.write_manifest(repo_name, branch_suffix, healed)
                        logger.info(
                            f"[Vec] Backfilled sha256 for {len(missing_hashes)} "
                            f"manifest entries"
                        )
                    except Exception as e:
                        logger.warning(f"[Vec] Manifest backfill failed (non-fatal): {e}")
                return documents
            logger.warning(
                "[Vec] Manifest is clean but vector store has no documents; "
                "falling through to full reprocess"
            )
            delta.to_embed = list(candidate_file_infos)
            delta.unchanged = set()
            delta.reason = "manifest_clean_but_empty_store"

        # ----- Drop chunks for files that were deleted or replaced -----
        if delta.to_delete and prev_manifest:
            removed = vector_storage.delete_files_for_sources(
                repo_name, branch_suffix, delta.to_delete, prev_manifest,
            )
            logger.info(
                f"[Vec] Removed {removed} chunk files for "
                f"{len(delta.to_delete)} deleted/changed sources"
            )

        # ----- Embed only the to_embed slice -----
        delta_paths = {
            fi[1].replace("\\", "/") for fi in delta.to_embed
        } if not force_reprocess else None

        if not delta.to_embed and not force_reprocess:
            # Nothing to embed (e.g. only deletions). Skip the call entirely
            # so we don't pay the file-walk cost twice.
            chunk_count = 0
            transformed_docs: List[Document] = []
            chunks_by_source_new: Dict[str, List[str]] = {}
        else:
            chunk_count, transformed_docs, chunks_by_source_new = (
                transform_documents_and_save_as_json(
                    save_repo_dir,
                    repo_name,
                    branch_suffix,
                    repo_url=self.repo_url_or_path,
                    repo_type=self.repo_paths.get("repo_type"),
                    excluded_dirs=excluded_dirs,
                    excluded_files=excluded_files,
                    included_dirs=included_dirs,
                    included_files=included_files,
                    delta_to_embed=delta_paths,
                    return_chunks_by_source=True,
                )
            )

        # ----- Write the new manifest -----
        try:
            new_manifest = _build_manifest(
                prev_manifest=prev_manifest if not force_reprocess else None,
                delta=delta,
                chunks_by_source_new=chunks_by_source_new,
                current_commit_hash=current_commit_hash,
                embedder_sig=embedder_sig,
            )
            vector_storage.write_manifest(repo_name, branch_suffix, new_manifest)
        except Exception as e:
            logger.warning(f"[Vec] Failed to write manifest (non-fatal): {e}")

        # ----- Hand back full document set for FAISS -----
        if chunk_count == 0 and not delta.unchanged:
            logger.warning("No documents found to process")
            return []

        # Load any unchanged documents so the caller (FAISS) gets the
        # complete set. ``transformed_docs`` only contains the freshly
        # embedded slice.
        if delta.unchanged and prev_manifest:
            logger.info(
                f"[Vec] Loading {len(delta.unchanged)} unchanged source "
                f"files from vector store"
            )
            for batch in vector_storage.iter_documents_for_sources(
                repo_name, branch_suffix, delta.unchanged, prev_manifest,
            ):
                transformed_docs.extend(batch)

        logger.info(
            f"[Vec] Index ready: {len(transformed_docs)} total documents "
            f"({chunk_count} freshly embedded)"
        )
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
