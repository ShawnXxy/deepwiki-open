"""
Document processing and transformation for RAG.

Provides functions for reading, splitting, and embedding documents.
"""

import gc
import os
import logging
from typing import List

import adalflow as adal
from adalflow.core.types import Document
from adalflow.components.data_process import TextSplitter, ToEmbeddings
from adalflow.core.db import LocalDB

from backend.config import (
    configs, get_file_filters_config,
)
from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.clients.vector_storage import get_vector_storage
from backend.types import FileFilter
from backend.clients.embedding_client import get_embedder
from backend.modules.embedder.tokenizer import safe_read_file, count_tokens, MAX_EMBEDDING_TOKENS
from backend.modules.embedder.code_splitter import split_and_enrich_documents

logger = logging.getLogger(__name__)


def read_all_documents(
    path: str,
    repo_url: str = None,
    repo_type: str = None,
    branch: str = None,
    embedder_type: str = None,
    is_ollama_embedder: bool = None,
    excluded_dirs: List[str] = None,
    excluded_files: List[str] = None,
    included_dirs: List[str] = None,
    included_files: List[str] = None
) -> List[Document]:
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.
        repo_url (str, optional): The URL of the repository (for creating links in metadata).
        repo_type (str, optional): The type of repository (github, azuredevops, etc.).
        branch (str, optional): The branch name (for creating links in metadata).
        embedder_type (str, optional): Kept for backward compatibility, ignored.
        is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.
        excluded_dirs (List[str], optional): List of directories to exclude from processing.
            Overrides the default configuration if provided.
        excluded_files (List[str], optional): List of file patterns to exclude from processing.
            Overrides the default configuration if provided.
        included_dirs (List[str], optional): List of directories to include exclusively.
            When provided, only files in these directories will be processed.
        included_files (List[str], optional): List of file patterns to include exclusively.
            When provided, only files matching these patterns will be processed.

    Returns:
        list: A list of Document objects with metadata.
    """
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # Initialize FileFilter with inclusion/exclusion rules
    has_dirs = included_dirs is not None and len(included_dirs) > 0
    has_files = included_files is not None and len(included_files) > 0
    if has_dirs or has_files:
        # Inclusion mode: only process specified directories and files
        file_filter = FileFilter(
            included_dirs=set(included_dirs) if included_dirs else set(),
            included_patterns=set(included_files) if included_files else set(),
            max_file_size_mb=10
        )
        logger.info("Using inclusion mode")
        logger.info(f"Included directories: {list(file_filter.included_dirs)}")
        logger.info(f"Included patterns: {list(file_filter.included_patterns)}")
    else:
        # Exclusion mode: load filters from repo.json (single source of truth)
        file_filters = get_file_filters_config()
        final_excluded_dirs = set(file_filters["excluded_dirs"])
        final_excluded_patterns = set(file_filters["excluded_files"])

        # Add any explicitly provided excluded directories and files
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)
        if excluded_files is not None:
            final_excluded_patterns.update(excluded_files)

        file_filter = FileFilter(
            excluded_dirs=final_excluded_dirs,
            excluded_patterns=final_excluded_patterns,
            max_file_size_mb=10
        )
        logger.info("Using exclusion mode")
        logger.info(f"Excluded directories: {list(file_filter.excluded_dirs)}")
        logger.info(f"Excluded patterns: {list(file_filter.excluded_patterns)}")

    logger.info(f"Reading documents from {path}")

    # Build extension sets for O(1) lookup
    code_ext_set = set(code_extensions)
    doc_ext_set = set(doc_extensions)
    all_ext_set = code_ext_set | doc_ext_set

    def _compute_file_url(relative_path: str) -> str:
        """Helper to compute the file URL based on repo settings."""
        if not repo_url or not repo_url.startswith(("http://", "https://")):
            return relative_path

        # Normalize relative path (forward slashes)
        file_path = relative_path.replace("\\", "/")
        if not file_path.startswith("/"):
            file_path = "/" + file_path

        branch_name = branch if branch and branch.strip() else "main"

        if repo_type == "azuredevops":
            # Format: .../_git/repo?version=GB{branch}&path=/{path}
            return f"{repo_url.rstrip('/')}?version=GB{branch_name}&path={file_path}"

        elif repo_type in ["github", "gitlab"]:
            # Format: .../blob/{branch}/{path}
            base_url = repo_url
            if base_url.endswith(".git"):
                base_url = base_url[:-4]
            return f"{base_url.rstrip('/')}/blob/{branch_name}{file_path}"

        elif repo_type == "bitbucket":
            # Format: .../src/{branch}/{path}
            base_url = repo_url
            if base_url.endswith(".git"):
                base_url = base_url[:-4]
            return f"{base_url.rstrip('/')}/src/{branch_name}{file_path}"

        return relative_path

    # Single os.walk() pass — replaces 22 separate glob traversals
    # Collect code files and doc files in one traversal
    skip_dirs = {
        '.git', 'node_modules', '__pycache__', '.venv', 'venv',
        'dist', 'build', '.next', '.nuxt', 'coverage', '.tox',
        'egg-info', '.eggs',
    }
    code_files = []
    doc_files = []

    for root, dirs, files in os.walk(path):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in all_ext_set:
                continue
            full_path = os.path.join(root, fname)
            if ext in code_ext_set:
                code_files.append((full_path, ext))
            else:
                doc_files.append((full_path, ext))

    # Process code files first (higher priority for embedding)
    for file_path, ext in code_files:
        relative_path = os.path.relpath(file_path, path)
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            continue
        if not file_filter.should_process_file(relative_path, file_size):
            continue

        try:
            content = safe_read_file(file_path)

            is_implementation = (
                not relative_path.startswith("test_")
                and not relative_path.startswith("app_")
                and "test" not in relative_path.lower()
            )

            token_count = count_tokens(content, embedder_type)
            if token_count > MAX_EMBEDDING_TOKENS * 10:
                logger.info(
                    f"Large code file {relative_path}: "
                    f"{token_count} tokens "
                    f"(will be split into ~{token_count // 2000} chunks)"
                )

            doc = Document(
                text=content,
                meta_data={
                    "file_path": relative_path,
                    "type": ext[1:],
                    "is_code": True,
                    "is_implementation": is_implementation,
                    "url": _compute_file_url(relative_path),
                    "token_count": token_count,
                },
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"[BE] Error reading {file_path}: {e}")

    # Then process documentation files
    for file_path, ext in doc_files:
        relative_path = os.path.relpath(file_path, path)
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            continue
        if not file_filter.should_process_file(relative_path, file_size):
            continue

        try:
            content = safe_read_file(file_path)

            token_count = count_tokens(content, embedder_type)
            if token_count > MAX_EMBEDDING_TOKENS * 10:
                logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                continue

            doc = Document(
                text=content,
                meta_data={
                    "file_path": relative_path,
                    "type": ext[1:],
                    "is_code": False,
                    "is_implementation": False,
                    "url": _compute_file_url(relative_path),
                    "token_count": token_count,
                },
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"[BE] Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents


def prepare_data_pipeline(embedder_type: str = None, is_ollama_embedder: bool = None):
    """
    Creates and returns the data transformation pipeline.
    DEPRECATED: Used only for legacy pkl format. New code uses
    prepare_embed_only_pipeline() with code-aware pre-splitting.

    Args:
        embedder_type (str, optional): Kept for backward compatibility, ignored.
        is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.

    Returns:
        adal.Sequential: The data transformation pipeline
    """
    from backend.config import get_embedder_config

    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()

    embedder = get_embedder()

    # Use batch processing for Azure OpenAI embeddings
    batch_size = embedder_config.get("batch_size", 500)
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=batch_size
    )

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer


def prepare_embed_only_pipeline():
    """
    Creates a pipeline that only embeds (no splitting).

    Used with code-aware pre-splitting where documents
    are already split into enriched chunks before embedding.

    Returns:
        ToEmbeddings: The embedding transformer
    """
    from backend.config import get_embedder_config

    embedder_config = get_embedder_config()
    embedder = get_embedder()
    batch_size = embedder_config.get("batch_size", 500)

    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=batch_size
    )
    return embedder_transformer


def transform_documents_and_save_to_db(
    documents: List[Document],
    db_path: str,
    embedder_type: str = None,
    is_ollama_embedder: bool = None,
    blob_path: str = None
) -> LocalDB:
    """
    Transforms a list of documents and saves them to storage (Azure Blob or local).

    DEPRECATED: This function uses pickle format. New code should use
    transform_documents_and_save_as_json() for memory-efficient JSON storage.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file (used as fallback or for local storage).
        embedder_type (str, optional): Kept for backward compatibility, ignored.
        is_ollama_embedder (bool, optional): DEPRECATED. Kept for backward compatibility.
        blob_path (str, optional): Path in blob storage (e.g., "databases/owner_repo.pkl")

    Returns:
        LocalDB: The transformed database
    """
    # Get the data transformer
    data_transformer = prepare_data_pipeline()

    # Save the documents to a local database
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")

    # Save to Azure Blob Storage when configured (no fallback to local)
    if blob_path and is_blob_storage_configured():
        try:
            blob_client = get_blob_storage_client()
            if not blob_client:
                error_msg = "Azure Blob Storage is configured but failed to create client. Check MSI configuration."
                logger.error(error_msg)
                raise ConnectionError(error_msg)

            if blob_client.save_pickle(blob_path, db):
                logger.info(f"Database saved to Azure Blob Storage: {blob_path}")
                return db
            else:
                error_msg = f"Failed to save database to Azure Blob Storage: {blob_path}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
        except ConnectionError:
            raise  # Re-raise connection errors
        except Exception as e:
            error_msg = f"Failed to connect to Azure Blob Storage for saving: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    # Local storage mode (blob not configured)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    logger.info(f"Database saved to local storage: {db_path}")
    return db


def transform_documents_and_save_as_json(
    documents: List[Document],
    repo_name: str,
    branch: str,
    progress_callback: callable = None
) -> List[Document]:
    """
    Transforms documents and saves them as JSON chunk files.

    Uses code-aware splitting (processor_design approach) for code files:
    - Splits at function/class boundaries instead of arbitrary token positions
    - Enriches embedding text with file context and structural metadata
    - Extracts code elements (functions, classes) for richer retrieval

    For documentation files:
    - Splits at heading/paragraph boundaries
    - Adds file path context to embedding text

    Storage structure:
        vectors/{repo_name}_{branch}/
            └── {source_file_path}_chunk_001.json
            └── {source_file_path}_chunk_002.json
            ...

    Args:
        documents: List of Document objects (raw, before splitting)
        repo_name: Repository name (owner_repo format)
        branch: Branch name
        progress_callback: Optional callback(saved, total) for progress updates

    Returns:
        List of transformed Document objects with embeddings

    Raises:
        ConnectionError: If Azure Blob Storage is configured but connection fails
        ValueError: If transformation or saving fails
    """
    logger.info(
        f"[Vec] Transforming {len(documents)} documents for "
        f"{repo_name}_{branch} (code-aware splitting)"
    )

    # Step 1: Code-aware splitting + structural enrichment
    # This replaces the naive TextSplitter with boundary-aware splitting
    # and enriches each chunk with file context for better embedding quality
    enriched_chunks = split_and_enrich_documents(documents)

    if not enriched_chunks:
        logger.warning("[Vec] No chunks after splitting")
        return []

    logger.info(
        f"[Vec] Code-aware split: {len(documents)} files -> "
        f"{len(enriched_chunks)} enriched chunks"
    )

    # Release raw documents — content is now in enriched_chunks
    del documents
    gc.collect()

    # Step 2: Embed the enriched chunks (splitting already done)
    embed_pipeline = prepare_embed_only_pipeline()

    db = LocalDB()
    db.register_transformer(
        transformer=embed_pipeline, key="embed_only"
    )
    db.load(enriched_chunks)

    import time
    batch_size = configs.get("embedder", {}).get("batch_size", 10)
    total_batches = (len(enriched_chunks) + batch_size - 1) // batch_size
    logger.info(
        f"[Vec] Starting embedding: {len(enriched_chunks)} chunks "
        f"in ~{total_batches} batches (batch_size={batch_size})"
    )
    embed_start = time.time()

    db.transform(key="embed_only")

    embed_elapsed = time.time() - embed_start
    logger.info(
        f"[Vec] Embedding completed in {embed_elapsed:.1f}s "
        f"({len(enriched_chunks) / max(embed_elapsed, 0.1):.0f} "
        f"chunks/sec)"
    )

    # Get transformed documents with embeddings
    transformed_docs = db.get_transformed_data(key="embed_only")

    # Release LocalDB and enriched_chunks — data is in transformed_docs now
    del db
    del enriched_chunks
    gc.collect()

    if not transformed_docs:
        logger.warning("[Vec] No documents after embedding")
        return []

    logger.info(
        f"[Vec] Generated embeddings for {len(transformed_docs)} chunks, "
        f"saving as JSON..."
    )

    # Step 4: Save to JSON vector storage
    vector_storage = get_vector_storage()

    if not vector_storage.save_documents(
        transformed_docs,
        repo_name,
        branch,
        progress_callback=progress_callback
    ):
        raise ValueError(
            f"[Vec] Failed to save vectors for {repo_name}_{branch}"
        )

    logger.info(
        f"[Vec] Successfully saved {len(transformed_docs)} "
        f"enriched chunks as JSON"
    )
    return transformed_docs
