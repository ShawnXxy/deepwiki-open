"""
Document processing and transformation for RAG.

Provides functions for reading, splitting, and embedding documents.
"""

import gc
import os
import re
import logging
from typing import Dict, List, Optional, Set, Tuple

import adalflow as adal
from adalflow.core.types import Document
from adalflow.components.data_process import ToEmbeddings

from backend.config import (
    configs, get_file_filters_config, get_included_config,
)
from backend.clients.blob_client import get_blob_storage_client, is_blob_storage_configured
from backend.clients.vector_storage import get_vector_storage
from backend.types import FileFilter
from backend.clients.embedding_client import get_embedder
from backend.modules.embedder.tokenizer import safe_read_file, count_tokens, MAX_EMBEDDING_TOKENS
from backend.modules.embedder.code_splitter import split_and_enrich_documents
from backend.utils.filter import load_gitignore, is_gitignored

logger = logging.getLogger(__name__)


# Per-file size caps (bytes). Files above the cap for their category are
# skipped before tokenization. Stops a single multi-MB SQL dump or
# generated CSV from blocking the embedding pass and OOMing AML compute.
_SIZE_CAP_BYTES_CODE = 1_000_000     # 1 MB for source code
_SIZE_CAP_BYTES_DATA = 500_000       # 500 KB for json/yaml/xml
_SIZE_CAP_BYTES_DOC = 200_000        # 200 KB for markdown / rst

_DATA_EXTENSIONS = {'.json', '.yaml', '.yml', '.xml', '.xaml'}


def _size_cap_for(ext: str, is_code: bool) -> int:
    """Return the per-file size cap (bytes) for an extension."""
    if ext in _DATA_EXTENSIONS:
        return _SIZE_CAP_BYTES_DATA
    if is_code:
        return _SIZE_CAP_BYTES_CODE
    return _SIZE_CAP_BYTES_DOC


# Path patterns that mark a file as a test / fixture / mock. A file
# matching any of these gets `is_implementation=False` and is dropped
# from the index entirely when the caller is in wiki-gen mode (default).
_TEST_PATH_PATTERNS = [
    re.compile(r'(^|/)tests?(/|$)', re.IGNORECASE),
    re.compile(r'(^|/)e2e(/|$)', re.IGNORECASE),
    re.compile(r'(^|/)__mocks__(/|$)'),
    re.compile(r'(^|/)__fixtures__(/|$)'),
    re.compile(r'(^|/)__snapshots__(/|$)'),
    re.compile(r'(^|/)conftest\.py$'),
    re.compile(r'(^|/)[^/]+\.test\.[A-Za-z0-9]+$'),
    re.compile(r'(^|/)[^/]+\.spec\.[A-Za-z0-9]+$'),
    re.compile(r'(^|/)[^/]+_test\.[A-Za-z0-9]+$'),
    re.compile(r'(^|/)test_[^/]+\.py$'),
]


def _is_test_path(relative_path: str) -> bool:
    """True if the file path matches a test / fixture / mock convention."""
    norm = relative_path.replace('\\', '/')
    return any(pat.search(norm) for pat in _TEST_PATH_PATTERNS)


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

    # Load supported extensions from included.json (single source of truth)
    included = get_included_config()
    code_ext_set = set(included["code"])
    doc_ext_set = set(included["doc"])
    all_ext_set = code_ext_set | doc_ext_set

    # Initialize FileFilter with inclusion/exclusion rules
    has_dirs = included_dirs is not None and len(included_dirs) > 0
    has_files = included_files is not None and len(included_files) > 0
    if has_dirs or has_files:
        # Inclusion mode: only process specified directories and files
        file_filter = FileFilter(
            included_dirs=set(included_dirs) if included_dirs else set(),
            included_patterns=set(included_files) if included_files else set(),
        )
        logger.info("Using inclusion mode")
        logger.info(f"Included directories: {list(file_filter.included_dirs)}")
        logger.info(f"Included patterns: {list(file_filter.included_patterns)}")
    else:
        # Exclusion mode: load filters from excluded.json (single source of truth)
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
        )
        logger.info("Using exclusion mode")
        logger.info(f"Excluded directories: {list(file_filter.excluded_dirs)}")
        logger.info(f"Excluded patterns: {list(file_filter.excluded_patterns)}")

    logger.info(f"Reading documents from {path}")

    # Load .gitignore from the repo for dynamic filtering
    gitignore_spec = load_gitignore(path)

    # Load excluded_dirs for os.walk() pruning (single source of truth from excluded.json)
    file_filters_cfg = get_file_filters_config()
    walk_excluded_dirs = set(file_filters_cfg["excluded_dirs"])

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

    # Single os.walk() pass with two-layer filtering
    # Layer 1: Exclusion (.gitignore + excluded.json)
    # Layer 2: Inclusion (included.json supported extensions)
    code_files = []
    doc_files = []
    skipped_gitignore = 0
    skipped_excluded = 0
    skipped_ext = 0

    for root, dirs, files in os.walk(path):
        # Prune excluded directories in-place (from excluded.json)
        dirs[:] = [d for d in dirs if d not in walk_excluded_dirs]
        for fname in files:
            full_path = os.path.join(root, fname)
            relative_path = os.path.relpath(full_path, path)

            # Layer 1a: Check .gitignore
            if is_gitignored(gitignore_spec, relative_path):
                skipped_gitignore += 1
                logger.debug(f"Skipped (gitignored): {relative_path}")
                continue

            # Layer 1b: Check excluded file patterns
            if not file_filter.should_process_file(relative_path):
                skipped_excluded += 1
                logger.debug(f"Skipped (excluded): {relative_path}")
                continue

            # Layer 2: Check included extensions
            ext = os.path.splitext(fname)[1].lower()
            if ext not in all_ext_set:
                skipped_ext += 1
                logger.debug(f"Skipped (unsupported ext '{ext}'): {relative_path}")
                continue

            if ext in code_ext_set:
                code_files.append((full_path, ext))
            else:
                doc_files.append((full_path, ext))

    # Process code files first (higher priority for embedding)
    skipped_oversize = 0
    for file_path, ext in code_files:
        # Forward-slash form so retriever lookup matches LLM-emitted paths.
        relative_path = os.path.relpath(file_path, path).replace(os.sep, '/')

        cap = _size_cap_for(ext, is_code=True)
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        if size > cap:
            skipped_oversize += 1
            logger.info(
                f"Skipped (oversize {size} > {cap}): {relative_path}"
            )
            continue

        try:
            content = safe_read_file(file_path)

            is_implementation = not _is_test_path(relative_path)

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
        relative_path = os.path.relpath(file_path, path).replace(os.sep, '/')

        cap = _size_cap_for(ext, is_code=False)
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        if size > cap:
            skipped_oversize += 1
            logger.info(
                f"Skipped (oversize {size} > {cap}): {relative_path}"
            )
            continue

        try:
            content = safe_read_file(file_path)

            token_count = count_tokens(content, embedder_type)
            if token_count > MAX_EMBEDDING_TOKENS * 10:
                logger.info(
                    f"Large doc file {relative_path}: "
                    f"{token_count} tokens "
                    f"(will be split into ~{token_count // 2000} chunks)"
                )

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

    logger.info(
        f"Found {len(documents)} documents. "
        f"Skipped: {skipped_gitignore} gitignored, "
        f"{skipped_excluded} excluded, "
        f"{skipped_ext} unsupported ext, "
        f"{skipped_oversize} oversize"
    )
    return documents


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


def _compute_file_url(
    relative_path: str,
    repo_url: str,
    repo_type: str,
    branch: str,
) -> str:
    """Compute the source URL for a file in a repository."""
    if not repo_url or not repo_url.startswith(("http://", "https://")):
        return relative_path

    file_path = relative_path.replace("\\", "/")
    if not file_path.startswith("/"):
        file_path = "/" + file_path

    branch_name = branch if branch and branch.strip() else "main"

    if repo_type == "azuredevops":
        return (
            f"{repo_url.rstrip('/')}?version=GB{branch_name}"
            f"&path={file_path}"
        )
    elif repo_type in ["github", "gitlab"]:
        base_url = repo_url
        if base_url.endswith(".git"):
            base_url = base_url[:-4]
        return f"{base_url.rstrip('/')}/blob/{branch_name}{file_path}"
    elif repo_type == "bitbucket":
        base_url = repo_url
        if base_url.endswith(".git"):
            base_url = base_url[:-4]
        return f"{base_url.rstrip('/')}/src/{branch_name}{file_path}"

    return relative_path


def transform_documents_and_save_as_json(
    repo_path: str,
    repo_name: str,
    branch: str,
    repo_url: str = None,
    repo_type: str = None,
    excluded_dirs: List[str] = None,
    excluded_files: List[str] = None,
    included_dirs: List[str] = None,
    included_files: List[str] = None,
    progress_callback: callable = None,
    skip_accumulate: bool = False,
    delta_to_embed: Optional[Set[str]] = None,
    return_chunks_by_source: bool = False,
):
    """
    Reads, splits, embeds and saves documents as JSON chunk files.

    Fully fused pipeline: each file batch is read → split → embedded →
    saved to disk in one pass. No intermediate accumulation of all chunks.
    Peak memory is bounded by FILE_BATCH_SIZE (~120 MB) regardless of
    total repo size, instead of growing with total chunk count.

    Returns both the chunk count and the embedded documents so that
    callers (e.g. FAISS) can use them directly without reloading
    from disk. When skip_accumulate=True (cloud mode), returns an
    empty list to avoid holding all documents in memory.

    Storage structure:
        vectors/{repo_name}_{branch}/
            └── {source_file_path}_chunk_001.json
            ...

    Args:
        repo_path: Local path to cloned repository
        repo_name: Repository name (owner_repo format)
        branch: Branch name
        repo_url: Repository URL (for file link metadata)
        repo_type: Repository type (azuredevops, github, etc.)
        excluded_dirs: Directories to exclude
        excluded_files: File patterns to exclude
        included_dirs: Directories to include exclusively
        included_files: File patterns to include exclusively
        progress_callback: Optional callback(saved, total) for progress
        skip_accumulate: If True, skip collecting documents for return
            (cloud mode — FAISS not needed, saves ~1.5 GB for 100K chunks)

    Returns:
        Tuple of (chunk_count, documents):
            chunk_count: Total number of chunks embedded and saved
            documents: List of Document objects with vectors attached,
                ready for FAISS index construction.
                Empty list when skip_accumulate=True.

    Raises:
        ConnectionError: If Azure Blob Storage is configured but fails
        ValueError: If transformation or saving fails
    """
    import time

    # ================================================================
    # Step 1: Collect file paths (lightweight — no content read)
    # ================================================================

    # Load supported extensions from included.json (single source of truth)
    included = get_included_config()
    code_extensions = set(included["code"])
    doc_extensions = set(included["doc"])
    all_ext_set = code_extensions | doc_extensions

    # Build file filter
    has_dirs = included_dirs is not None and len(included_dirs) > 0
    has_files = included_files is not None and len(included_files) > 0
    if has_dirs or has_files:
        file_filter = FileFilter(
            included_dirs=set(included_dirs) if included_dirs else set(),
            included_patterns=(
                set(included_files) if included_files else set()
            ),
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

    # Load .gitignore from the repo for dynamic filtering
    gitignore_spec = load_gitignore(repo_path)

    # Load excluded_dirs for os.walk() pruning (single source of truth)
    file_filters_cfg = get_file_filters_config()
    walk_excluded_dirs = set(file_filters_cfg["excluded_dirs"])

    # Walk directory once with two-layer filtering
    # Layer 1: Exclusion (.gitignore + excluded.json)
    # Layer 2: Inclusion (included.json supported extensions)
    file_infos = []
    skipped_gitignore = 0
    skipped_excluded = 0
    skipped_ext = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in walk_excluded_dirs]
        for fname in files:
            full_path = os.path.join(root, fname)
            # Forward-slash form so retriever lookup matches LLM-emitted paths.
            relative_path = os.path.relpath(full_path, repo_path).replace(os.sep, '/')

            # Layer 1a: Check .gitignore
            if is_gitignored(gitignore_spec, relative_path):
                skipped_gitignore += 1
                logger.debug(f"Skipped (gitignored): {relative_path}")
                continue

            # Layer 1b: Check excluded file patterns
            if not file_filter.should_process_file(relative_path):
                skipped_excluded += 1
                logger.debug(f"Skipped (excluded): {relative_path}")
                continue

            # Layer 2: Check included extensions
            ext = os.path.splitext(fname)[1].lower()
            if ext not in all_ext_set:
                skipped_ext += 1
                logger.debug(f"Skipped (unsupported ext '{ext}'): {relative_path}")
                continue

            is_code = ext in code_extensions
            file_infos.append(
                (full_path, relative_path, ext, is_code)
            )

    # Sort: code files first (higher priority), then docs
    file_infos.sort(key=lambda fi: (not fi[3], fi[1]))

    # Optional delta filter: keep only files explicitly marked for re-embedding.
    # ``delta_to_embed`` contains forward-slash relative paths; we normalise
    # ``relative_path`` to match before lookup so Windows-style paths line up.
    if delta_to_embed is not None:
        before = len(file_infos)
        file_infos = [
            fi for fi in file_infos
            if fi[1].replace("\\", "/") in delta_to_embed
        ]
        logger.info(
            f"[Vec] Delta filter applied: {before} candidate files -> "
            f"{len(file_infos)} to embed"
        )

    total_files = len(file_infos)
    if total_files == 0:
        logger.warning("[Vec] No files found to process")
        if return_chunks_by_source:
            return 0, [], {}
        return 0, []

    logger.info(
        f"[Vec] Collected {total_files} file paths for "
        f"{repo_name}_{branch}. "
        f"Skipped: {skipped_gitignore} gitignored, "
        f"{skipped_excluded} excluded, "
        f"{skipped_ext} unsupported ext"
    )

    # ================================================================
    # Step 2: Fused read + split + embed + save pipeline
    #
    # Each file batch is read → split → embedded → saved to disk
    # in one pass. No accumulation of all chunks in memory.
    # Peak memory: ~FILE_BATCH_SIZE files worth of chunks (~120MB)
    # instead of ALL chunks across ALL batches (~1.5GB for 100K).
    # ================================================================
    FILE_BATCH_SIZE = configs.get("embedder", {}).get("file_batch_size", 1000)
    EMBED_BATCH_SIZE = configs.get("embedder", {}).get("embed_batch_size", 500)
    embedder = get_embedder()
    vector_storage = get_vector_storage()
    api_batch_size = configs.get("embedder", {}).get("batch_size", 10)

    files_read = 0
    chunks_saved = 0
    total_chunks = 0
    all_embedded_docs: List[Document] = []
    # When return_chunks_by_source=True we record the deterministic chunk
    # filenames generated for each source file. The caller uses this to
    # update the embedding manifest without re-walking the vector store.
    chunks_by_source: Dict[str, List[str]] = {}
    pipeline_start = time.time()
    _read_errors: List[str] = []  # accumulated per-file read errors
    _embed_warnings: List[str] = []  # accumulated embed batch warnings

    total_file_batches = (
        (total_files + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE
    )

    logger.info(
        f"[Vec] Starting fused pipeline: {total_files} files "
        f"in {total_file_batches} file batches "
        f"(file_batch={FILE_BATCH_SIZE}, embed_batch={EMBED_BATCH_SIZE}, "
        f"api_batch={api_batch_size})"
    )

    for fb_idx in range(total_file_batches):
        fb_start = fb_idx * FILE_BATCH_SIZE
        fb_end = min(fb_start + FILE_BATCH_SIZE, total_files)
        batch_infos = file_infos[fb_start:fb_end]

        # --- Read this batch of files into Documents ---
        batch_docs = []
        for full_path, relative_path, ext, is_code in batch_infos:
            cap = _size_cap_for(ext, is_code=is_code)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                size = 0
            if size > cap:
                logger.info(
                    f"Skipped (oversize {size} > {cap}): {relative_path}"
                )
                continue

            try:
                content = safe_read_file(full_path)

                token_count = count_tokens(content)
                if token_count > MAX_EMBEDDING_TOKENS * 10:
                    logger.info(
                        f"Large file {relative_path}: "
                        f"{token_count} tokens "
                        f"(will be split into ~{token_count // 2000} chunks)"
                    )

                is_implementation = is_code and not _is_test_path(relative_path)

                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": ext[1:],
                        "is_code": is_code,
                        "is_implementation": is_implementation,
                        "url": _compute_file_url(
                            relative_path, repo_url, repo_type, branch
                        ),
                    },
                )
                batch_docs.append(doc)
            except Exception as e:
                _read_errors.append(f"{full_path}: {e}")

        # --- Split into enriched chunks ---
        batch_chunks = []
        if batch_docs:
            batch_chunks = split_and_enrich_documents(batch_docs)
        del batch_docs
        gc.collect()

        files_read += len(batch_infos)
        total_chunks += len(batch_chunks)

        if not batch_chunks:
            if total_file_batches > 1:
                logger.info(
                    f"[Vec] File batch {fb_idx + 1}/"
                    f"{total_file_batches}: {files_read}/{total_files} "
                    f"files -> 0 chunks (skipped)"
                )
            continue

        # --- Embed + save this batch's chunks immediately ---
        batch_embed_batches = (
            (len(batch_chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        )
        for eb_idx in range(batch_embed_batches):
            eb_start = eb_idx * EMBED_BATCH_SIZE
            eb_end = min(eb_start + EMBED_BATCH_SIZE, len(batch_chunks))
            embed_batch = batch_chunks[eb_start:eb_end]

            # Embed via direct API calls in sub-batches
            for sub_start in range(0, len(embed_batch), api_batch_size):
                sub_end = min(
                    sub_start + api_batch_size, len(embed_batch)
                )
                sub_batch = embed_batch[sub_start:sub_end]
                texts = [doc.text for doc in sub_batch]
                result = embedder(texts)
                if hasattr(result, 'data') and result.data:
                    for i, emb in enumerate(result.data):
                        if i < len(sub_batch):
                            sub_batch[i].vector = (
                                emb.embedding
                                if hasattr(emb, 'embedding')
                                else emb
                            )

            batch_transformed = [
                doc for doc in embed_batch
                if hasattr(doc, 'vector') and doc.vector is not None
            ]

            if not batch_transformed:
                _embed_warnings.append(
                    f"file_batch={fb_idx + 1}, embed_batch={eb_idx + 1}"
                )
                continue

            # Capture chunks_saved in a default argument to avoid
            # late-binding closure issues with the lambda
            if not vector_storage.save_documents(
                batch_transformed,
                repo_name,
                branch,
                progress_callback=(
                    lambda saved, total, _base=chunks_saved: (
                        progress_callback(_base + saved, total_chunks)
                    )
                ) if progress_callback else None
            ):
                raise ValueError(
                    f"[Vec] Failed to save file batch {fb_idx + 1} "
                    f"embed batch {eb_idx + 1} "
                    f"for {repo_name}_{branch}"
                )

            if not skip_accumulate:
                all_embedded_docs.extend(batch_transformed)
            chunks_saved += len(batch_transformed)

            # Track chunk filenames per source for the manifest. We mirror
            # the grouping that save_documents uses so the numbers match the
            # files that were just written to disk.
            if return_chunks_by_source:
                by_src_in_batch: Dict[str, List[Document]] = {}
                for doc in batch_transformed:
                    src = (
                        doc.meta_data.get("file_path", "unknown")
                        if doc.meta_data else "unknown"
                    )
                    by_src_in_batch.setdefault(src, []).append(doc)
                for src, docs in by_src_in_batch.items():
                    start_idx = len(chunks_by_source.get(src, []))
                    filenames = chunks_by_source.setdefault(src, [])
                    for offset in range(len(docs)):
                        filenames.append(
                            vector_storage._get_json_filename(
                                src, start_idx + offset
                            )
                        )

            del batch_transformed, embed_batch
            gc.collect()

        del batch_chunks
        gc.collect()

        if total_file_batches > 1:
            logger.debug(
                f"[Vec] File batch {fb_idx + 1}/"
                f"{total_file_batches}: {files_read}/{total_files} "
                f"files, {chunks_saved}/{total_chunks} chunks saved"
            )

    # Release file_infos
    del file_infos
    gc.collect()

    # Log accumulated errors/warnings from the pipeline
    if _read_errors:
        logger.error(
            f"[Vec] {len(_read_errors)} file read errors "
            f"(first: {_read_errors[0]})"
        )
    if _embed_warnings:
        logger.warning(
            f"[Vec] {len(_embed_warnings)} embed batches produced "
            f"no documents after embedding"
        )

    pipeline_elapsed = time.time() - pipeline_start
    logger.info(
        f"[Vec] Fused pipeline complete in {pipeline_elapsed:.1f}s: "
        f"{files_read} files -> {chunks_saved} chunks saved "
        f"({chunks_saved / max(pipeline_elapsed, 0.1):.0f} chunks/sec)"
    )

    if chunks_saved == 0:
        logger.warning("[Vec] No documents after embedding")
        if return_chunks_by_source:
            return 0, [], {}
        return 0, []

    logger.info(
        f"[Vec] Successfully embedded and saved {chunks_saved} "
        f"enriched chunks as JSON"
    )
    if return_chunks_by_source:
        return chunks_saved, all_embedded_docs, chunks_by_source
    return chunks_saved, all_embedded_docs
