"""
DeepWiki Code Processor - standalone CLI for wiki generation.

Usage:
    python -m backend.processor.code_processor --repo=URL --branch=main --mode=local
    python -m backend.processor.code_processor --config=run.json

Modes:
    local   - Run on developer machine (PAT or az login, FAISS, local disk)
    docker  - Build & run in container  (PAT + API key only, FAISS, local disk)
    cloud   - Run inside AML only       (UMI, blob, AI Search)

    For cloud setup, use: python -m backend.processor.aml_dispatcher
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ============================================================================
# Config directory for each mode (relative to backend/)
# ============================================================================
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_CONFIG_DEFAULT = str(_BACKEND_DIR / 'config')
_CONFIG_CLOUD = str(_BACKEND_DIR / 'config' / '.cloud')
_CONFIG_DOCKER = str(_BACKEND_DIR / 'config' / '.local')


# ============================================================================
# CLI Parsing
# ============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments with --config file support."""
    parser = argparse.ArgumentParser(
        description='DeepWiki Code Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --repo="https://dev.azure.com/org/proj/_git/repo" --branch=main --mode=local
  %(prog)s --config=run.json
        """,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (e.g., run.json)')
    parser.add_argument('--repo', type=str, default=None,
                        help='Azure DevOps repo URL')
    parser.add_argument('--branch', type=str, default=None,
                        help='Branch name to process')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['local', 'docker', 'cloud'],
                        help='Execution mode: local, docker, or cloud')
    parser.add_argument('--comprehensive', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--language', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--skip-codemap', action='store_true',
                        help='Skip codemap graph generation')
    parser.add_argument('--full-reprocess', action='store_true',
                        help=(
                            'Ignore the embedding manifest and re-embed '
                            'every file (full rebuild). Default behaviour '
                            'is incremental: only changed files are '
                            're-embedded.'
                        ))

    args = parser.parse_args()

    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            parser.error(f"Config file not found: {args.config}")
        with open(config_path, 'r') as f:
            config = json.load(f)

    final = {
        'repo': args.repo or config.get('repo'),
        'branch': args.branch or config.get('branch'),
        'mode': args.mode or config.get('mode'),
        'comprehensive': True,
        'language': args.language or config.get('language', 'en'),
        'skip_codemap': args.skip_codemap or config.get('skip_codemap', False),
        'full_reprocess': args.full_reprocess or config.get(
            'full_reprocess', False
        ),
    }

    if args.comprehensive is not None:
        final['comprehensive'] = args.comprehensive.lower() in ('true', '1', 'yes')

    if not final['repo']:
        parser.error("--repo is required (or set 'repo' in config file)")
    if not final['branch']:
        parser.error("--branch is required (or set 'branch' in config file)")
    if not final['mode']:
        parser.error("--mode is required (or set 'mode' in config file)")

    return argparse.Namespace(**final)


# ============================================================================
# Reusable helpers
# ============================================================================

def _extract_owner_repo(repo_url: str) -> tuple:
    """Extract owner (organization) and repo name from Azure DevOps URL.

    Examples:
        https://dev.azure.com/org/proj/_git/repo -> ('org', 'repo')
        https://msdata.visualstudio.com/Database%20Systems/_git/orcasql-myfile
            -> ('msdata', 'orcasql-myfile')
    """
    from urllib.parse import unquote, urlparse
    url = unquote(repo_url.rstrip('/'))
    parsed = urlparse(url)
    host = parsed.hostname or ''
    path_parts = [p for p in parsed.path.split('/') if p]

    if '/_git/' in url and path_parts:
        git_idx = path_parts.index('_git') if '_git' in path_parts else -1
        repo = path_parts[git_idx + 1] if git_idx >= 0 and git_idx + 1 < len(path_parts) else 'unknown'

        if 'visualstudio.com' in host:
            owner = host.split('.')[0]
        elif 'dev.azure.com' in host:
            owner = path_parts[0] if path_parts else 'unknown'
        else:
            owner = path_parts[git_idx - 1] if git_idx >= 1 else 'unknown'
        return owner, repo

    if len(path_parts) >= 2:
        return path_parts[-2], path_parts[-1]
    return 'unknown', path_parts[-1] if path_parts else 'unknown'


# ============================================================================
# Step functions - each is a discrete, reusable pipeline step
# ============================================================================

def resolve_auth(mode: str) -> tuple:
    """Resolve an access token for the repository.

    Auth strategy per mode:
        local  - PAT from env → Git Credential Manager (no token needed)
        docker - PAT from env only → error if missing
        cloud  - PAT from env → UMI from infra.json managed_identity.client_id

    Returns:
        Tuple of (token, token_type) where:
        - token_type is 'pat', 'bearer', or 'gcm'
        - When token_type='gcm', token is None (GCM handles auth natively)
    """
    pat = (os.environ.get('REPO_ACCESS_TOKEN', '')
           or os.environ.get('ADO_PAT', '')
           or os.environ.get('AZURE_DEVOPS_PAT', ''))

    if pat:
        masked = pat[:6] + '***' if len(pat) > 6 else '***'
        logger.info(f"Auth: PAT from environment ({masked})")
        return pat, 'pat'

    if mode == 'docker':
        logger.error("Docker mode requires REPO_ACCESS_TOKEN in .env")
        sys.exit(1)

    if mode == 'cloud':
        # Cloud mode (AML compute): must use managed identity
        try:
            from azure.identity import DefaultAzureCredential
            from backend.config import get_managed_identity_client_id
            msi_client_id = get_managed_identity_client_id()
            if not msi_client_id:
                logger.error("managed_identity.client_id not set in infra.json")
                sys.exit(1)
            logger.info(f"Auth: managed identity ({msi_client_id[:8]}...)")
            credential = DefaultAzureCredential(
                managed_identity_client_id=msi_client_id,
            )
            token = credential.get_token(
                "499b84ac-1321-427f-aa17-267ca6975798/.default"
            )
            logger.info("Auth: token acquired via managed identity")
            return token.token, 'bearer'
        except Exception as e:
            logger.error(f"Could not acquire managed identity token: {e}")
            logger.error("Check managed identity in infra.json")
            sys.exit(1)

    # Local mode: let Git Credential Manager handle authentication.
    # GCM supports AAD/Azure CLI/browser SSO for Azure DevOps natively,
    # which is more reliable than injecting bearer tokens via extraHeader.
    logger.info(
        "Auth: delegating to Git Credential Manager "
        "(no PAT found, local mode)"
    )
    return None, 'gcm'


def _resolve_umi_auth() -> tuple:
    """Acquire a bearer token via User-assigned Managed Identity (UMI).

    Used as a fallback when a PAT from the environment fails in cloud mode.

    Returns:
        Tuple of (token, 'bearer')

    Raises:
        RuntimeError: If UMI token acquisition fails.
    """
    from azure.identity import DefaultAzureCredential
    from backend.config import get_managed_identity_client_id

    msi_client_id = get_managed_identity_client_id()
    if not msi_client_id:
        raise RuntimeError(
            "managed_identity.client_id not set in infra.json"
        )
    logger.info(f"Auth: falling back to managed identity ({msi_client_id[:8]}...)")
    credential = DefaultAzureCredential(
        managed_identity_client_id=msi_client_id,
    )
    token = credential.get_token(
        "499b84ac-1321-427f-aa17-267ca6975798/.default"
    )
    logger.info("Auth: token acquired via managed identity (fallback)")
    return token.token, 'bearer'


def step_clone(repo_url, token, branch, repo_name, token_type='pat'):
    """Clone repository to local disk. Returns (repo_dir, commit_hash)."""
    from backend.modules.repository.git_ops import (
        download_repo, get_head_commit_hash,
    )
    from backend.paths import get_repos_path
    from backend.utils.filter import sanitize_branch_for_path

    logger.info("Step 1: Cloning repository")
    branch_safe = sanitize_branch_for_path(branch or 'main')
    save_dir = os.path.join(get_repos_path(), f"{repo_name}_{branch_safe}")
    download_repo(
        repo_url=repo_url,
        local_path=save_dir,
        type='azuredevops',
        access_token=token,
        branch=branch,
        force_update=True,
        token_type=token_type,
    )
    commit = get_head_commit_hash(save_dir)
    logger.info(f"Cloned to: {save_dir}, commit: {commit[:7] if commit else 'unknown'}")
    return save_dir, commit


def _log_rss(label: str, level: int = logging.INFO):
    """Log current process RSS for memory debugging.

    Default level is INFO so that breadcrumbs survive an OOM crash and
    appear in AML compute logs without raising the global log level.
    Pass `level=logging.DEBUG` for very-frequent call sites.
    """
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.log(level, f"[MEM] {label}: {rss_mb:.0f} MB RSS")
    except ImportError:
        pass


def step_build_codemap(repo_path, owner, repo, repo_type, branch):
    """Build codemap graph from cloned repository."""
    from backend.modules.codemap.graph_builder import build_codemap
    from backend.modules.codemap.cache import save_codemap_cache
    from backend.config import get_file_filters_config
    from backend.types import FileFilter

    logger.info("Step: Building codemap graph")

    # Reuse the same file filter as the embedder (excluded.json)
    file_filters = get_file_filters_config()
    file_filter = FileFilter(
        excluded_dirs=set(file_filters["excluded_dirs"]),
        excluded_patterns=set(file_filters["excluded_files"]),
    )

    codemap = build_codemap(repo_path, file_filter=file_filter)
    codemap.metadata.owner = owner
    codemap.metadata.repo = repo
    codemap.metadata.repo_type = repo_type
    codemap.metadata.branch = branch

    save_codemap_cache(codemap, owner, repo, repo_type, branch)
    logger.info(
        f"CodeMap: {codemap.metadata.total_files} files, "
        f"{codemap.metadata.total_symbols} symbols, "
        f"{codemap.metadata.total_edges} edges"
    )
    return codemap


def step_embed(repo_url, token, branch, repo_dir=None, force_reprocess=False):
    """Embed documents and build retriever. Returns RAG instance.

    Storage backend (local or blob) and AOAI auth (MSI or API key)
    are driven by config - this step does not need to know.

    ``force_reprocess`` opts out of the manifest-driven delta and forces a
    full re-embed of every file. Defaults to False, so reruns only embed
    files that actually changed.
    """
    from backend.modules.embedder.retriever import RAG

    logger.info("Step 2: Embedding documents")
    rag = RAG(provider='azure')
    rag.prepare_retriever(
        repo_url_or_path=repo_url,
        type='azuredevops',
        access_token=token,
        branch=branch,
        force_reprocess=force_reprocess,
        repo_dir=repo_dir,
    )
    logger.info(f"Retriever ready ({len(rag.transformed_docs)} docs)")

    # Release components unused during processor wiki generation:
    # - generator: wiki_generator.py uses its own _call_llm()
    # - memory: no conversation history during batch processing
    # - db_manager: database already prepared, not needed further
    rag.generator = None
    rag.memory = None
    rag.db_manager = None

    return rag


def step_embed_cloud(repo_url, token, branch, repo_dir=None, force_reprocess=False):
    """Embed documents and save to blob (cloud mode, no FAISS).

    Skips FAISS construction and document accumulation since wiki
    generation will use Azure AI Search for retrieval instead. Returns
    ``(chunk_count, delta_changed_sources)`` where ``delta_changed_sources``
    is the set of source paths whose chunks were rewritten in this run —
    Step 5 (push_to_search) uses it to delete the corresponding AI Search
    docs before re-pushing.
    """
    from backend.modules.embedder.indexer import (
        DatabaseManager,
        _walk_candidate_files,
        _current_embedder_signature,
        _build_manifest,
    )
    from backend.modules.embedder.delta import (
        compute_file_delta, summarize_for_log,
    )
    from backend.modules.embedder.document import (
        transform_documents_and_save_as_json,
    )
    from backend.modules.repository.git_ops import get_head_commit_hash
    from backend.clients.vector_storage import get_vector_storage

    logger.info("Step 2: Embedding documents (cloud mode)")
    db_manager = DatabaseManager()
    db_manager._create_repo(
        repo_url, 'azuredevops', token, branch,
        force_reprocess=force_reprocess,
        repo_dir=repo_dir,
    )

    repo_name = db_manager.repo_paths["repo_name"]
    branch_suffix = db_manager.repo_paths["branch_suffix"]
    save_repo_dir = db_manager.repo_paths["save_repo_dir"]
    vector_storage = get_vector_storage()

    # ----- Delta detection -----
    prev_manifest = (
        vector_storage.read_manifest(repo_name, branch_suffix)
        if not force_reprocess else None
    )
    current_commit_hash = ""
    if save_repo_dir and os.path.isdir(os.path.join(save_repo_dir, ".git")):
        try:
            current_commit_hash = get_head_commit_hash(save_repo_dir)
        except Exception as e:
            logger.warning(f"[Delta] Could not read HEAD commit: {e}")

    candidate_file_infos = _walk_candidate_files(save_repo_dir)
    embedder_sig = _current_embedder_signature()
    delta = compute_file_delta(
        repo_dir=save_repo_dir,
        file_infos=candidate_file_infos,
        prev_manifest=prev_manifest,
        current_embedder=embedder_sig,
        current_commit_hash=current_commit_hash,
        verify_hash=True,
    )
    logger.info(f"[Vec] Delta summary (cloud): {summarize_for_log(delta)}")

    # Fast-path no-op exit.
    if (
        prev_manifest
        and not delta.to_embed
        and not delta.to_delete
        and vector_storage.exists(repo_name, branch_suffix)
    ):
        logger.info("[Vec] No changes detected — skipping cloud embed")
        return 0, set()

    # ----- Drop chunks for deleted/replaced sources -----
    if delta.to_delete and prev_manifest:
        removed = vector_storage.delete_files_for_sources(
            repo_name, branch_suffix, delta.to_delete, prev_manifest,
        )
        logger.info(
            f"[Vec] Removed {removed} cloud chunk files for "
            f"{len(delta.to_delete)} deleted/changed sources"
        )

    # ----- Embed only the to_embed slice (skip FAISS accumulation) -----
    delta_paths = {
        fi[1].replace("\\", "/") for fi in delta.to_embed
    } if not force_reprocess else None

    chunk_count, _, chunks_by_source_new = transform_documents_and_save_as_json(
        save_repo_dir,
        repo_name,
        branch_suffix,
        repo_url=repo_url,
        repo_type='azuredevops',
        skip_accumulate=True,
        delta_to_embed=delta_paths,
        return_chunks_by_source=True,
    )

    # ----- Write the new manifest -----
    try:
        new_manifest = _build_manifest(
            prev_manifest=prev_manifest,
            delta=delta,
            chunks_by_source_new=chunks_by_source_new,
            current_commit_hash=current_commit_hash,
            embedder_sig=embedder_sig,
        )
        vector_storage.write_manifest(repo_name, branch_suffix, new_manifest)
    except Exception as e:
        logger.warning(f"[Vec] Failed to write manifest (non-fatal): {e}")

    # ``changed_sources`` = sources whose chunks now differ in blob storage,
    # i.e. anything we just re-embedded plus anything we deleted. The cloud
    # push step uses this to keep AI Search in sync.
    #
    # Cold-start optimisation: when there was no prior manifest every file
    # was embedded fresh, so the "diff" is "everything". Return ``None`` to
    # signal the push step to use its streaming full-push path instead of
    # iterating per-file — same correctness, far fewer filter queries.
    if prev_manifest is None:
        changed_sources = None
    else:
        changed_sources = set(
            fi[1].replace("\\", "/") for fi in delta.to_embed
        ) | set(delta.to_delete)

    logger.info(
        f"Embedded {chunk_count} chunks (saved to blob, no FAISS); "
        f"changed sources: "
        f"{'ALL (cold start)' if changed_sources is None else len(changed_sources)}"
    )
    return chunk_count, changed_sources


def step_generate_wiki(
    repo_url, branch, repo_path, retriever,
    commit_hash, language, comprehensive, owner, repo,
    codemap=None,
):
    """Generate wiki pages. Returns wiki_data."""
    from backend.processor.wiki_generator import generate_wiki

    logger.info("Step 3: Generating wiki")
    return generate_wiki(
        repo_url=repo_url,
        branch=branch,
        repo_type='azuredevops',
        repo_path=repo_path,
        retriever=retriever,
        commit_hash=commit_hash,
        language=language,
        comprehensive=comprehensive,
        owner=owner,
        repo=repo,
        codemap=codemap,
    )


def step_save_wiki(wiki_data, language, comprehensive):
    """Save wiki cache (storage backend determined by config)."""
    import asyncio
    from backend.modules.wiki.cache import save_wiki_cache
    from backend.modules.wiki.models import WikiCacheRequest

    logger.info("Step 4: Saving wiki cache")
    cache_request = WikiCacheRequest(
        repo=wiki_data.repo,
        language=language,
        comprehensive=comprehensive,
        wiki_structure=wiki_data.wiki_structure,
        generated_pages=wiki_data.generated_pages,
        provider=wiki_data.provider or 'azure',
        model=wiki_data.model or '',
        is_partial=False,
        commit_hash=wiki_data.commit_hash,
        indexed_at=wiki_data.indexed_at,
    )
    try:
        result = asyncio.run(save_wiki_cache(cache_request))
        if result:
            logger.info("Wiki cache saved")
        else:
            logger.warning("save_wiki_cache returned False")
    except Exception as e:
        logger.error(f"Failed to save wiki cache: {e}", exc_info=True)


def step_push_to_search(
    owner, repo, branch, wait=False,
    changed_sources=None, full_reprocess=False,
):
    """Push vectors to AI Search and trigger indexer (cloud only).

    Loads and pushes in batches of PUSH_BATCH_SIZE to avoid holding
    all vectors in memory at once (~725MB for 50K chunks).

    The selective-push contract:
      * ``full_reprocess=True`` OR ``changed_sources is None`` → push every
        chunk currently in blob storage (legacy behaviour, used on cold
        start / explicit full rebuild).
      * ``changed_sources`` is a (possibly empty) set of source-file paths
        → delete the existing AI Search rows for those files (if any) then
        push only the chunks belonging to those files. Files not in the
        set are assumed to already be in the index with current content.

    Args:
        owner: Repo owner
        repo: Repo name
        branch: Branch name
        wait: If True, block until indexer completes (for cloud-first flow)
        changed_sources: Set of relative source paths whose chunks were
            re-embedded or deleted in this run. ``None`` means "unknown,
            push everything".
        full_reprocess: When True, forces the full push path regardless of
            ``changed_sources``.
    """
    from backend.clients.search_client import (
        get_index_name, push_documents, run_indexer,
        index_exists, wait_for_indexer, delete_documents_for_sources,
    )
    from backend.clients.vector_storage import get_vector_storage

    idx_name = get_index_name(owner, repo, branch or 'main')
    if not index_exists(idx_name):
        logger.info(f"AI Search index '{idx_name}' not found, skipping push")
        return

    logger.info("Step 5: Pushing vectors to AI Search")
    repo_name = f"{owner}_{repo}"
    vector_storage = get_vector_storage()

    selective = (
        not full_reprocess
        and changed_sources is not None
    )

    if selective and not changed_sources:
        logger.info(
            "[Search] No source changes detected — skipping push entirely"
        )
        return

    # ----- Selective path: delete-by-source then push only those files -----
    if selective:
        # Normalise to forward slashes (matches what we stored in filepath).
        norm_sources = {
            s.replace("\\", "/") for s in changed_sources if s
        }
        # Drop AI Search rows for changed/deleted files first so old chunks
        # don't linger for any file that shrank between runs.
        try:
            delete_documents_for_sources(
                idx_name, repo_name, branch or 'main', norm_sources,
            )
        except Exception as e:
            logger.warning(
                f"[Search] delete_documents_for_sources failed "
                f"(continuing with push): {e}"
            )

        # iter_documents_for_sources only reads chunks for changed files —
        # cheap on blob storage because we know the exact file list. It needs
        # the manifest to resolve each source path to its chunk-file blobs.
        # (read_manifest sanitises `branch` internally, matching the path the
        # embed step wrote, so the raw branch value is safe to pass here.)
        manifest = vector_storage.read_manifest(repo_name, branch) or {}
        pushed = 0
        _log_rss("before push_to_search load")
        for batch in vector_storage.iter_documents_for_sources(
            repo_name, branch, norm_sources, manifest,
        ):
            if not batch:
                continue
            push_documents(idx_name, batch, repo_name, branch)
            pushed += len(batch)
            for d in batch:
                d.vector = None
        _log_rss("after push_to_search load")

        if pushed == 0:
            logger.info(
                "[Search] Selective push found 0 chunks for "
                f"{len(norm_sources)} sources (deletes only)"
            )
        else:
            logger.info(
                f"[Search] Selective push: {pushed} chunks across "
                f"{len(norm_sources)} sources"
            )

        import gc
        gc.collect()
        run_indexer(idx_name)
        logger.info("Indexer triggered")
        if wait:
            logger.info("Waiting for indexer to complete...")
            success = wait_for_indexer(idx_name, timeout_seconds=600)
            if not success:
                logger.error(
                    "Indexer timeout — wiki generation may have "
                    "incomplete results"
                )
            else:
                logger.info("Indexer completed successfully")
        return

    # ----- Full-push path (legacy / cold start / --full-reprocess) -----
    # Stream chunks in batches -- never materialise the full vector set in
    # memory. Each batch is converted to upload payload, pushed, and freed
    # before the next batch is fetched. Bounds peak RSS at
    # O(PUSH_BATCH_SIZE x vector_size) regardless of total chunk count.
    PUSH_BATCH_SIZE = 1000
    pushed = 0
    _log_rss("before push_to_search load")
    for batch in vector_storage.iter_documents(
        repo_name, branch, batch_size=PUSH_BATCH_SIZE,
    ):
        if not batch:
            continue
        # Deterministic per-file keys (see `_build_doc_key`) make this
        # idempotent — re-pushing the same chunk overwrites its row instead
        # of duplicating it.
        push_documents(idx_name, batch, repo_name, branch)
        pushed += len(batch)
        # Release vectors before fetching the next batch.
        for doc in batch:
            doc.vector = None
        del batch
    _log_rss("after push_to_search load")

    if pushed == 0:
        logger.info("No vector documents found to push")
        return

    import gc
    gc.collect()

    logger.info(f"Pushed {pushed} documents to AI Search")
    run_indexer(idx_name)
    logger.info("Indexer triggered")

    if wait:
        logger.info("Waiting for indexer to complete...")
        success = wait_for_indexer(idx_name, timeout_seconds=600)
        if not success:
            logger.error(
                "Indexer timeout — wiki generation may have "
                "incomplete results"
            )
        else:
            logger.info("Indexer completed successfully")


# ============================================================================
# Processing pipeline (all modes)
# ============================================================================

def step_generate_wiki_cloud(
    repo_url, branch, repo_path, owner, repo,
    commit_hash, language, comprehensive,
    codemap=None,
):
    """Generate wiki using Azure AI Search for retrieval (cloud mode).

    Creates a lightweight RAG instance with cloud search enabled.
    No FAISS, no in-memory document loading (~50 MB steady-state).
    """
    from backend.modules.embedder.retriever import RAG
    from backend.clients.search_client import get_index_name
    from backend.processor.wiki_generator import generate_wiki

    logger.info("Step 3: Generating wiki (cloud retrieval)")

    # Create lightweight RAG for cloud retrieval only
    rag = RAG(provider='azure')
    idx_name = get_index_name(owner, repo, branch or 'main')
    rag.prepare_for_cloud(idx_name)

    # Release unused components
    rag.generator = None
    rag.memory = None
    rag.db_manager = None

    return generate_wiki(
        repo_url=repo_url,
        branch=branch,
        repo_type='azuredevops',
        repo_path=repo_path,
        retriever=rag,
        commit_hash=commit_hash,
        language=language,
        comprehensive=comprehensive,
        owner=owner,
        repo=repo,
        codemap=codemap,
    )


def _process(mode, repo_url, branch, language, comprehensive,
             skip_codemap=False, full_reprocess=False):
    """Run the processing pipeline using step functions.

    The mode determines auth; config (already set before this call)
    determines storage backend (local/blob) and retrieval (FAISS/AI Search).

    Cloud mode uses a reordered pipeline:
        embed → push to AI Search → wait for indexer → generate via Search
    This avoids building FAISS (~1.5 GB) during wiki generation.

    Local/Docker mode uses the original pipeline:
        embed + FAISS → generate via FAISS → push to Search (if configured)

    Args:
        full_reprocess: When True, ignores the embedding manifest and
            re-embeds every file. Defaults to False, so reruns only embed
            files that changed since the last manifest.
    """
    # Lazy import: keeps backend.config out of code_processor's
    # import-time dependency surface.
    from backend.utils.guard_session import GuardSession

    owner, repo = _extract_owner_repo(repo_url)
    repo_name = f"{owner}_{repo}"

    logger.info(
        f"Processing: repo={repo_url}, branch={branch}, "
        f"mode={mode}, language={language}, owner={owner}, repo={repo}"
    )

    # Snapshot/relax/restore content-filter policies around the
    # pipeline body. __exit__ runs unconditionally — clean exit,
    # exception, KeyboardInterrupt — restoring the policy to its
    # pre-run state. Auto-disabled in Docker mode and when
    # DEEPWIKI_GUARD_CHECKER_DISABLED is set.
    with GuardSession.from_infra(mode=mode):
        token, token_type = resolve_auth(mode)

        # Clone to local temp disk (all modes — even cloud clones
        # locally on AML compute; vectors and wiki go to blob via
        # storage abstraction)
        try:
            repo_dir, commit_hash = step_clone(
                repo_url, token, branch, repo_name, token_type,
            )
        except ValueError as e:
            if mode == 'cloud' and token_type == 'pat':
                logger.warning(
                    "PAT clone failed in cloud mode, "
                    "falling back to managed identity"
                )
                token, token_type = _resolve_umi_auth()
                repo_dir, commit_hash = step_clone(
                    repo_url, token, branch, repo_name, token_type,
                )
            else:
                logger.error(f"Clone failed: {e}")
                raise

        _log_rss("after clone")

        # ----- Fast-path: skip everything if wiki cache matches commit -----
        # When the same commit was already processed end-to-end (wiki cache
        # holds the matching ``commit_hash``), there is nothing to do. Skip
        # codemap + embed + push + wiki gen + save entirely. ``--full-reprocess``
        # opts out of this shortcut and forces a full rebuild.
        if not full_reprocess:
            try:
                from backend.modules.wiki.cache import (
                    wiki_cache_exists_for_commit,
                )
                if commit_hash and wiki_cache_exists_for_commit(
                    owner=owner,
                    repo=repo,
                    repo_type='azuredevops',
                    language=language,
                    commit_hash=commit_hash,
                    comprehensive=comprehensive,
                    branch=branch,
                ):
                    logger.info(
                        f"[FastPath] Wiki cache already exists for commit "
                        f"{commit_hash[:7]} (lang={language}, "
                        f"comprehensive={comprehensive}) — "
                        f"skipping embed / push / wiki regeneration. "
                        f"Pass --full-reprocess to force a rebuild."
                    )
                    return None
            except Exception as e:
                # Fast-path is purely an optimisation; any error → fall
                # through to the normal pipeline.
                logger.debug(f"Wiki fast-path probe skipped: {e}")

        # Build codemap graph (all modes, unless skipped)
        codemap = None
        if not skip_codemap:
            try:
                codemap = step_build_codemap(
                    repo_dir, owner, repo, 'azuredevops', branch,
                )
            except Exception as e:
                logger.warning(f"Codemap generation failed (non-fatal): {e}")
            finally:
                gc.collect()

        _log_rss("after codemap")

        if mode == 'cloud':
            # ============================================================
            # CLOUD MODE: embed → push to search → generate via search
            # No FAISS, no in-memory document loading (~120 MB peak)
            # ============================================================
            chunk_count, changed_sources = step_embed_cloud(
                repo_url, token, branch, repo_dir=repo_dir,
                force_reprocess=full_reprocess,
            )
            _log_rss("after embed_cloud")

            step_push_to_search(
                owner, repo, branch, wait=True,
                changed_sources=changed_sources,
                full_reprocess=full_reprocess,
            )
            _log_rss("after push_to_search")

            try:
                wiki_data = step_generate_wiki_cloud(
                    repo_url, branch, repo_dir, owner, repo,
                    commit_hash, language, comprehensive,
                    codemap=codemap,
                )
            except Exception as e:
                logger.error(
                    f"Wiki generation failed: {e}", exc_info=True
                )
                sys.exit(1)

            # Codemap is no longer needed after wiki generation;
            # release it before save_wiki so the JSON-write phase
            # runs at minimum RSS. (generate_wiki internally already
            # drops its reference; this is the outer-scope drop.)
            codemap = None
            gc.collect()
            _log_rss("after generate_wiki (codemap freed)")
            step_save_wiki(wiki_data, language, comprehensive)
        else:
            # ============================================================
            # LOCAL / DOCKER MODE: embed + FAISS → generate via FAISS
            # (unchanged from existing implementation)
            # ============================================================
            retriever = step_embed(
                repo_url, token, branch, repo_dir=repo_dir,
                force_reprocess=full_reprocess,
            )

            try:
                wiki_data = step_generate_wiki(
                    repo_url, branch, repo_dir, retriever,
                    commit_hash, language, comprehensive, owner, repo,
                    codemap=codemap,
                )
            except Exception as e:
                logger.error(
                    f"Wiki generation failed: {e}", exc_info=True
                )
                sys.exit(1)

            # Free FAISS index + transformed_docs + codemap before save
            del retriever
            codemap = None
            gc.collect()
            _log_rss("after generate_wiki (FAISS + codemap freed)")

            step_save_wiki(wiki_data, language, comprehensive)

        logger.info(
            f"Processing complete: {len(wiki_data.generated_pages)} pages, "
            f"commit={commit_hash[:7] if commit_hash else 'N/A'}"
        )

        return wiki_data


# ============================================================================
# Docker build & launch (runs on user's machine)
# ============================================================================

def _run_docker_build(args):
    """Build Docker image and run processor inside container."""
    import subprocess
    from backend.processor.cloud_setup import write_docker_config

    project_root = Path(__file__).resolve().parents[2]
    dockerfile = project_root / 'Dockerfile.processor'
    env_file = project_root / 'backend' / '.env'
    image_name = 'deepwiki-processor'

    if not dockerfile.is_file():
        logger.error(f"Dockerfile.processor not found at {dockerfile}")
        sys.exit(1)

    write_docker_config()

    logger.info("Building Docker image")
    build_cmd = [
        'docker', 'build',
        '-f', str(dockerfile),
        '-t', image_name,
        str(project_root),
    ]
    logger.info(f"$ {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, cwd=str(project_root))
    if result.returncode != 0:
        logger.error("Docker build failed")
        sys.exit(1)
    logger.info("Image built")

    logger.info("Running processor in container")
    adalflow_dir = Path.home() / '.adalflow'
    adalflow_dir.mkdir(parents=True, exist_ok=True)

    run_cmd = [
        'docker', 'run', '--rm',
        '-v', f'{adalflow_dir}:/root/.adalflow',
        '-e', '_DEEPWIKI_INSIDE_DOCKER=1',
    ]
    if env_file.is_file():
        run_cmd.extend(['--env-file', str(env_file)])

    for key in ('AZURE_OPENAI_API_KEY', 'AZURE_CLIENT_ID', 'REPO_ACCESS_TOKEN'):
        val = os.environ.get(key)
        if val:
            run_cmd.extend(['-e', f'{key}={val}'])

    run_cmd.extend([
        image_name,
        '--repo', args.repo,
        '--branch', args.branch,
        '--mode', 'docker',
        '--language', args.language,
    ])

    logger.info(f"$ docker run ... {image_name} --repo=... --branch={args.branch}")
    result = subprocess.run(run_cmd)
    if result.returncode != 0:
        logger.error("Docker run failed")
        sys.exit(1)
    logger.info("Docker processing complete")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """CLI entry point - mode acts as a switch for the entire flow."""
    _backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(_backend_dir / '.env')

    from backend.logger import setup_logging
    setup_logging(log_prefix="processor")

    args = _parse_args()

    # --- Docker outer shell (build image & launch container) ---
    if args.mode == 'docker' and not os.environ.get('_DEEPWIKI_INSIDE_DOCKER'):
        _run_docker_build(args)
        return

    # --- Set config directory based on mode ---
    from backend.config import set_config_dir

    if args.mode == 'cloud':
        set_config_dir(_CONFIG_CLOUD)
        logger.info(f"Config: {_CONFIG_CLOUD}")
    elif args.mode == 'docker':
        set_config_dir(_CONFIG_DOCKER)
        logger.info(f"Config: {_CONFIG_DOCKER}")
    else:
        logger.info(f"Config: {_CONFIG_DEFAULT}")

    # --- Run processing ---
    try:
        _process(
            mode=args.mode,
            repo_url=args.repo,
            branch=args.branch,
            language=args.language,
            comprehensive=args.comprehensive,
            skip_codemap=args.skip_codemap,
            full_reprocess=args.full_reprocess,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
