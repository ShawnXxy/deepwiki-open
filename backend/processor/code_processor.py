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
        local  - PAT from env → Azure CLI identity (exclude MSI)
        docker - PAT from env only → error if missing
        cloud  - PAT from env → UMI from infra.json managed_identity.client_id

    Returns:
        Tuple of (token, token_type) where token_type is 'pat' or 'bearer'
    """
    pat = (os.environ.get('REPO_ACCESS_TOKEN', '')
           or os.environ.get('ADO_PAT', '')
           or os.environ.get('AZURE_DEVOPS_PAT', ''))

    if pat:
        masked = pat[:6] + '***' if len(pat) > 6 else '***'
        print(f"  Auth: PAT from environment ({masked})")
        return pat, 'pat'

    if mode == 'docker':
        print("  ERROR: Docker mode requires REPO_ACCESS_TOKEN in .env")
        sys.exit(1)

    try:
        from azure.identity import DefaultAzureCredential
        if mode == 'cloud':
            from backend.config import get_managed_identity_client_id
            msi_client_id = get_managed_identity_client_id()
            if not msi_client_id:
                print("  ERROR: managed_identity.client_id not set in infra.json")
                sys.exit(1)
            print(f"  Auth: managed identity ({msi_client_id[:8]}...)")
            credential = DefaultAzureCredential(
                managed_identity_client_id=msi_client_id,
            )
        else:
            print("  Auth: Azure CLI identity (no PAT found)...")
            credential = DefaultAzureCredential(
                exclude_managed_identity_credential=True,
            )
        token = credential.get_token(
            "499b84ac-1321-427f-aa17-267ca6975798/.default"
        )
        method = 'managed identity' if mode == 'cloud' else 'Azure CLI'
        print(f"  Auth: token acquired via {method}")
        return token.token, 'bearer'
    except Exception as e:
        logger.warning(f"Could not acquire Azure identity token: {e}")
        print("  ERROR: No authentication available for private repo.")
        if mode == 'cloud':
            print("  Check managed identity in infra.json")
        else:
            print("  1. Set REPO_ACCESS_TOKEN in backend/.env")
            print("  2. Run 'az login' first")
        sys.exit(1)


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
    print(f"  Auth: falling back to managed identity ({msi_client_id[:8]}...)")
    credential = DefaultAzureCredential(
        managed_identity_client_id=msi_client_id,
    )
    token = credential.get_token(
        "499b84ac-1321-427f-aa17-267ca6975798/.default"
    )
    print("  Auth: token acquired via managed identity (fallback)")
    return token.token, 'bearer'


def step_clone(repo_url, token, branch, repo_name, token_type='pat'):
    """Clone repository to local disk. Returns (repo_dir, commit_hash)."""
    from backend.modules.repository.git_ops import (
        download_repo, get_head_commit_hash,
    )
    from backend.paths import get_repos_path

    print("\n--- Step 1: Cloning repository ---")
    save_dir = os.path.join(get_repos_path(), repo_name)
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
    print(f"  Cloned to: {save_dir}")
    print(f"  Commit: {commit[:7] if commit else 'unknown'}")
    return save_dir, commit


def step_build_codemap(repo_path, owner, repo, repo_type, branch):
    """Build codemap graph from cloned repository."""
    from backend.modules.codemap.graph_builder import build_codemap
    from backend.modules.codemap.cache import save_codemap_cache

    print("\n--- Step: Building codemap graph ---")
    codemap = build_codemap(repo_path)
    codemap.metadata.owner = owner
    codemap.metadata.repo = repo
    codemap.metadata.repo_type = repo_type
    codemap.metadata.branch = branch

    save_codemap_cache(codemap, owner, repo, repo_type, branch)
    print(
        f"  CodeMap: {codemap.metadata.total_files} files, "
        f"{codemap.metadata.total_symbols} symbols, "
        f"{codemap.metadata.total_edges} edges"
    )
    return codemap


def step_embed(repo_url, token, branch):
    """Embed documents and build retriever. Returns RAG instance.

    Storage backend (local or blob) and AOAI auth (MSI or API key)
    are driven by config - this step does not need to know.
    """
    from backend.modules.embedder.retriever import RAG

    print("\n--- Step 2: Embedding documents ---")
    rag = RAG(provider='azure')
    rag.prepare_retriever(
        repo_url_or_path=repo_url,
        type='azuredevops',
        access_token=token,
        branch=branch,
        force_reprocess=True,
    )
    print(f"  Retriever ready ({len(rag.transformed_docs)} docs)")

    # Release components unused during processor wiki generation:
    # - generator: wiki_generator.py uses its own _call_llm()
    # - memory: no conversation history during batch processing
    # - db_manager: database already prepared, not needed further
    rag.generator = None
    rag.memory = None
    rag.db_manager = None

    return rag


def step_embed_cloud(repo_url, token, branch):
    """Embed documents and save to blob (cloud mode, no FAISS).

    Skips FAISS construction and document accumulation since wiki
    generation will use Azure AI Search for retrieval instead.
    Returns (chunk_count, owner, repo, branch_suffix) for downstream steps.
    """
    from backend.modules.embedder.indexer import DatabaseManager
    from backend.modules.embedder.document import (
        transform_documents_and_save_as_json,
    )
    from backend.clients.vector_storage import get_vector_storage

    print("\n--- Step 2: Embedding documents (cloud mode) ---")
    db_manager = DatabaseManager()
    db_manager._create_repo(
        repo_url, 'azuredevops', token, branch,
        force_reprocess=True,
    )

    repo_name = db_manager.repo_paths["repo_name"]
    branch_suffix = db_manager.repo_paths["branch_suffix"]

    # Delete legacy pkl if exists
    db_manager._delete_legacy_pkl(repo_name, branch_suffix)

    # Snapshot for orphan cleanup
    vector_storage = get_vector_storage()
    old_files = vector_storage.list_files(repo_name, branch_suffix)

    # Fused embed+save pipeline — skip all_embedded_docs accumulation
    chunk_count, _ = transform_documents_and_save_as_json(
        db_manager.repo_paths["save_repo_dir"],
        repo_name,
        branch_suffix,
        repo_url=repo_url,
        repo_type='azuredevops',
        skip_accumulate=True,
    )

    # Orphan cleanup
    if old_files:
        new_files = vector_storage.list_files(repo_name, branch_suffix)
        orphans = old_files - new_files
        if orphans:
            logger.info(
                f"[Vec] Cleaning {len(orphans)} orphan files"
            )
            vector_storage.delete_files(
                repo_name, branch_suffix, orphans
            )

    print(f"  Embedded {chunk_count} chunks (saved to blob, no FAISS)")
    return chunk_count


def step_generate_wiki(
    repo_url, branch, repo_path, retriever,
    commit_hash, language, comprehensive, owner, repo,
):
    """Generate wiki pages. Returns wiki_data."""
    from backend.processor.wiki_generator import generate_wiki

    print("\n--- Step 3: Generating wiki ---")
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
    )


def step_save_wiki(wiki_data, language, comprehensive):
    """Save wiki cache (storage backend determined by config)."""
    import asyncio
    from backend.modules.wiki.cache import save_wiki_cache
    from backend.modules.wiki.models import WikiCacheRequest

    print("\n--- Step 4: Saving wiki cache ---")
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
            print("  Wiki cache saved")
        else:
            print("  WARNING: save_wiki_cache returned False")
    except Exception as e:
        logger.error(f"Failed to save wiki cache: {e}", exc_info=True)
        print(f"  ERROR saving wiki cache: {e}")


def step_push_to_search(owner, repo, branch, wait=False):
    """Push vectors to AI Search and trigger indexer (cloud only).

    Loads and pushes in batches of PUSH_BATCH_SIZE to avoid holding
    all vectors in memory at once (~725MB for 50K chunks).

    Args:
        owner: Repo owner
        repo: Repo name
        branch: Branch name
        wait: If True, block until indexer completes (for cloud-first flow)
    """
    from backend.clients.search_client import (
        get_index_name, push_documents, run_indexer,
        index_exists, wait_for_indexer,
    )
    from backend.clients.vector_storage import get_vector_storage

    idx_name = get_index_name(owner, repo, branch or 'main')
    if not index_exists(idx_name):
        logger.info(f"AI Search index '{idx_name}' not found, skipping push")
        return

    print("\n--- Step 5: Pushing vectors to AI Search ---")
    # Use bare repo name (not owner_repo) — matches what
    # indexer.py._extract_repo_name_from_url() produces for ADO repos.
    repo_name = repo
    vector_storage = get_vector_storage()

    # Load all doc paths first (lightweight — just file listing)
    docs = vector_storage.load_documents(repo_name, branch)

    if not docs:
        print("  No vector documents found to push")
        return

    # Push in batches to limit memory during upload
    PUSH_BATCH_SIZE = 1000
    total = len(docs)
    pushed = 0
    for i in range(0, total, PUSH_BATCH_SIZE):
        batch = docs[i:i + PUSH_BATCH_SIZE]
        push_documents(idx_name, batch, repo_name, branch)
        pushed += len(batch)
        # Release batch vectors after push
        for doc in batch:
            doc.vector = None
        logger.info(
            f"Pushed batch {i // PUSH_BATCH_SIZE + 1}: "
            f"{pushed}/{total} docs"
        )

    del docs
    import gc
    gc.collect()

    print(f"  Pushed {pushed} documents to AI Search")
    run_indexer(idx_name)
    print("  Indexer triggered")

    if wait:
        print("  Waiting for indexer to complete...")
        success = wait_for_indexer(idx_name, timeout_seconds=600)
        if not success:
            print("  WARNING: Indexer did not complete in time")
            logger.error(
                "Indexer timeout — wiki generation may have "
                "incomplete results"
            )
        else:
            print("  Indexer completed successfully")


# ============================================================================
# Processing pipeline (all modes)
# ============================================================================

def step_generate_wiki_cloud(
    repo_url, branch, repo_path, owner, repo,
    commit_hash, language, comprehensive,
):
    """Generate wiki using Azure AI Search for retrieval (cloud mode).

    Creates a lightweight RAG instance with cloud search enabled.
    No FAISS, no in-memory document loading (~50 MB steady-state).
    """
    from backend.modules.embedder.retriever import RAG
    from backend.clients.search_client import get_index_name
    from backend.processor.wiki_generator import generate_wiki

    print("\n--- Step 3: Generating wiki (cloud retrieval) ---")

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
    )


def _process(mode, repo_url, branch, language, comprehensive,
             skip_codemap=False):
    """Run the processing pipeline using step functions.

    The mode determines auth; config (already set before this call)
    determines storage backend (local/blob) and retrieval (FAISS/AI Search).

    Cloud mode uses a reordered pipeline:
        embed → push to AI Search → wait for indexer → generate via Search
    This avoids building FAISS (~1.5 GB) during wiki generation.

    Local/Docker mode uses the original pipeline:
        embed + FAISS → generate via FAISS → push to Search (if configured)
    """
    owner, repo = _extract_owner_repo(repo_url)
    repo_name = f"{owner}_{repo}"

    print(f"\n{'='*60}")
    print("DeepWiki Code Processor")
    print(f"{'='*60}")
    print(f"  Repo:     {repo_url}")
    print(f"  Branch:   {branch}")
    print(f"  Mode:     {mode}")
    print(f"  Language: {language}")
    print(f"  Owner:    {owner}")
    print(f"  Repo:     {repo}")

    token, token_type = resolve_auth(mode)

    # Clone to local temp disk (all modes — even cloud clones locally
    # on AML compute; vectors and wiki go to blob via storage abstraction)
    try:
        repo_dir, commit_hash = step_clone(
            repo_url, token, branch, repo_name, token_type,
        )
    except ValueError:
        if mode == 'cloud' and token_type == 'pat':
            logger.warning(
                "PAT clone failed in cloud mode, "
                "falling back to managed identity"
            )
            print("  WARN: PAT clone failed, retrying with managed identity...")
            token, token_type = _resolve_umi_auth()
            repo_dir, commit_hash = step_clone(
                repo_url, token, branch, repo_name, token_type,
            )
        else:
            raise

    # Build codemap graph (all modes, unless skipped)
    if not skip_codemap:
        try:
            step_build_codemap(
                repo_dir, owner, repo, 'azuredevops', branch,
            )
        except Exception as e:
            logger.warning(f"Codemap generation failed (non-fatal): {e}")
            print(f"  WARNING: Codemap generation failed: {e}")

    if mode == 'cloud':
        # ============================================================
        # CLOUD MODE: embed → push to search → generate via search
        # No FAISS, no in-memory document loading (~120 MB peak)
        # ============================================================
        step_embed_cloud(repo_url, token, branch)

        step_push_to_search(owner, repo, branch, wait=True)

        try:
            wiki_data = step_generate_wiki_cloud(
                repo_url, branch, repo_dir, owner, repo,
                commit_hash, language, comprehensive,
            )
        except Exception as e:
            logger.error(
                f"Wiki generation failed: {e}", exc_info=True
            )
            print(f"\n  ERROR in wiki generation: {e}")
            sys.exit(1)

        step_save_wiki(wiki_data, language, comprehensive)
    else:
        # ============================================================
        # LOCAL / DOCKER MODE: embed + FAISS → generate via FAISS
        # (unchanged from existing implementation)
        # ============================================================
        retriever = step_embed(repo_url, token, branch)

        try:
            wiki_data = step_generate_wiki(
                repo_url, branch, repo_dir, retriever,
                commit_hash, language, comprehensive, owner, repo,
            )
        except Exception as e:
            logger.error(
                f"Wiki generation failed: {e}", exc_info=True
            )
            print(f"\n  ERROR in wiki generation: {e}")
            sys.exit(1)

        # Free FAISS index + transformed_docs before save/push
        del retriever
        gc.collect()

        step_save_wiki(wiki_data, language, comprehensive)

    print(f"\n{'='*60}")
    print("Processing complete")
    print(f"  Pages generated: {len(wiki_data.generated_pages)}")
    print(f"  Commit hash: {commit_hash[:7] if commit_hash else 'N/A'}")
    print(f"{'='*60}\n")

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
        print(f"  Dockerfile.processor not found at {dockerfile}")
        sys.exit(1)

    write_docker_config()

    print("\n--- Building Docker image ---")
    build_cmd = [
        'docker', 'build',
        '-f', str(dockerfile),
        '-t', image_name,
        str(project_root),
    ]
    print(f"  $ {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, cwd=str(project_root))
    if result.returncode != 0:
        print("  Docker build failed")
        sys.exit(1)
    print("  Image built")

    print("\n--- Running processor in container ---")
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

    print(f"  $ docker run ... {image_name} --repo=... --branch={args.branch}")
    result = subprocess.run(run_cmd)
    if result.returncode != 0:
        print("  Docker run failed")
        sys.exit(1)
    print("  Docker processing complete")


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
        print(f"Config: {_CONFIG_CLOUD}")
    elif args.mode == 'docker':
        set_config_dir(_CONFIG_DOCKER)
        print(f"Config: {_CONFIG_DOCKER}")
    else:
        print(f"Config: {_CONFIG_DEFAULT}")

    # --- Run processing ---
    _process(
        mode=args.mode,
        repo_url=args.repo,
        branch=args.branch,
        language=args.language,
        comprehensive=args.comprehensive,
        skip_codemap=args.skip_codemap,
    )


if __name__ == '__main__':
    main()
