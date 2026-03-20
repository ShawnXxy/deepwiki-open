"""
DeepWiki Code Processor — standalone CLI for wiki generation.

Usage:
    python -m backend.processor.code_processor --repo=URL --branch=main --mode=local
    python -m backend.processor.code_processor --config=run.json
    python -m backend.processor.code_processor --config=run.json --branch=dev

Modes:
    local   — Run directly in current Python env (PAT from .env)
    docker  — Build Docker image and run inside container
    cloud   — Submit Azure ML pipeline job (MSI auth)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments with --config file support."""
    parser = argparse.ArgumentParser(
        description='DeepWiki Code Processor — generate wikis for code repositories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --repo="https://dev.azure.com/org/proj/_git/repo" --branch=main --mode=local
  %(prog)s --config=run.json
  %(prog)s --config=run.json --branch=dev
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
                        help=argparse.SUPPRESS)  # Hidden, default true
    parser.add_argument('--language', type=str, default=None,
                        help=argparse.SUPPRESS)  # Hidden, default 'en'

    args = parser.parse_args()

    # Load config file if specified
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            parser.error(f"Config file not found: {args.config}")
        with open(config_path, 'r') as f:
            config = json.load(f)

    # CLI args override config file values
    final = {
        'repo': args.repo or config.get('repo'),
        'branch': args.branch or config.get('branch'),
        'mode': args.mode or config.get('mode'),
        'comprehensive': True,  # Always comprehensive
        'language': args.language or config.get('language', 'en'),
    }

    # Handle comprehensive as string from CLI
    if args.comprehensive is not None:
        final['comprehensive'] = args.comprehensive.lower() in ('true', '1', 'yes')

    # Validate required params
    if not final['repo']:
        parser.error("--repo is required (or set 'repo' in config file)")
    if not final['branch']:
        parser.error("--branch is required (or set 'branch' in config file)")
    if not final['mode']:
        parser.error("--mode is required (or set 'mode' in config file)")

    return argparse.Namespace(**final)


def _extract_owner_repo(repo_url: str) -> tuple:
    """Extract owner (organization) and repo name from Azure DevOps URL.

    The owner is the ADO **organization**, not the project name.
    This matches the frontend's URL routing: /:owner/:repo.

    Examples:
        https://dev.azure.com/org/proj/_git/repo → ('org', 'repo')
        https://msdata.visualstudio.com/Database%20Systems/_git/orcasql-myfile
            → ('msdata', 'orcasql-myfile')
    """
    from urllib.parse import unquote, urlparse
    url = unquote(repo_url.rstrip('/'))
    parsed = urlparse(url)
    host = parsed.hostname or ''
    path_parts = [p for p in parsed.path.split('/') if p]

    # Azure DevOps: extract org from URL and repo from /_git/ segment
    if '/_git/' in url and path_parts:
        # Repo name is after _git
        git_idx = path_parts.index('_git') if '_git' in path_parts else -1
        repo = path_parts[git_idx + 1] if git_idx >= 0 and git_idx + 1 < len(path_parts) else 'unknown'

        # Organization extraction:
        # - visualstudio.com: org is subdomain (msdata.visualstudio.com → msdata)
        # - dev.azure.com: org is first path segment (dev.azure.com/org/proj/_git/repo → org)
        if 'visualstudio.com' in host:
            owner = host.split('.')[0]
        elif 'dev.azure.com' in host:
            owner = path_parts[0] if path_parts else 'unknown'
        else:
            # Fallback: use segment before _git (project name)
            owner = path_parts[git_idx - 1] if git_idx >= 1 else 'unknown'

        return owner, repo

    # Fallback: last two segments
    if len(path_parts) >= 2:
        return path_parts[-2], path_parts[-1]
    return 'unknown', path_parts[-1] if path_parts else 'unknown'


def run_code_processor(
    repo_url: str,
    branch: str,
    mode: str = 'local',
    language: str = 'en',
    comprehensive: bool = True,
):
    """Run the code processing pipeline.

    Args:
        repo_url: Azure DevOps repository URL
        branch: Branch name to process
        mode: 'local', 'docker', or 'cloud'
        language: Wiki language code (default 'en')
        comprehensive: True for 15-25 pages (default)
    """
    from backend.modules.repository.git_ops import (
        download_repo, get_head_commit_hash,
    )
    from backend.modules.rag.retriever import RAG
    from backend.modules.wiki.cache import save_wiki_cache
    from backend.modules.wiki.models import WikiCacheRequest
    from backend.processor.wiki_generator import generate_wiki

    repo_type = 'azuredevops'  # Only ADO supported
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

    # --- Auth: resolve PAT or Azure identity token ---
    pat = os.environ.get('REPO_ACCESS_TOKEN', '')
    if not pat:
        # Try other common env var names
        pat = (os.environ.get('ADO_PAT', '')
               or os.environ.get('AZURE_DEVOPS_PAT', ''))

    if mode == 'docker' and not pat:
        print("  ✗ ERROR: Docker mode requires REPO_ACCESS_TOKEN in .env")
        print("  Set REPO_ACCESS_TOKEN in your .env file and retry.")
        sys.exit(1)

    if not pat:
        # No PAT available — try to get a token from Azure CLI / MSI
        # This covers: local (user's az login) and cloud (managed identity)
        try:
            from azure.identity import DefaultAzureCredential
            print("  No PAT found — acquiring Azure DevOps token via identity...")
            credential = DefaultAzureCredential()
            # Azure DevOps resource ID for token scope
            token = credential.get_token(
                "499b84ac-1321-427f-aa17-267ca6975798/.default"
            )
            pat = token.token
            auth_method = ("MSI" if mode == "cloud"
                           else "Azure CLI / user identity")
            print(f"  ✓ Token acquired via {auth_method}")
        except Exception as e:
            logger.warning(f"Could not acquire Azure identity token: {e}")
            print("  ✗ ERROR: No authentication available for private repo.")
            print("  Options:")
            print("    1. Set REPO_ACCESS_TOKEN in backend/.env")
            print("    2. Run 'az login' first (local mode)")
            sys.exit(1)
    else:
        print(f"  ✓ Using PAT from environment")

    # Step 1: Clone repo
    print("\n--- Step 1: Cloning repository ---")
    from backend.utils.paths import get_repos_path
    save_repo_dir = os.path.join(get_repos_path(), repo_name)

    download_repo(
        repo_url=repo_url,
        local_path=save_repo_dir,
        type=repo_type,
        access_token=pat,
        branch=branch,
        force_update=True,
    )
    print(f"  ✓ Cloned to: {save_repo_dir}")

    commit_hash = get_head_commit_hash(save_repo_dir)
    print(f"  ✓ Commit: {commit_hash[:7] if commit_hash else 'unknown'}")

    # Step 2: Embed documents (RAG preparation)
    print("\n--- Step 2: Embedding documents ---")
    request_rag = RAG(provider='azure')
    request_rag.prepare_retriever(
        repo_url_or_path=repo_url,
        type=repo_type,
        access_token=pat,
        branch=branch,
        force_reprocess=True,
    )
    print(f"  ✓ Retriever ready ({len(request_rag.transformed_docs)} docs)")

    # Step 3: Generate wiki
    print("\n--- Step 3: Generating wiki ---")
    wiki_data = generate_wiki(
        repo_url=repo_url,
        branch=branch,
        repo_type=repo_type,
        repo_path=save_repo_dir,
        retriever=request_rag,
        commit_hash=commit_hash,
        language=language,
        comprehensive=comprehensive,
        owner=owner,
        repo=repo,
    )

    # Step 4: Save wiki cache
    print("\n--- Step 4: Saving wiki cache ---")
    import asyncio

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

    # save_wiki_cache is async, run it in event loop
    result = asyncio.run(save_wiki_cache(cache_request))
    if result:
        print("  ✓ Wiki cache saved successfully")
    else:
        print("  ✗ Failed to save wiki cache")

    print(f"\n{'='*60}")
    print("✓ Code processing complete")
    print(f"  Pages generated: {len(wiki_data.generated_pages)}")
    print(f"  Commit hash: {commit_hash[:7] if commit_hash else 'N/A'}")
    print(f"{'='*60}\n")

    return wiki_data


def main():
    """CLI entry point."""
    # Load backend/.env for PAT and OpenAI key
    _backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(_backend_dir / '.env')

    # Setup logging
    from backend.infra.logger import setup_logging
    setup_logging(log_prefix="processor")

    args = _parse_args()

    if args.mode == 'docker':
        print("Docker mode: building and running in container...")
        _run_docker_mode(args)
    elif args.mode == 'cloud':
        print("Cloud mode: submitting Azure ML pipeline job...")
        _run_cloud_mode(args)
    else:
        # Local mode
        run_code_processor(
            repo_url=args.repo,
            branch=args.branch,
            mode='local',
            language=args.language,
            comprehensive=args.comprehensive,
        )


def _run_docker_mode(args):
    """Build Docker image and run processor inside container."""
    # Placeholder for Phase 3B implementation
    print("Docker mode is not yet implemented.")
    print("Use --mode=local to run directly.")
    sys.exit(1)


def _run_cloud_mode(args):
    """Submit Azure ML pipeline job."""
    # Placeholder for Phase 3C/3D implementation
    print("Cloud mode is not yet implemented.")
    print("Use --mode=local to run directly.")
    sys.exit(1)


if __name__ == '__main__':
    main()
