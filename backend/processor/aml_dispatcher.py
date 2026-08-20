"""
AML Dispatcher — sets up Azure ML resources for cloud processing.

This is the ONLY entry point users run for cloud mode.
It copies config, creates AML resources, and exits.
The AML pipeline will then run code_processor.py --mode cloud inside AML.

Usage:
    python -m backend.processor.aml_dispatcher --config=backend/run.json
    python -m backend.processor.aml_dispatcher --config=backend/run.json --run-now
    python -m backend.processor.aml_dispatcher --repo=URL --branch=main
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='DeepWiki AML Dispatcher — provision cloud resources',
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (e.g., backend/run.json)')
    parser.add_argument('--repo', type=str, default=None,
                        help='Azure DevOps repo URL')
    parser.add_argument('--branch', type=str, default=None,
                        help='Branch name')
    parser.add_argument(
        '--run-now',
        action='store_true',
        help=(
            'Submit the AML pipeline immediately after resource setup. '
            'New schedules run immediately without this flag.'
        ),
    )

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
        'run_now': args.run_now,
    }
    if not final['repo']:
        parser.error("--repo is required (or set 'repo' in config file)")
    if not final['branch']:
        parser.error("--branch is required (or set 'branch' in config file)")

    return argparse.Namespace(**final)


def main():
    """Dispatch: copy config -> setup AML resources -> exit."""
    _backend_dir = Path(__file__).resolve().parents[1]
    load_dotenv(_backend_dir / '.env')

    from backend.logger import setup_logging
    setup_logging(log_prefix="aml_dispatcher")

    args = _parse_args()

    from backend.processor.code_processor import _extract_owner_repo
    owner, repo = _extract_owner_repo(args.repo)

    logger.info(
        f"DeepWiki AML Dispatcher: repo={args.repo}, branch={args.branch}, "
        f"owner={owner}, repo={repo}, run_now={args.run_now}"
    )

    # Step 1: Copy config to .cloud/ with Azure services force-enabled
    from backend.processor.cloud_setup import write_cloud_config
    write_cloud_config()

    # Step 2: Enable cloud services in-memory (for setup_cloud_resources)
    from backend.config import enable_cloud_services
    enable_cloud_services()

    # Step 3: Setup AML resources (compute, AI Search, pipeline, image)
    from backend.processor.cloud_setup import setup_cloud_resources
    logger.info("Starting cloud resource setup...")
    resources = setup_cloud_resources(
        repo_url=args.repo,
        branch=args.branch,
        owner=owner,
        repo=repo,
        run_now=args.run_now,
    )
    logger.info(f"Cloud setup complete: resources={resources}")


if __name__ == '__main__':
    main()
