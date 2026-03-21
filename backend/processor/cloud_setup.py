"""
Cloud resource orchestration for DeepWiki.

Creates/updates per-repo Azure resources:
- AI Search index (1 per repo+branch)
- AML scheduled pipeline (1 per repo+branch)
- Auto-scales AML compute cluster capacity

Usage:
    from backend.processor.cloud_setup import setup_cloud_resources, teardown_cloud_resources

    setup_cloud_resources(repo_url, branch, owner, repo)
    teardown_cloud_resources(owner, repo, branch)
"""

import logging

from backend.clients.search_client import (
    get_index_name, create_or_update_index,
    delete_index,
)
from backend.config import get_aml_config, is_search_configured

logger = logging.getLogger(__name__)


def _get_aml_pipeline_config() -> dict:
    """Get AML pipeline config from infra.json azure_ml section."""
    aml_config = get_aml_config()
    if not aml_config:
        return {
            "compute_name": "deepwiki-compute",
            "compute_size": "STANDARD_D2_V2",
            "compute_min_instances": 0,
            "compute_max_instances": 4,
            "schedule_interval_hours": 480,
            "environment_name": "deepwiki-processor",
        }
    # Convert Pydantic model to dict, use defaults for missing fields
    config = aml_config.model_dump() if hasattr(aml_config, 'model_dump') else {}
    defaults = {
        "compute_name": "deepwiki-compute",
        "compute_size": "STANDARD_D2_V2",
        "compute_min_instances": 0,
        "compute_max_instances": 4,
        "schedule_interval_hours": 480,
        "environment_name": "deepwiki-processor",
        "idle_time_before_scale_down": 600,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)
    return config


def _get_ml_client():
    """Create an authenticated MLClient."""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    aml_config = get_aml_config()
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=aml_config.subscription_id,
        resource_group_name=aml_config.resource_group,
        workspace_name=aml_config.workspace_name,
    )


def _pipeline_name(owner: str, repo: str, branch: str) -> str:
    """Derive AML pipeline/schedule name."""
    import re
    name = f"deepwiki-{owner}-{repo}-{branch}".lower()
    return re.sub(r'[^a-z0-9-]', '-', name)[:128]


def setup_cloud_resources(
    repo_url: str,
    branch: str,
    owner: str,
    repo: str,
) -> dict:
    """Create/update all cloud resources for a repo.

    1. AI Search index (if search is configured)
    2. AML scheduled pipeline (if AML is configured)

    Args:
        repo_url: Full repository URL
        branch: Branch name
        owner: Repository owner (organization)
        repo: Repository name

    Returns:
        Dict with created resource names
    """
    result = {}

    # --- AI Search index ---
    if is_search_configured():
        index_name = get_index_name(owner, repo, branch)
        print(f"\n--- Setting up AI Search index: {index_name} ---")
        create_or_update_index(index_name)
        result['search_index'] = index_name
        print(f"  ✓ Index ready: {index_name}")
    else:
        print("  ⊘ AI Search not configured (skipping)")

    # --- AML pipeline ---
    aml_config = get_aml_config()
    if aml_config and aml_config.enabled:
        pipeline_config = _get_aml_pipeline_config()
        name = _pipeline_name(owner, repo, branch)
        print(f"\n--- Setting up AML pipeline: {name} ---")

        try:
            ml_client = _get_ml_client()

            # Ensure compute exists
            _ensure_compute(ml_client, pipeline_config)

            # Create/update scheduled pipeline
            _create_or_update_pipeline(
                ml_client, name, repo_url, branch,
                owner, repo, pipeline_config,
            )
            result['aml_pipeline'] = name
            print(f"  ✓ Pipeline ready: {name}")
        except Exception as e:
            logger.error(f"AML setup failed: {e}")
            print(f"  ✗ AML setup failed: {e}")
    else:
        print("  ⊘ Azure ML not configured (skipping)")

    return result


def _ensure_compute(ml_client, config: dict) -> None:
    """Ensure AML compute cluster exists with sufficient capacity."""
    from azure.ai.ml.entities import AmlCompute

    compute_name = config['compute_name']

    try:
        compute = ml_client.compute.get(compute_name)
        logger.info(
            f"Compute {compute_name} exists: "
            f"max_instances={compute.max_instances}"
        )
        print(f"  ✓ Compute cluster exists: {compute_name}")
    except Exception:
        # Create new cluster
        compute = AmlCompute(
            name=compute_name,
            size=config.get('compute_size', 'STANDARD_D2_V2'),
            min_instances=config.get('compute_min_instances', 0),
            max_instances=config.get('compute_max_instances', 4),
            idle_time_before_scale_down=600,
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"  ✓ Created compute: {compute_name}")


def _create_or_update_pipeline(
    ml_client, name: str,
    repo_url: str, branch: str,
    owner: str, repo: str,
    config: dict,
) -> None:
    """Create or update an AML scheduled pipeline job."""
    from azure.ai.ml import command
    from azure.ai.ml.entities import (
        RecurrenceTrigger,
        JobSchedule,
    )

    # Build the command that runs inside AML
    cmd = (
        f"python -m backend.processor.code_processor "
        f"--repo {repo_url} --branch {branch} --mode local"
    )

    # Create command job
    job = command(
        name=name,
        display_name=f"DeepWiki: {owner}/{repo} ({branch})",
        command=cmd,
        compute=config['compute_name'],
        environment=f"{config['environment_name']}@latest",
    )

    # Check if schedule already exists
    try:
        ml_client.schedules.get(name)
        # Disable old, create new
        ml_client.schedules.begin_disable(name).result()
        logger.info(f"Disabled existing schedule: {name}")
    except Exception:
        pass  # Schedule doesn't exist yet

    # Create recurring schedule
    interval_hours = config.get('schedule_interval_hours', 480)
    schedule = JobSchedule(
        name=name,
        trigger=RecurrenceTrigger(
            frequency="hour",
            interval=interval_hours,
        ),
        create_job=job,
    )

    ml_client.schedules.begin_create_or_update(schedule).result()
    logger.info(f"Created/updated schedule: {name} (every {interval_hours}h)")


def teardown_cloud_resources(owner: str, repo: str, branch: str) -> None:
    """Remove all cloud resources for a repo.

    Deletes AI Search index and AML scheduled pipeline.
    """
    # Delete AI Search index
    if is_search_configured():
        index_name = get_index_name(owner, repo, branch)
        print(f"  Deleting AI Search index: {index_name}")
        delete_index(index_name)

    # Delete AML pipeline
    aml_config = get_aml_config()
    if aml_config and aml_config.enabled:
        name = _pipeline_name(owner, repo, branch)
        print(f"  Deleting AML pipeline: {name}")
        try:
            ml_client = _get_ml_client()
            ml_client.schedules.begin_disable(name).result()
            ml_client.schedules.begin_delete(name).result()
            logger.info(f"Deleted AML schedule: {name}")
        except Exception as e:
            logger.warning(f"Could not delete AML schedule {name}: {e}")
