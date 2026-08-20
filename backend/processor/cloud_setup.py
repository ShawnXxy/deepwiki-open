"""
Cloud resource orchestration for DeepWiki.

Creates/updates per-repo Azure resources:
- AI Search index + data source + indexer (1 per repo+branch)
- AML scheduled pipeline (1 per repo+branch)
- Auto-scales AML compute cluster capacity

Usage:
    from backend.processor.cloud_setup import setup_cloud_resources, teardown_cloud_resources

    setup_cloud_resources(repo_url, branch, owner, repo)
    teardown_cloud_resources(owner, repo, branch)
"""

import json
import logging
import shutil

from backend.clients.search_client import (
    get_index_name, create_or_update_index,
    delete_index, delete_indexer,
    create_data_source, create_indexer,
)
from backend.config import (
    get_aml_config, get_search_config, is_search_configured,
)

logger = logging.getLogger(__name__)


def write_cloud_config() -> None:
    """Copy config files to .cloud/ with Azure services force-enabled.

    Creates ``backend/config/.cloud/`` containing a copy of every JSON
    config file.  ``infra.json`` is patched so that
    ``azure_blob_storage``, ``azure_ai_search``, and ``azure_ml`` all
    have ``enabled: true``.  Other files are copied as-is.

    The .cloud/ directory is:
    - Committed in the AML code snapshot (not in .amlignore)
    - Ignored by git (.gitignore has backend/config/.cloud/)

    Inside AML, ``config.py`` detects .cloud/ and reads from it,
    so blob/search/AML are active without needing in-memory mutation.
    """
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "config"
    cloud_dir = config_dir / ".cloud"
    cloud_dir.mkdir(parents=True, exist_ok=True)

    # Copy all JSON config files
    for src in config_dir.glob("*.json"):
        dst = cloud_dir / src.name
        if src.name == "infra.json":
            # Patch: force-enable cloud services
            with open(src, 'r', encoding='utf-8') as f:
                infra = json.load(f)
            for section in ("azure_blob_storage", "azure_ai_search", "azure_ml"):
                if section in infra:
                    infra[section]["enabled"] = True
            with open(dst, 'w', encoding='utf-8') as f:
                json.dump(infra, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote cloud config: {dst} (services enabled)")
        else:
            shutil.copy2(src, dst)
            logger.info(f"Copied config: {dst}")

    logger.info(f"Cloud config written to {cloud_dir}")


def write_docker_config() -> None:
    """Copy config files to .local/ with cloud services force-disabled.

    Creates ``backend/config/.local/`` containing a copy of every JSON
    config file.  ``infra.json`` is patched so that
    ``azure_blob_storage``, ``azure_ai_search``, and ``azure_ml`` all
    have ``enabled: false``.

    Inside Docker, the processor reads from .local/ so it uses
    local disk storage and FAISS — no cloud dependencies.
    """
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "config"
    local_dir = config_dir / ".local"
    local_dir.mkdir(parents=True, exist_ok=True)

    for src in config_dir.glob("*.json"):
        dst = local_dir / src.name
        if src.name == "infra.json":
            with open(src, 'r', encoding='utf-8') as f:
                infra = json.load(f)
            for section in ("azure_blob_storage", "azure_ai_search", "azure_ml"):
                if section in infra:
                    infra[section]["enabled"] = False
            with open(dst, 'w', encoding='utf-8') as f:
                json.dump(infra, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote docker config: {dst} (services disabled)")
        else:
            shutil.copy2(src, dst)
            logger.info(f"Copied config: {dst}")

    logger.info(f"Docker config written to {local_dir}")


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

    from backend.config import get_infra_config
    infra = get_infra_config()
    aml_config = infra.azure_ml
    account = infra.account
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=account.subscription_id,
        resource_group_name=account.resource_group,
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
    run_now: bool = False,
) -> dict:
    """Create/update cloud resources and optionally submit the AML pipeline.

    Args:
        repo_url: Full repository URL.
        branch: Branch name.
        owner: Repository owner or organization.
        repo: Repository name.
        run_now: Submit an immediate job for an existing schedule. New
            schedules always submit one immediate job.

    Returns:
        Created resource names. Includes ``aml_job`` only when an immediate
        job was submitted.
    """
    result = {}
    repo_name = f"{owner}_{repo}"

    # --- AI Search: index + data source + indexer ---
    if is_search_configured():
        search_config = get_search_config()
        index_name = get_index_name(owner, repo, branch)
        logger.info(f"Setting up AI Search: {index_name}")

        # Optionally recreate (delete first) on initial run
        if search_config.recreate_index:
            logger.info("Recreating index (recreate_index=true)")
            delete_indexer(index_name)
            delete_index(index_name)

        create_or_update_index(index_name)
        logger.info(f"Index ready: {index_name}")

        ds_name = create_data_source(index_name, repo_name, branch)
        logger.info(f"Data source ready: {ds_name}")

        indexer_name = create_indexer(
            index_name, ds_name,
            interval=search_config.indexer_interval,
        )
        logger.info(f"Indexer ready: {indexer_name} "
              f"(schedule={search_config.indexer_interval})")

        result['search_index'] = index_name
        result['search_indexer'] = indexer_name
    else:
        logger.info("AI Search not configured (skipping)")

    # --- AML pipeline ---
    aml_config = get_aml_config()
    if aml_config and aml_config.enabled:
        pipeline_config = _get_aml_pipeline_config()
        name = _pipeline_name(owner, repo, branch)
        logger.info(f"Setting up AML pipeline: {name}")

        try:
            ml_client = _get_ml_client()

            # Ensure environment is registered
            _ensure_environment(ml_client, pipeline_config)

            # Ensure compute exists
            _ensure_compute(ml_client, pipeline_config)

            # Create/update scheduled pipeline
            submitted_job_name = _create_or_update_pipeline(
                ml_client, name, repo_url, branch,
                owner, repo, pipeline_config, run_now=run_now,
            )
            result['aml_pipeline'] = name
            if submitted_job_name:
                result['aml_job'] = submitted_job_name
            logger.info(f"Pipeline ready: {name}")
        except Exception as e:
            logger.error(f"AML setup failed: {e}")
            raise
    else:
        logger.info("Azure ML not configured (skipping)")

    return result


def _ensure_environment(ml_client, config: dict) -> None:
    """Register or update AML environment from Dockerfile.processor."""
    from azure.ai.ml.entities import Environment, BuildContext
    from pathlib import Path

    env_name = config.get('environment_name', 'deepwiki-processor')

    project_root = Path(__file__).resolve().parents[2]
    dockerfile = project_root / 'Dockerfile.processor'

    if not dockerfile.is_file():
        logger.warning(
            f"Dockerfile.processor not found at {dockerfile}, "
            f"skipping environment creation"
        )

        return

    env = Environment(
        name=env_name,
        description="DeepWiki processor environment",
        build=BuildContext(
            path=str(project_root),
            dockerfile_path="Dockerfile.processor",
        ),
    )

    ml_client.environments.create_or_update(env)
    logger.info(f"Environment created/updated: {env_name}")


def _ensure_compute(ml_client, config: dict) -> None:
    """Create or update AML compute cluster with managed identity."""
    from azure.ai.ml.entities import (
        AmlCompute, ManagedIdentityConfiguration,
        IdentityConfiguration,
    )
    from backend.config import get_infra_config

    infra = get_infra_config()
    compute_name = config['compute_name']

    # Build managed identity resource ID
    mi_resource_id = (
        f"/subscriptions/{infra.account.subscription_id}"
        f"/resourceGroups/{infra.account.resource_group}"
        f"/providers/Microsoft.ManagedIdentity"
        f"/userAssignedIdentities/{infra.managed_identity.name}"
    )

    compute = AmlCompute(
        name=compute_name,
        size=config.get('compute_size', 'STANDARD_D2_V2'),
        min_instances=config.get('compute_min_instances', 0),
        max_instances=config.get('compute_max_instances', 4),
        idle_time_before_scale_down=config.get(
            'idle_time_before_scale_down', 600
        ),
        identity=IdentityConfiguration(
            type="user_assigned",
            user_assigned_identities=[
                ManagedIdentityConfiguration(
                    resource_id=mi_resource_id,
                )
            ],
        ),
    )
    ml_client.compute.begin_create_or_update(compute).result()
    logger.info(f"Compute created/updated: {compute_name} "
          f"(identity: {infra.managed_identity.name})")


def _create_or_update_pipeline(
    ml_client, name: str,
    repo_url: str, branch: str,
    owner: str, repo: str,
    config: dict,
    run_now: bool = False,
) -> str | None:
    """Create or update an AML schedule and optionally submit its pipeline."""
    import os
    import re
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from azure.ai.ml import command
    from azure.ai.ml.constants import TimeZone
    from azure.ai.ml.dsl import pipeline
    from azure.ai.ml.entities import (
        RecurrenceTrigger,
        JobSchedule,
    )
    from azure.core.exceptions import ResourceNotFoundError

    # Command that runs INSIDE AML.
    # --mode cloud makes the processor read from config/.cloud/
    # which has Azure services force-enabled.
    cmd = (
        f"python -m backend.processor.code_processor "
        f"--repo {repo_url} --branch {branch} --mode cloud"
    )

    # Project root — uploaded as AML code snapshot
    # .amlignore filters out frontend/node_modules/etc.
    project_root = str(Path(__file__).resolve().parents[2])

    # Pass REPO_ACCESS_TOKEN to the AML job if available.
    # In cloud mode, resolve_auth() tries PAT first before MSI.
    # Without this, the PAT from local .env never reaches AML.
    env_vars = {}
    pat = (os.environ.get('REPO_ACCESS_TOKEN', ''))
    if pat:
        env_vars['REPO_ACCESS_TOKEN'] = pat
        masked = pat[:6] + '***' if len(pat) > 6 else '***'
        logger.info(f"Passing REPO_ACCESS_TOKEN to AML job ({masked})")

    # Define the command component with code upload.
    # is_deterministic=False: target git repo HEAD changes outside AML's view,
    # so step reuse must be disabled — otherwise scheduled runs return cached
    # output from the previous fresh run (silent no-op, ~2s "Completed").
    processor_command = command(
        name=f"{name}-step",
        display_name=f"DeepWiki: {owner}/{repo} ({branch})",
        command=cmd,
        compute=config['compute_name'],
        environment=f"{config['environment_name']}@latest",
        code=project_root,
        environment_variables=env_vars if env_vars else None,
        is_deterministic=False,
    )

    # Wrap in a pipeline (schedules require PipelineJob, not CommandJob)
    @pipeline(
        name=name,
        display_name=f"DeepWiki Pipeline: {owner}/{repo} ({branch})",
        compute=config['compute_name'],
    )
    def deepwiki_pipeline():
        processor_command()

    pipeline_job = deepwiki_pipeline()
    # Experiment name: letters, numbers, underscores, dashes only
    experiment = re.sub(r'[^a-zA-Z0-9_-]', '-', f"{owner}-{repo}-{branch}")
    pipeline_job.experiment_name = experiment

    try:
        ml_client.schedules.get(name)
    except ResourceNotFoundError:
        is_new = True
        logger.info(f"Schedule does not exist yet: {name}")
    else:
        is_new = False
        logger.info(f"Updating existing schedule: {name}")

    interval_hours = config.get('schedule_interval_hours', 480)
    next_run = datetime.now(timezone.utc) + timedelta(
        hours=interval_hours
    )
    next_run_iso = next_run.strftime("%Y-%m-%dT%H:%M:%S")
    schedule = JobSchedule(
        name=name,
        trigger=RecurrenceTrigger(
            frequency="hour",
            interval=interval_hours,
            start_time=next_run_iso,
            time_zone=TimeZone.UTC,
        ),
        create_job=pipeline_job,
    )

    updated_schedule = (
        ml_client.schedules.begin_create_or_update(schedule).result()
    )
    if not updated_schedule.is_enabled:
        ml_client.schedules.begin_enable(name).result()
        logger.info(f"Enabled schedule: {name}")
    logger.info(
        f"Created/updated schedule: {name} "
        f"(every {interval_hours}h, next run={next_run_iso})"
    )

    if not (is_new or run_now):
        logger.info(
            f"Immediate job not submitted for existing schedule: {name}"
        )
        return None

    reason = "new schedule" if is_new else "--run-now requested"
    logger.info(f"Submitting immediate AML pipeline job ({reason}): {name}")
    try:
        submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    except Exception:
        logger.exception(
            f"Immediate AML job submission failed for {name}. "
            "The schedule is configured; retry with --run-now."
        )
        raise

    logger.info(
        f"Submitted AML job: name={submitted_job.name}, "
        f"status={submitted_job.status}"
    )
    return submitted_job.name


def teardown_cloud_resources(owner: str, repo: str, branch: str) -> None:
    """Remove all cloud resources for a repo.

    Deletes AI Search index, indexer, data source, and AML scheduled pipeline.
    """
    # Delete AI Search resources (indexer + data source + index)
    if is_search_configured():
        index_name = get_index_name(owner, repo, branch)
        logger.info(f"Deleting AI Search resources: {index_name}")
        delete_indexer(index_name)
        delete_index(index_name)

    # Delete AML pipeline
    aml_config = get_aml_config()
    if aml_config and aml_config.enabled:
        name = _pipeline_name(owner, repo, branch)
        logger.info(f"Deleting AML pipeline: {name}")
        try:
            ml_client = _get_ml_client()
            ml_client.schedules.begin_disable(name).result()
            ml_client.schedules.begin_delete(name).result()
            logger.info(f"Deleted AML schedule: {name}")
        except Exception as e:
            logger.warning(f"Could not delete AML schedule {name}: {e}")
