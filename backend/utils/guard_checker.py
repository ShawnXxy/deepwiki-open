"""
Azure OpenAI content-filter (RAI policy) inspector.

Reads the configured Azure OpenAI resource from ``infra.json`` and queries
the Azure Resource Manager (management plane) to report which RAI policy
each deployment uses and what categories that policy blocks.

Authentication
--------------
Uses :class:`azure.identity.DefaultAzureCredential` so the same code works
in both local and cloud modes:

* **Local** (``mode=local``) — falls through MSI → Azure CLI → VS Code, so
  the developer's own AAD identity is used (requires ``az login``).
* **Cloud** (``mode=cloud``, Container Apps / AML / App Service) — uses
  the user-assigned managed identity whose ``client_id`` is recorded in
  ``infra.json`` under ``managed_identity.client_id``.

Disabling
---------
The checker can be turned off three ways (any one is sufficient):

1. ``check_content_filters(enabled=False)`` — explicit caller override.
2. Env var ``DEEPWIKI_GUARD_CHECKER_DISABLED=1``.
3. Docker / containerised processor runs — auto-disabled when the
   processor exports ``_DEEPWIKI_INSIDE_DOCKER=1`` (set by
   ``code_processor.py --mode=docker``).

Public API
----------
* :func:`check_content_filters` — returns a structured report dict.
* :func:`format_filter_report` — pretty-prints the dict for logs / CLI.
* :func:`update_content_filter` — toggle / tune one filter on a policy.

CLI
---
Run as a module to inspect the currently configured account::

    python -m backend.utils.guard_checker

Apply a change (requires ``--confirm``)::

    python -m backend.utils.guard_checker \
        --policy CustomContentFilter403 \
        --filter Profanity --source Completion \
        --enabled false --blocking false --confirm

.. warning::
   Some categories (Hate / Sexual / Violence / Self-harm at High,
   Jailbreak, Indirect Attack) are gated by Azure's *Modified Content
   Filters* approval and the PUT call will fail with HTTP 400 unless
   the subscription has been granted that exemption.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

from backend.config import get_account_config, get_infra_config

logger = logging.getLogger(__name__)

# Management-plane API version that supports RAI policy CRUD.
_ARM_API_VERSION = "2024-10-01"
_ARM_HOST = "https://management.azure.com"
_ARM_SCOPE = "https://management.azure.com/.default"


# ---------------------------------------------------------------------------
# Enable / disable detection
# ---------------------------------------------------------------------------

def is_guard_check_enabled(enabled: Optional[bool] = None) -> bool:
    """Resolve whether the guard checker should run.

    Resolution order:

    1. Explicit ``enabled`` argument (``True``/``False``) — wins outright.
    2. Env var ``DEEPWIKI_GUARD_CHECKER_DISABLED`` — when truthy, disables.
    3. Docker auto-detection — when ``_DEEPWIKI_INSIDE_DOCKER`` is truthy
       (set by the processor for ``--mode=docker``), disables.
    4. Default — enabled.
    """
    if enabled is not None:
        return bool(enabled)

    def _truthy(name: str) -> bool:
        return (os.environ.get(name) or "").strip().lower() in {
            "1", "true", "yes", "on",
        }

    if _truthy("DEEPWIKI_GUARD_CHECKER_DISABLED"):
        return False
    if _truthy("_DEEPWIKI_INSIDE_DOCKER"):
        return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _account_name_from_endpoint(endpoint: str) -> str:
    """Extract the Cognitive Services account name from an endpoint URL.

    e.g. ``https://aoai-orcas-deepwiki-kc.openai.azure.com`` →
    ``aoai-orcas-deepwiki-kc``.
    """
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint is empty in infra.json")
    host = urlparse(endpoint).hostname or ""
    if not host:
        raise ValueError(f"Cannot parse hostname from endpoint: {endpoint!r}")
    return host.split(".", 1)[0]


def _build_credential() -> DefaultAzureCredential:
    """Build a credential matching the project's auth pattern.

    * Cloud: MSI step inside ``DefaultAzureCredential`` succeeds first
      and uses the user-assigned identity from ``infra.json``.
    * Local: MSI step fails, chain falls back to Azure CLI / VS Code.
    """
    try:
        infra = get_infra_config()
        client_id = infra.managed_identity.client_id if infra.managed_identity else None
    except Exception:
        client_id = None

    if client_id:
        logger.debug(
            "[GuardChecker] DefaultAzureCredential with MSI client_id=%s...",
            client_id[:8],
        )
        return DefaultAzureCredential(managed_identity_client_id=client_id)

    logger.debug("[GuardChecker] DefaultAzureCredential (no MSI client_id)")
    return DefaultAzureCredential()


def _arm_get(
    credential: DefaultAzureCredential,
    path: str,
    api_version: str = _ARM_API_VERSION,
) -> Dict[str, Any]:
    """Perform a GET against ARM and return the JSON body.

    ``path`` must start with ``/subscriptions/...``.
    """
    token = credential.get_token(_ARM_SCOPE).token
    url = f"{_ARM_HOST}{path}"
    params = {"api-version": api_version}
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"ARM GET {path} failed: {resp.status_code} {resp.text[:300]}"
        )
    return resp.json()


def _arm_put(
    credential: DefaultAzureCredential,
    path: str,
    body: Dict[str, Any],
    api_version: str = _ARM_API_VERSION,
) -> Dict[str, Any]:
    """Perform a PUT against ARM and return the JSON body.

    ``path`` must start with ``/subscriptions/...``. ARM may answer with
    202 Accepted for long-running ops; for RAI policies the response is
    typically synchronous (200/201) so we only handle that here.
    """
    token = credential.get_token(_ARM_SCOPE).token
    url = f"{_ARM_HOST}{path}"
    params = {"api-version": api_version}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = requests.put(
        url, headers=headers, params=params,
        data=json.dumps(body), timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"ARM PUT {path} failed: {resp.status_code} {resp.text[:600]}"
        )
    if not resp.content:
        return {}
    return resp.json()


def _list_deployments(
    credential: DefaultAzureCredential,
    sub_id: str,
    rg: str,
    account: str,
) -> List[Dict[str, Any]]:
    path = (
        f"/subscriptions/{sub_id}/resourceGroups/{rg}"
        f"/providers/Microsoft.CognitiveServices/accounts/{account}/deployments"
    )
    body = _arm_get(credential, path)
    return body.get("value", []) or []


def _get_rai_policy(
    credential: DefaultAzureCredential,
    sub_id: str,
    rg: str,
    account: str,
    policy_name: str,
) -> Optional[Dict[str, Any]]:
    path = (
        f"/subscriptions/{sub_id}/resourceGroups/{rg}"
        f"/providers/Microsoft.CognitiveServices/accounts/{account}"
        f"/raiPolicies/{policy_name}"
    )
    try:
        return _arm_get(credential, path)
    except RuntimeError as exc:
        logger.warning("[GuardChecker] Could not fetch RAI policy %s: %s",
                       policy_name, exc)
        return None


def _normalise_filters(policy_body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pluck a flat filter list out of an RAI policy body."""
    if not policy_body:
        return []
    props = policy_body.get("properties") or {}
    raw = props.get("contentFilters") or []
    out: List[Dict[str, Any]] = []
    for f in raw:
        out.append({
            "name": f.get("name"),
            "source": f.get("source"),
            "enabled": f.get("enabled"),
            "blocking": f.get("blocking"),
            "severity_threshold": f.get("severityThreshold"),
        })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_content_filters(
    enabled: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Inspect content filter (RAI) policies on the configured AOAI account.

    Args:
        enabled: Tri-state override.

            * ``True``  — force run.
            * ``False`` — force skip; returns ``None``.
            * ``None``  — auto-detect via env vars (see module docstring).

    Returns:
        A report dict with shape::

            {
                "account": "aoai-orcas-deepwiki-kc",
                "subscription_id": "...",
                "resource_group": "...",
                "deployments": [
                    {
                        "name": "gpt-5.1-chat",
                        "model": "gpt-5.1",
                        "rai_policy": "Microsoft.DefaultV2",
                        "filters": [
                            {"name": "Hate", "source": "Prompt",
                             "enabled": True, "blocking": True,
                             "severity_threshold": "Medium"},
                            ...
                        ],
                    },
                    ...
                ],
            }

        or ``None`` if the checker is disabled.

    Raises:
        ClientAuthenticationError: AAD token acquisition failed.
        RuntimeError: ARM call failed (e.g. 403 missing RBAC).
        ValueError: ``infra.json`` is missing required fields.
    """
    if not is_guard_check_enabled(enabled):
        logger.info("[GuardChecker] Skipped (disabled)")
        return None

    infra = get_infra_config()
    account_cfg = get_account_config()
    sub_id = (account_cfg.subscription_id or "").strip()
    rg = (account_cfg.resource_group or "").strip()
    if not sub_id or not rg:
        raise ValueError(
            "infra.json: account.subscription_id and account.resource_group "
            "are required for content-filter inspection"
        )

    chat_endpoint = infra.azure_openai.chat.endpoint
    account = _account_name_from_endpoint(chat_endpoint)

    credential = _build_credential()
    try:
        deployments = _list_deployments(credential, sub_id, rg, account)
    except ClientAuthenticationError:
        logger.error("[GuardChecker] AAD authentication failed — "
                     "run 'az login' locally or verify MSI role assignment")
        raise

    # Cache policies so we don't re-fetch one per deployment.
    policy_cache: Dict[str, List[Dict[str, Any]]] = {}
    out_deployments: List[Dict[str, Any]] = []
    for d in deployments:
        props = d.get("properties") or {}
        model_props = props.get("model") or {}
        policy_name = props.get("raiPolicyName") or ""

        if policy_name and policy_name not in policy_cache:
            policy_body = _get_rai_policy(
                credential, sub_id, rg, account, policy_name
            )
            policy_cache[policy_name] = _normalise_filters(policy_body or {})

        out_deployments.append({
            "name": d.get("name"),
            "model": model_props.get("name"),
            "rai_policy": policy_name or None,
            "filters": policy_cache.get(policy_name, []),
        })

    report = {
        "account": account,
        "subscription_id": sub_id,
        "resource_group": rg,
        "deployments": out_deployments,
    }
    logger.info(
        "[GuardChecker] Inspected %d deployment(s) on account %s",
        len(out_deployments), account,
    )
    return report


def format_filter_report(report: Optional[Dict[str, Any]]) -> str:
    """Render :func:`check_content_filters` output as human-readable text."""
    if report is None:
        return "[GuardChecker] disabled"

    lines: List[str] = []
    lines.append(
        f"Account : {report['account']}  "
        f"(sub={report['subscription_id']}, rg={report['resource_group']})"
    )
    deployments = report.get("deployments") or []
    if not deployments:
        lines.append("  (no deployments found)")
        return "\n".join(lines)

    for d in deployments:
        policy = d.get("rai_policy") or "<account default>"
        lines.append("")
        lines.append(
            f"Deployment: {d['name']}  model={d.get('model')}  "
            f"rai_policy={policy}"
        )
        filters = d.get("filters") or []
        if not filters:
            lines.append("    (filter detail unavailable)")
            continue
        for f in filters:
            lines.append(
                f"    - {f.get('name'):<28} "
                f"source={f.get('source'):<11} "
                f"enabled={f.get('enabled')!s:<5} "
                f"blocking={f.get('blocking')!s:<5} "
                f"sev={f.get('severity_threshold')}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Setter
# ---------------------------------------------------------------------------

# Categories that Azure rejects unless the subscription has been granted
# the "Modified Content Filters" exemption. Used for a friendlier error
# message before we even hit ARM.
_GATED_CATEGORIES = {
    "hate", "sexual", "violence", "selfharm", "self-harm", "self_harm",
    "jailbreak", "indirect attack", "indirectattack",
}


def update_content_filter(
    policy_name: str,
    filter_name: str,
    source: str,
    *,
    new_enabled: Optional[bool] = None,
    new_blocking: Optional[bool] = None,
    new_severity: Optional[str] = None,
    confirm: bool = False,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Mutate a single content-filter entry on an existing RAI policy.

    The function fetches the current policy via GET, replaces the matching
    ``(filter_name, source)`` row with the supplied overrides (only the
    fields you pass are changed), then PUTs the full body back. Returns
    the updated policy body returned by ARM.

    Args:
        policy_name: Name of the RAI policy to modify
            (e.g. ``"CustomContentFilter403"``). Must already exist on
            the configured Azure OpenAI account.
        filter_name: Filter category, case-insensitive
            (e.g. ``"Profanity"``, ``"Hate"``, ``"Protected Material Code"``).
        source: ``"Prompt"`` or ``"Completion"`` \u2014 the side of the
            request the filter applies to.
        new_enabled: New value for ``enabled`` (or ``None`` to keep).
        new_blocking: New value for ``blocking`` (or ``None`` to keep).
            Note: ``blocking=True`` requires ``enabled=True``.
        new_severity: New severity threshold
            (``"Low"``/``"Medium"``/``"High"``) or ``None`` to keep.
        confirm: Must be ``True`` to actually apply the change. When
            ``False`` (default), the function performs a dry run and
            returns the prospective body without calling PUT.
        enabled: Same tri-state guard as :func:`check_content_filters`
            \u2014 ``False`` short-circuits and returns ``{}``.

    Returns:
        On a real run: the policy body returned by ARM after the PUT.
        On a dry run (``confirm=False``): a dict with key ``"dry_run"``
        set to ``True`` and the prospective body under ``"body"``.
        When the checker is disabled: an empty dict.

    Raises:
        ValueError: Filter not found on the policy, or no fields to change.
        RuntimeError: ARM call failed (e.g. gated category, missing RBAC).
    """
    if not is_guard_check_enabled(enabled):
        logger.info("[GuardChecker] update skipped (disabled)")
        return {}

    if (new_enabled is None and new_blocking is None
            and new_severity is None):
        raise ValueError(
            "update_content_filter: at least one of new_enabled, "
            "new_blocking, new_severity must be provided"
        )

    if source not in ("Prompt", "Completion"):
        raise ValueError(
            f"source must be 'Prompt' or 'Completion', got {source!r}"
        )

    infra = get_infra_config()
    account_cfg = get_account_config()
    sub_id = (account_cfg.subscription_id or "").strip()
    rg = (account_cfg.resource_group or "").strip()
    if not sub_id or not rg:
        raise ValueError(
            "infra.json: account.subscription_id and account.resource_group "
            "are required for content-filter updates"
        )

    account = _account_name_from_endpoint(infra.azure_openai.chat.endpoint)
    credential = _build_credential()

    path = (
        f"/subscriptions/{sub_id}/resourceGroups/{rg}"
        f"/providers/Microsoft.CognitiveServices/accounts/{account}"
        f"/raiPolicies/{policy_name}"
    )

    current = _arm_get(credential, path)
    props = dict(current.get("properties") or {})
    filters = list(props.get("contentFilters") or [])

    target_idx = -1
    for i, f in enumerate(filters):
        if (str(f.get("name", "")).lower() == filter_name.lower()
                and str(f.get("source", "")) == source):
            target_idx = i
            break
    if target_idx < 0:
        existing = ", ".join(
            f"{x.get('name')}/{x.get('source')}" for x in filters
        )
        raise ValueError(
            f"Filter '{filter_name}' (source={source}) not found on "
            f"policy '{policy_name}'. Existing entries: {existing}"
        )

    before = dict(filters[target_idx])
    after = dict(before)
    if new_enabled is not None:
        after["enabled"] = bool(new_enabled)
    if new_blocking is not None:
        after["blocking"] = bool(new_blocking)
    if new_severity is not None:
        after["severityThreshold"] = new_severity
    # blocking implies enabled per Azure validation.
    if after.get("blocking") and not after.get("enabled"):
        raise ValueError(
            "blocking=True requires enabled=True; pass new_enabled=True too"
        )

    filters[target_idx] = after
    props["contentFilters"] = filters

    # Pre-flight friendly warning for gated categories.
    is_gated = filter_name.strip().lower() in _GATED_CATEGORIES
    if is_gated and (new_enabled is False or new_blocking is False):
        logger.warning(
            "[GuardChecker] '%s' is a gated category. The PUT will fail "
            "unless this subscription has the 'Modified Content Filters' "
            "exemption.", filter_name,
        )

    new_body = {"properties": props}

    if not confirm:
        logger.info(
            "[GuardChecker] DRY RUN: would update %s/%s on policy %s: "
            "%s -> %s",
            filter_name, source, policy_name, before, after,
        )
        return {
            "dry_run": True,
            "path": path,
            "before": before,
            "after": after,
            "body": new_body,
        }

    logger.info(
        "[GuardChecker] APPLY: %s/%s on policy %s: %s -> %s",
        filter_name, source, policy_name, before, after,
    )
    return _arm_put(credential, path, new_body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_tri_bool(val: Optional[str]) -> Optional[bool]:
    """argparse ``type=`` helper: parse 'true'/'false'/'1'/'0' → bool."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    import argparse as _ap
    raise _ap.ArgumentTypeError(f"expected true/false, got {val!r}")


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="backend.utils.guard_checker",
        description=(
            "Inspect or update Azure OpenAI content-filter (RAI) policies "
            "for the resource configured in infra.json."
        ),
    )
    parser.add_argument("--policy", help="RAI policy name to update")
    parser.add_argument(
        "--filter", dest="filter_name",
        help="Filter category to update (e.g. Profanity)",
    )
    parser.add_argument(
        "--source", choices=["Prompt", "Completion"],
        help="Filter side: Prompt or Completion",
    )
    parser.add_argument(
        "--enabled", type=_parse_tri_bool, default=None,
        help="Set enabled true/false",
    )
    parser.add_argument(
        "--blocking", type=_parse_tri_bool, default=None,
        help="Set blocking true/false",
    )
    parser.add_argument(
        "--severity", choices=["Low", "Medium", "High"], default=None,
        help="Set severity threshold",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help=(
            "Apply the change. Without this flag, runs as a dry run "
            "and prints the prospective body."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    update_requested = any(
        x is not None for x in (args.enabled, args.blocking, args.severity)
    )
    if update_requested:
        if not (args.policy and args.filter_name and args.source):
            parser.error(
                "--policy, --filter and --source are required when "
                "updating a filter"
            )
        try:
            result = update_content_filter(
                policy_name=args.policy,
                filter_name=args.filter_name,
                source=args.source,
                new_enabled=args.enabled,
                new_blocking=args.blocking,
                new_severity=args.severity,
                confirm=args.confirm,
            )
        except Exception as exc:  # pragma: no cover - thin CLI wrapper
            logger.error("Guard update failed: %s", exc)
            return 1
        print(json.dumps(result, indent=2))
        if not args.confirm:
            print("\n(dry run — re-run with --confirm to apply)")
        return 0

    # Default: read-only inspection.
    try:
        report = check_content_filters()
    except Exception as exc:  # pragma: no cover - thin CLI wrapper
        logger.error("Guard check failed: %s", exc)
        return 1

    if report is None:
        print("[GuardChecker] disabled")
        return 0

    print(format_filter_report(report))
    print("")
    print("Raw JSON:")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
