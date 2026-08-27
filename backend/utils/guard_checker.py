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

# NOTE: ``backend.config`` is imported lazily inside each function that
# needs it. Eager import would create a circular dependency at module
# load: ``backend.config`` runs ``get_configs_dict()`` at top level,
# which imports ``backend.clients.azureai_client``, which now imports
# this module for ``is_content_filter_error``.

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
        from backend.config import get_infra_config

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
    body, _etag = _arm_get_with_etag(credential, path, api_version)
    return body


def _arm_get_with_etag(
    credential: DefaultAzureCredential,
    path: str,
    api_version: str = _ARM_API_VERSION,
) -> tuple[Dict[str, Any], Optional[str]]:
    """Perform a GET against ARM and return ``(body, etag)``.

    The ETag is read from the HTTP ``ETag`` response header first and
    falls back to the ``etag`` field in the response body when the
    header is absent. Returns ``None`` when neither is available.
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
    body = resp.json() if resp.content else {}
    etag = resp.headers.get("ETag") or (body.get("etag") if isinstance(body, dict) else None)
    return body, etag


def _arm_put(
    credential: DefaultAzureCredential,
    path: str,
    body: Dict[str, Any],
    api_version: str = _ARM_API_VERSION,
    *,
    if_match: Optional[str] = None,
) -> tuple[Dict[str, Any], Optional[str]]:
    """Perform a PUT against ARM and return ``(body, new_etag)``.

    ``path`` must start with ``/subscriptions/...``. ARM may answer with
    202 Accepted for long-running ops; for RAI policies the response is
    typically synchronous (200/201) so we only handle that here.

    The returned ``new_etag`` is the resource version *after* the PUT,
    read from the ``ETag`` response header first and falling back to
    the ``etag`` field inside the response body — matching the pattern
    used by :func:`_arm_get_with_etag`. Callers that issue a follow-up
    PUT (e.g. snapshot restore after a relax) MUST refresh their cached
    ETag with this value, otherwise the second PUT will 412.

    Args:
        if_match: Optional ETag for optimistic concurrency control.
            When provided, sent as the ``If-Match`` HTTP header. ARM
            answers HTTP 412 (Precondition Failed) if the resource has
            been modified since the snapshot — that surface as a
            ``RuntimeError`` containing ``412`` so callers can detect
            a concurrent edit and back off.
    """
    token = credential.get_token(_ARM_SCOPE).token
    url = f"{_ARM_HOST}{path}"
    params = {"api-version": api_version}
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if if_match:
        headers["If-Match"] = if_match

    resp = requests.put(
        url, headers=headers, params=params,
        data=json.dumps(body), timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"ARM PUT {path} failed: {resp.status_code} {resp.text[:600]}"
        )
    if not resp.content:
        return {}, resp.headers.get("ETag")
    try:
        body_out = resp.json()
    except ValueError:
        body_out = {}
    new_etag = resp.headers.get("ETag") or (
        body_out.get("etag") if isinstance(body_out, dict) else None
    )
    return body_out, new_etag


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
                        "name": "gpt-5.6-luna",
                        "model": "gpt-5.6-luna",
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

    from backend.config import get_account_config, get_infra_config

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
# Content-filter error detection & classifier (consumed by GuardSession)
# ---------------------------------------------------------------------------

# Substrings that identify a BadRequestError as a content-filter trip.
# The list is intentionally small and case-insensitive to avoid drift —
# kept here so the AzureAIClient retry decorator and the wiki/codemap
# diagnostics dump share one source of truth.
_CONTENT_FILTER_KEYWORDS: frozenset = frozenset({
    "content_filter",
    "content management policy",
    "content filtering",
    "responsibleaipolicy",
})


def is_content_filter_error(exc: BaseException) -> bool:
    """Return True iff *exc* looks like an Azure content-filter trip.

    Matches the exception message against
    :data:`_CONTENT_FILTER_KEYWORDS` (case-insensitive). Safe to call
    on any exception type — non-string conversions are coerced via
    ``str(exc)``.
    """
    if exc is None:
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in _CONTENT_FILTER_KEYWORDS)


# Mapping from Azure's content_filter_result API keys to the display
# names accepted by ``update_content_filter`` (which matches the
# ``name`` field on each filter row in the policy body).
_API_KEY_TO_DISPLAY_NAME: Dict[str, str] = {
    "hate": "Hate",
    "sexual": "Sexual",
    "violence": "Violence",
    "self_harm": "Selfharm",
    "selfharm": "Selfharm",
    "jailbreak": "Jailbreak",
    "indirect_attack": "Indirect Attack",
    "profanity": "Profanity",
    "protected_material_text": "Protected Material Text",
    "protected_material_code": "Protected Material Code",
}

# Categories that the GuardSession is allowed to flip off automatically
# without operator review. Anything in :data:`_GATED_CATEGORIES` is
# excluded by construction (Azure will reject the PUT anyway), and the
# "core safety four" (Hate/Sexual/Violence/Selfharm) are deliberately
# kept on this allow-list-by-policy-only — meaning the session will
# never auto-relax them. Custom blocklists (any non-built-in name)
# are matched at runtime against the snapshotted policy.
_SAFE_TO_AUTO_DISABLE: frozenset = frozenset({
    "Profanity",
    "Protected Material Text",
    "Protected Material Code",
    "Indirect Attack Spotlighting",
})


# Tier-2 fallback ordering, used when the server's body is Shape B
# (no key reports ``filtered: true``) or the body is missing entirely.
# Profanity comes first because that is the empirical default trigger
# observed on this resource.
_TIER2_FALLBACK_NAMES: tuple = ("Profanity",)


# Confidence labels for classifier output.
_CONF_EXPLICIT = "explicit"   # body shows {"filtered": true} for this key
_CONF_FALLBACK = "fallback"   # Shape B / no body → priority guess
_CONF_BLOCKLIST = "blocklist"  # custom blocklist named on the policy


class FilterCandidate:
    """One classifier candidate returned by :func:`extract_content_filter_categories`.

    Attributes:
        name: Display name (matches ``name`` on a filter row, e.g.
            ``"Profanity"`` or a custom blocklist name).
        source: ``"Prompt"`` or ``"Completion"``.
        confidence: One of ``"explicit"``, ``"fallback"``, ``"blocklist"``.
    """
    __slots__ = ("name", "source", "confidence")

    def __init__(self, name: str, source: str, confidence: str) -> None:
        self.name = name
        self.source = source
        self.confidence = confidence

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"FilterCandidate(name={self.name!r}, source={self.source!r}, "
            f"confidence={self.confidence!r})"
        )


def _extract_error_body(exc: BaseException) -> Dict[str, Any]:
    """Best-effort extraction of the structured error body from *exc*.

    OpenAI's ``BadRequestError`` exposes ``.body`` as a dict. As a
    fallback, parse the first JSON object found in ``str(exc)``.
    Returns an empty dict when neither path yields a dict.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body

    s = str(exc)
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(s[start:i + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
    return {}


def extract_content_filter_categories(
    exc: BaseException,
    *,
    policy_filters: Optional[List[Dict[str, Any]]] = None,
) -> List[FilterCandidate]:
    """Classify a content-filter exception into ranked relax candidates.

    The classifier walks two tiers:

    * **Tier 1 (explicit).** If the error body contains
      ``content_filter_result`` with at least one key whose
      ``filtered`` value is truthy, emit those keys (mapped via
      :data:`_API_KEY_TO_DISPLAY_NAME`) as
      :attr:`FilterCandidate.confidence` ``"explicit"``.
    * **Tier 2 (fallback).** If the error code is ``content_filter``
      but Tier 1 yields nothing — the empirically common Shape B
      where every standard key reports ``safe`` — emit the names in
      :data:`_TIER2_FALLBACK_NAMES` (today: ``Profanity``) followed
      by any custom-blocklist names that appear in *policy_filters*.

    Args:
        exc: A ``BadRequestError`` (or any exception). Non-content-filter
            errors return an empty list.
        policy_filters: Optional list as returned by
            :func:`_normalise_filters` for the snapshotted policy.
            Used to enumerate custom-blocklist names for Tier 2.

    Returns:
        A list of :class:`FilterCandidate`, in attempt order. May be
        empty when *exc* is not a content-filter error.
    """
    if not is_content_filter_error(exc):
        return []

    body = _extract_error_body(exc)
    err = body.get("error") if isinstance(body, dict) else None
    inner = (err or {}).get("innererror") if isinstance(err, dict) else None
    cfr = (inner or {}).get("content_filter_result") if isinstance(inner, dict) else None

    out: List[FilterCandidate] = []

    # Tier 1 — explicit hits.
    if isinstance(cfr, dict):
        for api_key, info in cfr.items():
            if not isinstance(info, dict):
                continue
            if not info.get("filtered"):
                continue
            display = _API_KEY_TO_DISPLAY_NAME.get(api_key.lower())
            if not display:
                continue
            # Source is not in the body; emit Prompt first, then Completion.
            out.append(FilterCandidate(display, "Prompt", _CONF_EXPLICIT))
            out.append(FilterCandidate(display, "Completion", _CONF_EXPLICIT))

    # Tier 2 — fallback if nothing explicit. We deliberately keep the
    # fallback active even when the body shows a Purview backend error
    # (408/500/cert): the user has empirically confirmed that disabling
    # Profanity makes such requests succeed (the Purview detail in the
    # response is a co-symptom, not always the trigger).
    if not out:
        for name in _TIER2_FALLBACK_NAMES:
            out.append(FilterCandidate(name, "Prompt", _CONF_FALLBACK))
            out.append(FilterCandidate(name, "Completion", _CONF_FALLBACK))

    # Tier 2 cont. — custom blocklists named on the snapshotted policy.
    if policy_filters:
        builtin = {v.lower() for v in _API_KEY_TO_DISPLAY_NAME.values()} | {
            "indirect attack spotlighting"
        }
        seen = {(c.name, c.source) for c in out}
        for f in policy_filters:
            name = f.get("name") or ""
            src = f.get("source") or ""
            if not name or not src:
                continue
            if name.lower() in builtin:
                continue
            if (name, src) in seen:
                continue
            out.append(FilterCandidate(name, src, _CONF_BLOCKLIST))

    return out


# ---------------------------------------------------------------------------
# Snapshot / restore (consumed by GuardSession)
# ---------------------------------------------------------------------------


def _resolve_account_target() -> tuple[DefaultAzureCredential, str, str, str]:
    """Resolve credential, sub_id, rg, account_name from infra.json.

    Centralised so the snapshot/restore helpers don't repeat the
    boilerplate that already lives in :func:`check_content_filters`
    and :func:`update_content_filter`.
    """
    from backend.config import get_account_config, get_infra_config

    infra = get_infra_config()
    account_cfg = get_account_config()
    sub_id = (account_cfg.subscription_id or "").strip()
    rg = (account_cfg.resource_group or "").strip()
    if not sub_id or not rg:
        raise ValueError(
            "infra.json: account.subscription_id and account.resource_group "
            "are required"
        )
    account = _account_name_from_endpoint(infra.azure_openai.chat.endpoint)
    return _build_credential(), sub_id, rg, account


def get_policy_snapshot(policy_name: str) -> Dict[str, Any]:
    """Read the full RAI policy body and stamp it with the response ETag.

    The returned dict has the exact shape ARM gave us, plus a top-level
    ``"_etag"`` key (added by us, distinct from any ``"etag"`` field
    inside the body) so the restore call can pass ``If-Match`` without
    re-reading. Pass the entire return value to
    :func:`restore_policy_snapshot` unchanged.
    """
    credential, sub_id, rg, account = _resolve_account_target()
    path = (
        f"/subscriptions/{sub_id}/resourceGroups/{rg}"
        f"/providers/Microsoft.CognitiveServices/accounts/{account}"
        f"/raiPolicies/{policy_name}"
    )
    body, etag = _arm_get_with_etag(credential, path)
    snap = dict(body)
    snap["_etag"] = etag
    snap["_policy_name"] = policy_name
    snap["_arm_path"] = path
    return snap


def restore_policy_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Re-PUT the snapshot's ``properties`` with ``If-Match`` set.

    Args:
        snapshot: The dict returned by :func:`get_policy_snapshot`.

    Returns:
        ARM's PUT response body. On HTTP 412 (precondition failed)
        the underlying :func:`_arm_put` raises ``RuntimeError`` and
        this function re-raises it after logging an ERROR with the
        manual recovery hint.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a dict from get_policy_snapshot()")
    path = snapshot.get("_arm_path")
    policy_name = snapshot.get("_policy_name") or "<unknown>"
    if not path:
        raise ValueError(
            "snapshot is missing '_arm_path'; only pass values returned by "
            "get_policy_snapshot()"
        )
    credential, _sub, _rg, _acct = _resolve_account_target()
    body = {"properties": snapshot.get("properties") or {}}
    etag = snapshot.get("_etag")
    try:
        result_body, _new_etag = _arm_put(credential, path, body, if_match=etag)
        return result_body
    except RuntimeError as exc:
        if "412" in str(exc):
            logger.error(
                "[GuardChecker] restore of policy %s failed with 412 "
                "(precondition failed). Another writer modified the "
                "policy during this session. Manual recovery: "
                "python -m backend.utils.guard_checker --policy %s "
                "--filter <name> --source <Prompt|Completion> "
                "--enabled true --blocking true --confirm",
                policy_name, policy_name,
            )
        raise


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
    if_match: Optional[str] = None,
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
        source: ``"Prompt"`` or ``"Completion"`` — the side of the
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
            — ``False`` short-circuits and returns ``{}``.
        if_match: Optional explicit ETag for optimistic concurrency.
            When ``None`` (default), the function captures the ETag
            from its own GET and uses it on the PUT so concurrent
            edits surface as HTTP 412 instead of being silently
            overwritten. Pass an empty string to opt out and PUT
            unconditionally.

    Returns:
        On a real run: the policy body returned by ARM after the PUT,
        with the post-PUT ETag stamped under the synthetic key
        ``"_new_etag"`` so callers (e.g. :class:`GuardSession`) can
        refresh their cached ETag and avoid a 412 on a follow-up PUT.
        On a dry run (``confirm=False``): a dict with key ``"dry_run"``
        set to ``True`` and the prospective body under ``"body"``.
        When the checker is disabled: an empty dict.

    Raises:
        ValueError: Filter not found on the policy, or no fields to change.
        RuntimeError: ARM call failed (e.g. gated category, missing RBAC,
            412 precondition failed).
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

    from backend.config import get_account_config, get_infra_config

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

    current, current_etag = _arm_get_with_etag(credential, path)
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
    # Resolve the ETag to send: explicit caller value wins (including
    # empty string == opt-out); otherwise use the one we just GET'd.
    etag_to_send: Optional[str]
    if if_match is None:
        etag_to_send = current_etag
    elif if_match == "":
        etag_to_send = None
    else:
        etag_to_send = if_match
    result_body, new_etag = _arm_put(
        credential, path, new_body, if_match=etag_to_send,
    )
    if isinstance(result_body, dict) and new_etag:
        # Stamp the post-PUT ETag so callers can refresh their cached
        # snapshot without re-GETting (avoids 412 on the restore PUT
        # after a relax — see GuardSession.relax / __exit__).
        result_body["_new_etag"] = new_etag

    # Persistence verification: read the post-PUT row out of the body
    # ARM returned (no extra GET) and log it. Helpful when debugging
    # claims like "the relax PUT didn't actually take effect" — Portal
    # only shows the *current* state, but auto-relax restores the row
    # on session exit, so a user opening Portal after a failed run
    # sees the original values. This log line proves what was
    # persisted between the PUT and the restore.
    if isinstance(result_body, dict):
        verify_filters = (result_body.get("properties") or {}).get(
            "contentFilters") or []
        for f in verify_filters:
            if (str(f.get("name", "")).lower() == filter_name.lower()
                    and str(f.get("source", "")) == source):
                logger.info(
                    "[GuardChecker] VERIFIED: %s/%s on %s after PUT: "
                    "enabled=%s blocking=%s (etag=%s)",
                    filter_name, source, policy_name,
                    f.get("enabled"), f.get("blocking"),
                    new_etag,
                )
                break

    return result_body


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
    parser.add_argument(
        "--show-blocklist-content", action="store_true",
        help=(
            "On read-only inspection, also dump the raw policy JSON "
            "(which can include custom-blocklist regex). Off by "
            "default to avoid leaking blocklist content into shell "
            "history or logs."
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
    if args.show_blocklist_content:
        print("")
        print("Raw JSON:")
        print(json.dumps(report, indent=2))
    else:
        print("")
        print(
            "(raw JSON suppressed; pass --show-blocklist-content "
            "to dump the full policy body)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
