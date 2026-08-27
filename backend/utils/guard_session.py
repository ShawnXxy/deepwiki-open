"""GuardSession — content-filter snapshot/relax/restore around a pipeline run.

A :class:`GuardSession` is a context manager owned by the processor's
``_process()`` body. Its three jobs (mirroring the four goals in
[content_filter_autorelax_plan.md](../processor/content_filter_autorelax_plan.md)):

1. ``__enter__``: snapshot the RAI policies bound to the deployments the
   pipeline will use, deduplicating by policy name. Register on a
   module-level :class:`contextvars.ContextVar` so the AzureAIClient
   retry decorator (Phase C) can find this session without parameter
   plumbing.
2. :meth:`relax`: temporarily flip one safe-to-toggle filter row off
   (called by the retry decorator after classifying a content_filter
   error). Sends ``If-Match: <etag>`` so concurrent edits surface as
   HTTP 412 instead of being silently overwritten.
3. ``__exit__``: restore each modified policy from its snapshot. Runs
   regardless of whether the wrapped block succeeds, raises, or is
   interrupted.

Failure isolation:

* If ``__enter__`` cannot read the policy (no RBAC, ARM 5xx, missing
  config), the session enters **degraded** mode: empty snapshots,
  :meth:`relax` returns ``False``, ``__exit__`` is a no-op. The
  pipeline proceeds unchanged. One WARNING line at entry; no further
  noise.
* :meth:`relax` failures are logged and recorded but never raise out
  of the session — the caller's original error must propagate from
  the AzureAIClient retry decorator.

Kill switches (evaluated in order, any one disables the feature):

1. ``GuardSession.from_infra(enabled=False)`` — programmatic.
2. ``DEEPWIKI_AUTO_RELAX_FILTERS=0`` — feature flag (snapshot still runs;
   :meth:`relax` becomes a no-op).
3. ``DEEPWIKI_GUARD_CHECKER_DISABLED=1`` — hard kill (no GET, no PUT,
   no contextvar registration).
4. ``_DEEPWIKI_INSIDE_DOCKER=1`` — set by ``code_processor --mode=docker``;
   feature auto-disabled to match :mod:`backend.utils.guard_checker`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from backend.utils.guard_checker import (
    _SAFE_TO_AUTO_DISABLE,
    _normalise_filters,
    get_policy_snapshot,
    is_guard_check_enabled,
    restore_policy_snapshot,
    update_content_filter,
)

logger = logging.getLogger(__name__)


# Module-level active-session pointer. Set by :meth:`__enter__`,
# cleared by :meth:`__exit__`. The AzureAIClient retry decorator
# reads this via :func:`get_active`.
_active_session: contextvars.ContextVar[Optional["GuardSession"]] = (
    contextvars.ContextVar("deepwiki_guard_session", default=None)
)


def get_active() -> Optional["GuardSession"]:
    """Return the currently-active GuardSession, or ``None``.

    Used by the AzureAIClient retry decorator (Phase C) to discover
    whether an auto-relax session is in scope without parameter
    plumbing through every LLM caller.
    """
    return _active_session.get()


def _is_auto_relax_enabled() -> bool:
    """``DEEPWIKI_AUTO_RELAX_FILTERS=0`` (or ``false``) disables relax PUTs."""
    val = os.environ.get("DEEPWIKI_AUTO_RELAX_FILTERS", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


def _is_inside_docker() -> bool:
    return os.environ.get("_DEEPWIKI_INSIDE_DOCKER", "").strip() == "1"


class GuardSession:
    """Context manager that owns the snapshot/relax/restore lifecycle.

    Construct via :meth:`from_infra` rather than calling this directly
    so the policy-name discovery (chat + reasoning deployments) is
    centralised. The bare constructor is used by tests that want to
    inject explicit policy names.

    Attributes:
        enabled: Effective auto-relax enable state. ``True`` means
            :meth:`relax` will attempt the PUT; ``False`` means
            :meth:`relax` is a no-op (snapshot/restore still run).
        mode: Pipeline mode (``'local'``/``'cloud'``/``'docker'``);
            informational, mirrored into the start log line.
        policy_names: Unique set of RAI policy names this session
            owns. Filled by :meth:`__enter__`.
    """

    def __init__(
        self,
        policy_names: Optional[List[str]] = None,
        *,
        enabled: bool = True,
        mode: str = "local",
    ) -> None:
        self.policy_names: List[str] = list(dict.fromkeys(policy_names or []))
        self.enabled: bool = bool(enabled)
        self.mode: str = mode
        # _snapshots[policy_name] = full snapshot dict from get_policy_snapshot
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        # _applied[policy_name] = list of (filter_name, source, before_dict)
        self._applied: Dict[str, List[Dict[str, Any]]] = {}
        self._token: Optional[contextvars.Token] = None
        self._degraded: bool = False
        # Tracks whether the auto-disable kill switch caused us to
        # short-circuit __enter__ entirely.
        self._skipped: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_infra(
        cls,
        *,
        mode: str = "local",
        enabled: Optional[bool] = None,
    ) -> "GuardSession":
        """Build a session from ``infra.json`` (chat + reasoning policies).

        Args:
            mode: Pipeline mode. Used for the start log line and to
                short-circuit when ``mode == 'docker'``.
            enabled: Tri-state override. ``None`` (default) honours
                env vars (:data:`DEEPWIKI_AUTO_RELAX_FILTERS` and
                :data:`DEEPWIKI_GUARD_CHECKER_DISABLED`). ``True``
                forces the feature on (still respects
                :data:`DEEPWIKI_GUARD_CHECKER_DISABLED`). ``False``
                forces it off — :meth:`relax` becomes a no-op,
                snapshot/restore are skipped.

        The classmethod resolves the bound RAI policy name for the
        chat and reasoning deployments and stashes them on
        :attr:`policy_names` so :meth:`__enter__` can snapshot each
        unique policy exactly once.
        """
        # Late imports to avoid pulling backend.config at module load.
        from backend.config import get_account_config, get_infra_config
        from backend.utils.guard_checker import (
            _account_name_from_endpoint,
            _build_credential,
            _list_deployments,
        )

        # Determine effective enabled flag.
        if enabled is False:
            sess = cls(policy_names=[], enabled=False, mode=mode)
            sess._skipped = True
            return sess

        if not is_guard_check_enabled(None):
            # DEEPWIKI_GUARD_CHECKER_DISABLED=1 or default-off in Docker.
            sess = cls(policy_names=[], enabled=False, mode=mode)
            sess._skipped = True
            return sess

        if not _is_auto_relax_enabled():
            # Feature flag off — still snapshot for read, but relax is no-op.
            effective_enabled = False
        else:
            effective_enabled = True if enabled is None else bool(enabled)

        try:
            infra = get_infra_config()
            account_cfg = get_account_config()
            sub_id = (account_cfg.subscription_id or "").strip()
            rg = (account_cfg.resource_group or "").strip()
            if not sub_id or not rg:
                raise ValueError(
                    "infra.json: account.subscription_id and "
                    "account.resource_group are required"
                )
            account = _account_name_from_endpoint(infra.azure_openai.chat.endpoint)
            credential = _build_credential()
            deployments = _list_deployments(credential, sub_id, rg, account)

            # Find the policies bound to the three LLM deployments. Embedding is
            # excluded by name match: we only care about the deployments
            # the pipeline will actually call for LLM completions.
            wanted_deployments = {
                infra.azure_openai.chat.deployment,
                infra.azure_openai.reasoning.deployment,
                infra.azure_openai.premium_reasoning.deployment,
            }
            policy_names: List[str] = []
            for d in deployments:
                if d.get("name") not in wanted_deployments:
                    continue
                pname = ((d.get("properties") or {})
                         .get("raiPolicyName") or "").strip()
                if pname and pname not in policy_names:
                    policy_names.append(pname)
        except Exception as exc:  # noqa: BLE001 — degraded mode catches everything
            logger.warning(
                "[GuardSession] could not resolve policy names from infra "
                "(%s: %s); session will run in degraded mode",
                type(exc).__name__, exc,
            )
            sess = cls(policy_names=[], enabled=effective_enabled, mode=mode)
            sess._degraded = True
            return sess

        return cls(policy_names=policy_names, enabled=effective_enabled, mode=mode)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "GuardSession":
        if self._skipped:
            logger.info(
                "[GuardSession] start: skipped (DEEPWIKI_GUARD_CHECKER_DISABLED "
                "or Docker mode); pipeline runs unchanged"
            )
            return self

        if _is_inside_docker():
            self.enabled = False
            self._skipped = True
            logger.info(
                "[GuardSession] start: skipped (running inside Docker container)"
            )
            return self

        if self._degraded:
            # from_infra already logged the WARNING.
            self._token = _active_session.set(self)
            return self

        # Snapshot each unique policy.
        for pname in self.policy_names:
            try:
                snap = get_policy_snapshot(pname)
                self._snapshots[pname] = snap
                self._applied.setdefault(pname, [])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[GuardSession] start: snapshot of %s failed (%s: %s); "
                    "this policy will not be auto-relaxed",
                    pname, type(exc).__name__, exc,
                )

        if not self._snapshots:
            self._degraded = True
            logger.warning(
                "[GuardSession] start: degraded (no policies snapshotted); "
                "pipeline runs unchanged"
            )
            self._token = _active_session.set(self)
            return self

        # Compose start log line. e.g.
        # [GuardSession] start: 1 policy (CustomContentFilter403, 16 filters);
        #   auto-relax=on, mode=local
        parts = []
        for pname, snap in self._snapshots.items():
            n_filters = len(_normalise_filters(snap))
            parts.append(f"{pname} ({n_filters} filters)")
        logger.info(
            "[GuardSession] start: %d policy (%s); auto-relax=%s, mode=%s",
            len(self._snapshots), ", ".join(parts),
            "on" if self.enabled else "off",
            self.mode,
        )

        self._token = _active_session.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if self._skipped or self._degraded:
                return

            ok: List[str] = []
            failed: List[str] = []
            for pname, applied in self._applied.items():
                if not applied:
                    # No relax PUT happened — restore is a no-op.
                    continue
                snap = self._snapshots.get(pname)
                if snap is None:
                    failed.append(f"{pname} (no snapshot)")
                    continue
                try:
                    restore_policy_snapshot(snap)
                    changed = ", ".join(
                        f"{a['filter']}/{a['source']}" for a in applied
                    )
                    logger.info(
                        "[GuardSession] restored %s -> reverted %d change(s) (%s)",
                        pname, len(applied), changed,
                    )
                    ok.append(pname)
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"{pname} ({type(exc).__name__}: {exc})")

            if failed:
                logger.error(
                    "[GuardSession] PARTIAL RESTORE: ok=%s failed=%s; "
                    "manual recovery: python -m backend.utils.guard_checker "
                    "--policy <name> --filter <name> --source <Prompt|Completion> "
                    "--enabled true --blocking true --confirm",
                    ok or "(none)", failed,
                )
        finally:
            if self._token is not None:
                _active_session.reset(self._token)
                self._token = None

    # ------------------------------------------------------------------
    # Public API consumed by AzureAIClient retry decorator
    # ------------------------------------------------------------------

    def snapshot_filters(self) -> List[Dict[str, Any]]:
        """Return a flat list of ``(name, source, ...)`` filter rows.

        Used by :func:`extract_content_filter_categories` so the
        Tier-2 fallback can enumerate custom blocklist names that
        actually exist on the snapshotted policy.

        When the session is degraded or has no snapshots, returns ``[]``.
        """
        out: List[Dict[str, Any]] = []
        for snap in self._snapshots.values():
            out.extend(_normalise_filters(snap))
        return out

    def relax(self, category: str, source: str) -> bool:
        """Disable one filter row and remember the change for restore.

        Returns ``True`` iff at least one PUT succeeded against a
        policy that contains this ``(category, source)``. Logs a
        WARNING on each successful relax and on each failure;
        never raises out of this method.

        Categories outside :data:`_SAFE_TO_AUTO_DISABLE` are rejected
        without contacting ARM (matches the gated-category guard in
        :func:`update_content_filter`). Custom blocklist names are
        accepted iff they appear in :meth:`snapshot_filters`.
        """
        if not self.enabled:
            return False
        if self._skipped or self._degraded:
            return False
        if not category or not source:
            return False

        # Allow built-in safe-to-toggle names plus any custom blocklist
        # name actually present on a snapshotted policy.
        allowed = set(_SAFE_TO_AUTO_DISABLE)
        for f in self.snapshot_filters():
            name = f.get("name")
            if name:
                allowed.add(name)

        # Match case-insensitively against the allowed set.
        category_lower = category.lower()
        canonical: Optional[str] = None
        for name in allowed:
            if name.lower() == category_lower:
                canonical = name
                break
        if canonical is None:
            logger.warning(
                "[GuardSession] candidate %s/%s is not in safe-to-auto-disable "
                "set; skipping",
                category, source,
            )
            return False

        any_ok = False
        for pname, snap in self._snapshots.items():
            # Only PUT against policies that actually contain this row.
            rows = _normalise_filters(snap)
            row = next(
                (r for r in rows
                 if str(r.get("name", "")).lower() == category_lower
                 and r.get("source") == source),
                None,
            )
            if row is None:
                continue
            before = dict(row)
            try:
                etag = (snap or {}).get("_etag")
                result = update_content_filter(
                    policy_name=pname,
                    filter_name=canonical,
                    source=source,
                    new_enabled=False,
                    new_blocking=False,
                    confirm=True,
                    if_match=etag if etag else "",
                )
                # Refresh the cached ETag so the restore PUT in
                # __exit__ sends the post-relax version. Without this
                # the restore would 412 (PreconditionFailed) and the
                # policy would be left in the relaxed state. See
                # update_content_filter docstring for the "_new_etag"
                # contract.
                if isinstance(result, dict):
                    new_etag = result.get("_new_etag")
                    if new_etag and isinstance(snap, dict):
                        snap["_etag"] = new_etag
                self._applied.setdefault(pname, []).append({
                    "filter": canonical,
                    "source": source,
                    "before": before,
                })
                logger.warning(
                    "[GuardSession] relaxed %s/%s on %s "
                    "(before: enabled=%s blocking=%s severity=%s)",
                    canonical, source, pname,
                    before.get("enabled"), before.get("blocking"),
                    before.get("severity_threshold"),
                )
                any_ok = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[GuardSession] relax %s/%s on %s failed (%s: %s)",
                    canonical, source, pname, type(exc).__name__, exc,
                )

        # Azure OpenAI's data-plane caches the resolved RAI policy for
        # roughly 30s — a retry fired immediately after a successful
        # ARM PUT can still hit the *pre-relax* cached policy and fail
        # for the same reason. Sleep briefly so the retry actually
        # exercises the relaxed state. Tunable via env, default 15s,
        # set to 0 to disable.
        if any_ok:
            try:
                delay = float(os.environ.get(
                    "DEEPWIKI_GUARD_PROPAGATION_DELAY_SECONDS", "15"
                ))
            except ValueError:
                delay = 15.0
            if delay > 0:
                logger.info(
                    "[GuardSession] sleeping %.1fs for policy propagation "
                    "before retry (DEEPWIKI_GUARD_PROPAGATION_DELAY_SECONDS=0 to skip)",
                    delay,
                )
                time.sleep(delay)

        return any_ok


# ---------------------------------------------------------------------------
# CLI: python -m backend.utils.guard_session --inspect
# ---------------------------------------------------------------------------

def _cli() -> int:
    """Print what a pipeline run would snapshot, without mutating anything.

    Useful sanity check before wiring ``code_processor._process`` to
    open a real session.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m backend.utils.guard_session",
        description=(
            "Inspect what GuardSession.from_infra() would snapshot for "
            "the next pipeline run. Read-only — does not mutate any "
            "policies."
        ),
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help=(
            "Print resolved policy names and filter rows that would be "
            "snapshotted. Default action when no other flag is given."
        ),
    )
    parser.add_argument(
        "--mode", default="local",
        choices=("local", "cloud", "docker"),
        help="Pipeline mode (default: local).",
    )
    args = parser.parse_args()

    sess = GuardSession.from_infra(mode=args.mode)

    print(f"GuardSession.from_infra(mode={args.mode!r})")
    print(f"  enabled (effective): {sess.enabled}")
    print(f"  degraded:            {sess._degraded}")
    print(f"  skipped:             {sess._skipped}")
    print(f"  policy_names:        {sess.policy_names}")

    if not sess.policy_names:
        print("  (no policies to snapshot — feature would be a no-op)")
        return 0

    print("")
    print("Performing read-only snapshot (no mutation)...")
    snapshots: Dict[str, Dict[str, Any]] = {}
    for pname in sess.policy_names:
        try:
            snapshots[pname] = get_policy_snapshot(pname)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pname}: snapshot failed ({type(exc).__name__}: {exc})")

    for pname, snap in snapshots.items():
        rows = _normalise_filters(snap)
        print("")
        print(f"  Policy: {pname}  (etag={snap.get('_etag')!r})")
        for r in rows:
            print(
                f"    - {r.get('name'):30s} source={r.get('source'):10s} "
                f"enabled={r.get('enabled')} blocking={r.get('blocking')} "
                f"sev={r.get('severity_threshold')}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
