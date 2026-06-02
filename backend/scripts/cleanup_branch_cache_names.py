"""One-time cleanup for legacy wiki/codemap cache filenames.

Background
----------
Wiki and codemap cache filenames embed a sanitised branch suffix. The
canonical form replaces every character outside ``[a-zA-Z0-9_-]`` with a
dash (see :func:`backend.utils.filter.sanitize_branch_for_path`), so a
branch like ``8.0`` becomes ``8-0``.

Older caches — and caches written by the Next.js routes before they were
fixed to sanitise the branch — kept the raw branch (e.g. ``8.0``), producing
duplicate files for the same wiki/codemap:

    deepwiki_cache_github_mysql_mysql-server_en_comprehensive_8.0-master.json   (legacy)
    deepwiki_cache_github_mysql_mysql-server_en_comprehensive_8-0-master.json   (canonical)

This script reconciles them to the canonical (dash) form for both local
storage (``~/.adalflow/wikicache`` and ``~/.adalflow/codemap``) and Azure
Blob Storage (``wikicache/`` and ``codemap/`` prefixes):

* If a canonical counterpart already exists, the legacy (non-canonical)
  file is a duplicate and is **deleted**.
* Otherwise the legacy file is **renamed** to its canonical name.

Safety
------
The script is **dry-run by default** and only reports what it would do.
Pass ``--apply`` to actually rename/delete. It is idempotent: a second run
after ``--apply`` finds nothing to do.

Usage
-----
    python -m backend.scripts.cleanup_branch_cache_names            # dry run
    python -m backend.scripts.cleanup_branch_cache_names --apply    # mutate
    python -m backend.scripts.cleanup_branch_cache_names --domain wiki --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from backend.clients.blob_client import (
    get_blob_storage_client,
    is_blob_storage_configured,
)
from backend.paths import get_codemap_path, get_wikicache_path

logger = logging.getLogger(__name__)

# A stem is already canonical when it only contains characters that
# sanitize_branch_for_path() never rewrites.
_CANONICAL_STEM = re.compile(r"^[A-Za-z0-9_-]+$")


def _canonical_stem(stem: str) -> str:
    """Return the canonical stem produced by dash-sanitisation.

    Mirrors :func:`backend.utils.filter.sanitize_branch_for_path` applied to
    the whole stem. This is safe because every fixed field in a cache
    filename (type, language, mode) already contains only canonical
    characters; only the branch (and, in rare cases, owner/repo) can carry a
    dot or other special character.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "-", stem)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized


@dataclass
class Plan:
    """A single reconciliation action."""

    action: str  # "delete" (duplicate) or "rename"
    src: str
    dst: Optional[str] = None


def _build_plans(names: list[str]) -> list[Plan]:
    """Compute reconciliation actions for a flat list of cache filenames."""
    existing = set(names)
    plans: list[Plan] = []

    for name in names:
        if not name.endswith(".json"):
            continue
        stem = name[: -len(".json")]
        if _CANONICAL_STEM.match(stem):
            continue  # already canonical

        canonical = f"{_canonical_stem(stem)}.json"
        if canonical == name:
            continue  # nothing to change (shouldn't happen given the regex)

        if canonical in existing:
            plans.append(Plan(action="delete", src=name))
        else:
            plans.append(Plan(action="rename", src=name, dst=canonical))

    return plans


def _reconcile_local(directory: str, apply: bool) -> int:
    if not os.path.isdir(directory):
        logger.info("Local dir not found, skipping: %s", directory)
        return 0

    names = [
        n for n in os.listdir(directory)
        if n.endswith(".json") and os.path.isfile(os.path.join(directory, n))
    ]
    plans = _build_plans(names)

    for plan in plans:
        src_path = os.path.join(directory, plan.src)
        if plan.action == "delete":
            logger.info("[local] duplicate -> delete: %s", plan.src)
            if apply:
                os.remove(src_path)
        else:  # rename
            dst_path = os.path.join(directory, plan.dst)  # type: ignore[arg-type]
            logger.info("[local] rename: %s -> %s", plan.src, plan.dst)
            if apply:
                os.replace(src_path, dst_path)

    return len(plans)


def _reconcile_blob(prefix: str, apply: bool) -> int:
    blob_client = get_blob_storage_client()
    if not blob_client:
        logger.warning("Blob storage configured but client unavailable; skipping %s", prefix)
        return 0

    blob_prefix = f"{prefix}/"
    blob_names = blob_client.list_blobs(prefix=blob_prefix)
    # Strip the prefix so _build_plans works on bare filenames.
    bare = [n[len(blob_prefix):] for n in blob_names if n.startswith(blob_prefix)]
    plans = _build_plans(bare)

    for plan in plans:
        src_blob = f"{blob_prefix}{plan.src}"
        if plan.action == "delete":
            logger.info("[blob] duplicate -> delete: %s", src_blob)
            if apply:
                blob_client.delete(src_blob)
        else:  # rename (copy to canonical, then delete original)
            dst_blob = f"{blob_prefix}{plan.dst}"
            logger.info("[blob] rename: %s -> %s", src_blob, dst_blob)
            if apply:
                content = blob_client.download_text(src_blob)
                if content is None:
                    logger.error("[blob] could not read %s; leaving in place", src_blob)
                    continue
                if blob_client.upload_text(dst_blob, content):
                    blob_client.delete(src_blob)
                else:
                    logger.error("[blob] failed to write %s; leaving original", dst_blob)

    return len(plans)


def _reconcile_domain(
    label: str,
    local_dir: str,
    blob_prefix: str,
    apply: bool,
) -> int:
    logger.info("=== %s ===", label)
    if is_blob_storage_configured():
        return _reconcile_blob(blob_prefix, apply)
    return _reconcile_local(local_dir, apply)


_DOMAINS: dict[str, Callable[[bool], int]] = {
    "wiki": lambda apply: _reconcile_domain(
        "wiki cache", get_wikicache_path(), "wikicache", apply
    ),
    "codemap": lambda apply: _reconcile_domain(
        "codemap cache", get_codemap_path(), "codemap", apply
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename/delete files. Without this flag the script only reports (dry run).",
    )
    parser.add_argument(
        "--domain",
        choices=["wiki", "codemap", "all"],
        default="all",
        help="Which cache domain to reconcile (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.apply:
        logger.info("DRY RUN — no files will be changed. Re-run with --apply to mutate.\n")

    domains = ["wiki", "codemap"] if args.domain == "all" else [args.domain]
    total = 0
    for name in domains:
        total += _DOMAINS[name](args.apply)

    verb = "applied" if args.apply else "would apply"
    logger.info("\nDone. %d action(s) %s.", total, verb)


if __name__ == "__main__":
    main()
