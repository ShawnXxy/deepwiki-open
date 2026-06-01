"""
Delta detector for incremental embedding.

Decides which source files in a freshly-walked file list need to be re-embedded
versus reused from a previous embedding manifest. The unit of change is the
*source file* (not chunk), because the code splitter is whole-file aware:
splitting a modified file may shift every chunk index, so a changed file must
have all of its chunks atomically rewritten.

Decision order (first match wins):
    1. Cold start              — no previous manifest available.
    2. Manifest schema bump    — stored ``version`` differs from current.
    3. Embedder change         — deployment / model_name / vector_dim differ.
    4. Fast path               — commit hashes match AND working tree clean.
    5. Git diff path           — use ``git diff --name-status`` + ``git status``
                                  to compute the change set.
    6. SHA-256 verification    — always wins over (4) and (5): if a file's
                                  content hash differs from the manifest,
                                  re-embed even if git says nothing changed.

The detector is *conservative*: when any signal is ambiguous, it falls back to
full reprocess by returning every input file in ``to_embed``.

This module has no side effects. The caller is responsible for actually
embedding ``to_embed``, deleting chunks for ``to_delete``, and writing an
updated manifest at the end of the run.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Bumping this invalidates every existing manifest and forces full reprocess.
CURRENT_MANIFEST_VERSION = 1


@dataclass
class DeltaResult:
    """Outcome of comparing a current file list against a previous manifest."""

    to_embed: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    """Subset of ``file_infos`` (new + changed) that must be re-embedded."""

    to_delete: Set[str] = field(default_factory=set)
    """Relative paths whose chunks must be removed (deleted or replaced)."""

    unchanged: Set[str] = field(default_factory=set)
    """Relative paths that can be reused as-is from the previous run."""

    fast_path_hit: bool = False
    """True when commit hashes match and the working tree is clean."""

    reason: str = ""
    """Short tag explaining the chosen path (logged at INFO)."""

    file_hashes: Dict[str, str] = field(default_factory=dict)
    """Newly computed sha256 hashes for ``to_embed`` files (kept so the
    caller can use them to build the next manifest without re-reading)."""


def _sha256_of_file(path: str) -> Optional[str]:
    """Return the sha256 of a file, or ``None`` on I/O failure."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.debug(f"[Delta] sha256 failed for {path}: {e}")
        return None


def _embedder_signature(current_embedder: Dict[str, object]) -> Tuple[str, str, int]:
    """Normalised tuple used for embedder-equality checks."""
    return (
        str(current_embedder.get("deployment", "")),
        str(current_embedder.get("model_name", "")),
        int(current_embedder.get("vector_dim") or 0),
    )


def _full_reprocess(
    file_infos: List[Tuple[str, str, str, bool]],
    reason: str,
) -> DeltaResult:
    """Mark every file for re-embedding.

    Populates ``file_hashes`` for every input file so the manifest written
    after a cold-start / embedder-change / schema-change run carries valid
    sha256 values — without them the *next* run cannot do hash-based delta
    detection and would degrade to "always reprocess".
    """
    file_hashes: Dict[str, str] = {}
    for fi in file_infos:
        full_path, rel = fi[0], fi[1].replace("\\", "/")
        h = _sha256_of_file(full_path)
        if h is not None:
            file_hashes[rel] = h
    return DeltaResult(
        to_embed=list(file_infos),
        to_delete=set(),
        unchanged=set(),
        fast_path_hit=False,
        reason=reason,
        file_hashes=file_hashes,
    )


def compute_file_delta(
    repo_dir: str,
    file_infos: List[Tuple[str, str, str, bool]],
    prev_manifest: Optional[Dict[str, object]],
    current_embedder: Dict[str, object],
    current_commit_hash: str,
    verify_hash: bool = True,
) -> DeltaResult:
    """Compute the embedding delta against a previous manifest.

    Args:
        repo_dir: Absolute path to the cloned repository (used to resolve
            ``full_path`` for sha256 verification).
        file_infos: ``(full_path, relative_path, ext, is_code)`` tuples as
            produced by ``transform_documents_and_save_as_json``. The
            ``relative_path`` must use forward slashes — the caller
            normalises this.
        prev_manifest: Parsed previous manifest dict, or ``None`` for cold
            start.
        current_embedder: ``{"deployment": str, "model_name": str,
            "vector_dim": int}`` describing the embedder that will be used
            for this run.
        current_commit_hash: HEAD commit hash of the current working tree.
            Empty string if unknown (forces full reprocess).
        verify_hash: When True, sha256 of each file is computed and compared
            against the manifest entry. A mismatch always forces re-embedding
            of that file, even when git diff says it is unchanged.

    Returns:
        ``DeltaResult`` describing the planned action.
    """
    # 1) Cold start
    if not prev_manifest:
        return _full_reprocess(file_infos, "cold_start")

    # 2) Schema bump
    if int(prev_manifest.get("version", 0)) != CURRENT_MANIFEST_VERSION:
        return _full_reprocess(file_infos, "manifest_schema_changed")

    # 3) Embedder change
    prev_embedder = prev_manifest.get("embedder") or {}
    if _embedder_signature(prev_embedder) != _embedder_signature(current_embedder):
        return _full_reprocess(file_infos, "embedder_changed")

    prev_files: Dict[str, Dict[str, object]] = (
        prev_manifest.get("files") or {}
    )  # type: ignore[assignment]
    prev_commit = str(prev_manifest.get("commit_hash") or "")

    # Normalise input list for set ops.
    current_paths: Set[str] = set()
    by_rel: Dict[str, Tuple[str, str, str, bool]] = {}
    for fi in file_infos:
        _, rel, _, _ = fi
        rel = rel.replace("\\", "/")
        current_paths.add(rel)
        by_rel[rel] = fi

    prev_paths: Set[str] = set(prev_files.keys())

    # Try the git fast path: commit equal AND working tree clean. The caller
    # passes the already-computed change set in via ``prev_manifest`` flow,
    # but we need a separate git probe here. To keep this module decoupled
    # we ask git_ops on demand.
    from backend.modules.repository.git_ops import get_changed_files

    fast_path_hit = False
    git_changes: Dict[str, str] = {}
    if current_commit_hash and prev_commit and current_commit_hash == prev_commit:
        # Commit unchanged — check working tree by asking for changes vs HEAD.
        # An empty result here means the tree is clean.
        try:
            import subprocess
            res = subprocess.run(
                ["git", "status", "--porcelain", "-z"],
                capture_output=True, text=True, cwd=repo_dir, check=False
            )
            if res.returncode == 0 and not res.stdout.strip():
                fast_path_hit = True
        except Exception as e:
            logger.debug(f"[Delta] working-tree probe failed: {e}")

    if fast_path_hit:
        # Even on the fast path we still honour file-set drift — e.g. a file
        # was added to disk between runs but is not yet tracked.
        new_paths = current_paths - prev_paths
        removed_paths = prev_paths - current_paths

        # Optional hash verification: catches the "same commit but file was
        # touched out-of-band" case. Cheap because we only hash unchanged
        # files when explicitly enabled.
        changed_by_hash: Set[str] = set()
        file_hashes: Dict[str, str] = {}
        if verify_hash:
            for rel in (current_paths & prev_paths):
                full_path, _, _, _ = by_rel[rel]
                h = _sha256_of_file(full_path)
                if h is None:
                    # Couldn't hash — be conservative, re-embed.
                    changed_by_hash.add(rel)
                    continue
                prev_h = str(prev_files.get(rel, {}).get("sha256") or "")
                if prev_h and prev_h != h:
                    changed_by_hash.add(rel)
                file_hashes[rel] = h

        to_embed_rels = new_paths | changed_by_hash
        to_delete = removed_paths | changed_by_hash
        unchanged = (current_paths & prev_paths) - to_embed_rels

        return DeltaResult(
            to_embed=[by_rel[r] for r in to_embed_rels if r in by_rel],
            to_delete=to_delete,
            unchanged=unchanged,
            fast_path_hit=True,
            reason="fast_path_commit_match",
            file_hashes=file_hashes,
        )

    # 5) Git diff path
    if current_commit_hash and prev_commit:
        try:
            git_changes = get_changed_files(repo_dir, prev_commit)
        except Exception as e:
            logger.warning(f"[Delta] git diff failed, falling back: {e}")
            return _full_reprocess(file_infos, "git_diff_failed")
    else:
        # No commit hash on either side — we cannot compute a precise delta.
        return _full_reprocess(file_infos, "missing_commit_hash")

    if not git_changes and current_paths == prev_paths:
        # Commit changed but diff returned nothing AND file list matches the
        # manifest exactly. This shouldn't happen in practice, but treat it
        # as "nothing to do" via the fast-path-equivalent branch.
        return DeltaResult(
            to_embed=[],
            to_delete=set(),
            unchanged=current_paths & prev_paths,
            fast_path_hit=False,
            reason="diff_empty_paths_match",
        )

    # Translate git status letters into our planned actions.
    deleted_in_git: Set[str] = {p for p, s in git_changes.items() if s == 'D'}
    changed_in_git: Set[str] = {
        p for p, s in git_changes.items()
        if s in ('A', 'M', 'R', 'C', 'T')
    }

    # Source-of-truth set adjustments:
    # - Anything currently on disk that git considers added/modified/renamed
    #   must be embedded — but only if it survives the inclusion filter
    #   (i.e. present in ``by_rel``).
    new_paths = current_paths - prev_paths
    removed_paths = (prev_paths - current_paths) | (deleted_in_git & prev_paths)

    to_embed_rels: Set[str] = new_paths.copy()
    for rel in changed_in_git:
        if rel in by_rel:
            to_embed_rels.add(rel)

    # 6) SHA verification — always wins over signals above for files that
    # claim to be unchanged.
    file_hashes: Dict[str, str] = {}
    if verify_hash:
        candidate_unchanged = (current_paths & prev_paths) - to_embed_rels
        for rel in candidate_unchanged:
            full_path, _, _, _ = by_rel[rel]
            h = _sha256_of_file(full_path)
            if h is None:
                to_embed_rels.add(rel)
                continue
            prev_h = str(prev_files.get(rel, {}).get("sha256") or "")
            if prev_h and prev_h != h:
                to_embed_rels.add(rel)
            else:
                file_hashes[rel] = h

    # Also hash everything we are about to embed (caller will use these
    # to populate the new manifest without re-reading the files).
    for rel in to_embed_rels:
        if rel in file_hashes:
            continue
        full_path, _, _, _ = by_rel.get(rel, (None, None, None, None))
        if not full_path:
            continue
        h = _sha256_of_file(full_path)
        if h is not None:
            file_hashes[rel] = h

    # ``to_delete`` includes:
    #   - paths gone from disk
    #   - paths the user explicitly removed via git
    #   - paths whose content changed (their old chunks must be cleared
    #     before the new ones are written, in case chunk counts shrink)
    to_delete = removed_paths | (to_embed_rels & prev_paths)
    unchanged = (current_paths & prev_paths) - to_embed_rels

    return DeltaResult(
        to_embed=[by_rel[r] for r in to_embed_rels if r in by_rel],
        to_delete=to_delete,
        unchanged=unchanged,
        fast_path_hit=False,
        reason="git_diff",
        file_hashes=file_hashes,
    )


def summarize_for_log(result: DeltaResult) -> str:
    """One-line summary suitable for the ``[Vec] Delta summary:`` log."""
    return (
        f"embed={len(result.to_embed)} "
        f"unchanged={len(result.unchanged)} "
        f"delete={len(result.to_delete)} "
        f"fast_path={'yes' if result.fast_path_hit else 'no'} "
        f"reason={result.reason}"
    )
