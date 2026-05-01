"""
Graph builder — walks a repository and assembles a CodeMapData graph.

Uses the AST analyzer to extract per-file symbols and relationships,
then resolves cross-file references (imports, calls, inheritance).
"""

import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from backend.modules.codemap.analyzer import (
    LANGUAGE_DISPLAY,
    SUPPORTED_EXTENSIONS,
    analyze_file,
)
from backend.modules.codemap.models import (
    CodeMapData,
    CodeMapMetadata,
    SymbolEdge,
    SymbolNode,
)

logger = logging.getLogger(__name__)

# Directories always excluded (mirrors code_splitter / repo.json defaults)
_SKIP_DIRS = {
    '.git', '.svn', '.hg', 'node_modules', '__pycache__',
    '.venv', 'venv', 'env', 'dist', 'build', 'out', 'target',
    'coverage', '.tox', '.eggs', '.next', '.nuxt', 'bld',
    'bower_components', 'jspm_packages',
}

# Maximum number of symbols to extract via AST analysis.
# Beyond this, files still appear as file nodes but without
# internal symbols (functions, classes, methods).
# Prevents unbounded memory growth for very large repos.
MAX_SYMBOLS = 200_000

# Maximum file size eligible for AST extraction. Files above this threshold
# get a file-level node but no symbol/edge extraction. This caps the
# transient native memory footprint of tree-sitter (which can be 10–50× the
# source size) and prevents a single huge file (vendored bundles, generated
# parsers, minified output) from blowing the parallel-analysis peak past
# the AML node ceiling.
#
# Quality impact: hand-written source > 1 MB is rare; skipped files remain
# discoverable in the codemap (file node retained) and citable in the wiki.
# Wiki retrieval works on chunk vectors, not codemap edges, so it is
# unaffected.
MAX_AST_FILE_BYTES = 1_000_000


def build_codemap(
    repo_path: str,
    file_filter=None,
    max_workers: Optional[int] = None,
) -> CodeMapData:
    """Build a complete code map graph from a repository.

    Args:
        repo_path: Absolute path to the cloned repository root.
        file_filter: Optional FileFilter instance for inclusion/exclusion.
        max_workers: Thread pool size (defaults to os.cpu_count()).

    Returns:
        CodeMapData with nodes, edges, and metadata.
    """
    repo_path = repo_path.rstrip(os.sep)
    prefix_len = len(repo_path) + 1

    # Phase 1: Collect files
    files = _collect_files(repo_path, prefix_len, file_filter)
    logger.info(f"[CodeMap] Collected {len(files)} files for analysis")

    if not files:
        return CodeMapData(metadata=CodeMapMetadata())

    # Phase 2: Analyze files in parallel
    all_nodes: List[SymbolNode] = []
    all_edges: List[SymbolEdge] = []
    language_stats: Dict[str, int] = defaultdict(int)
    file_nodes: List[SymbolNode] = []
    symbol_budget_exhausted = False
    current_symbol_count = 0

    workers = max_workers or min(os.cpu_count() or 4, len(files))
    skipped_large_files: List[Tuple[str, int]] = []  # (rel_path, size)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for rel_path, ext, abs_path in files:
            futures[pool.submit(
                _analyze_one_file, abs_path, rel_path, ext
            )] = (rel_path, ext)

        for future in as_completed(futures):
            rel_path, ext = futures[future]
            try:
                nodes, edges, skipped_size = future.result()
            except Exception as e:
                logger.debug(f"[CodeMap] Error analyzing {rel_path}: {e}")
                nodes, edges, skipped_size = [], [], None

            if skipped_size is not None:
                skipped_large_files.append((rel_path, skipped_size))

            lang_name = LANGUAGE_DISPLAY.get(ext, ext)
            language_stats[lang_name] += 1

            # Create file-level node (always kept, even when AST was skipped)
            file_node = SymbolNode(
                id=rel_path,
                name=os.path.basename(rel_path),
                kind='file',
                file_path=rel_path,
                language=lang_name,
            )
            file_nodes.append(file_node)

            # Only accumulate symbols if budget allows
            if not symbol_budget_exhausted:
                current_symbol_count += len(nodes)
                all_nodes.extend(nodes)
                all_edges.extend(edges)

                if current_symbol_count >= MAX_SYMBOLS:
                    symbol_budget_exhausted = True
                    logger.warning(
                        f"[CodeMap] Symbol budget ({MAX_SYMBOLS}) reached "
                        f"at {len(file_nodes)} files, "
                        f"{current_symbol_count} symbols. "
                        f"Remaining files will be file-level only."
                    )

    if skipped_large_files:
        sample = [
            f"{p} ({sz // 1024} KB)"
            for p, sz in skipped_large_files[:5]
        ]
        more = "" if len(skipped_large_files) <= 5 else " ..."
        logger.info(
            f"[CodeMap] Skipped AST extraction for {len(skipped_large_files)} "
            f"file(s) > {MAX_AST_FILE_BYTES // 1024} KB "
            f"(file-level nodes retained): {sample}{more}"
        )

    all_nodes = file_nodes + all_nodes

    # Phase 3: Resolve cross-file references
    resolved_edges = _resolve_references(
        all_nodes, all_edges, repo_path,
    )

    # Free raw edges — no longer needed after resolution
    del all_edges

    # Phase 4: Deduplicate edges
    seen_edges: Set[Tuple[str, str, str]] = set()
    unique_edges: List[SymbolEdge] = []
    for edge in resolved_edges:
        key = (edge.source_id, edge.target_id, edge.kind)
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    # Free intermediate structures
    del resolved_edges
    del seen_edges

    # Phase 5: Compute importance scores for LOD
    _compute_importance_scores(all_nodes, unique_edges)

    # Build metadata
    commit_hash = _read_commit_hash(repo_path)
    symbol_count = sum(
        1 for n in all_nodes if n.kind != 'file'
    )
    metadata = CodeMapMetadata(
        commit_hash=commit_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_files=len(file_nodes),
        total_symbols=symbol_count,
        total_edges=len(unique_edges),
        language_stats=dict(language_stats),
    )

    logger.info(
        f"[CodeMap] Built graph: {len(all_nodes)} nodes, "
        f"{len(unique_edges)} edges, "
        f"{len(file_nodes)} files"
    )

    return CodeMapData(
        nodes=all_nodes,
        edges=unique_edges,
        metadata=metadata,
    )


# ============================================================================
# Internal helpers
# ============================================================================

def _collect_files(
    repo_path: str, prefix_len: int, file_filter,
) -> List[Tuple[str, str, str]]:
    """Walk the repo and collect files to analyze.

    Returns list of (relative_path, extension, absolute_path).
    """
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        # Filter directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in _SKIP_DIRS and not d.startswith('.')
        ]

        for fname in filenames:
            abs_path = os.path.join(root, fname)
            rel_path = abs_path[prefix_len:].replace('\\', '/')

            # Get extension
            _, dot_ext = os.path.splitext(fname)
            ext = dot_ext.lstrip('.').lower()

            # Only analyze files with supported extensions
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Apply file filter if provided
            if file_filter:
                if not file_filter.should_process_file(rel_path):
                    continue

            files.append((rel_path, ext, abs_path))

    return files


def _analyze_one_file(
    abs_path: str, rel_path: str, ext: str,
) -> Tuple[List[SymbolNode], List[SymbolEdge], Optional[int]]:
    """Read and analyze a single file. Thread-safe.

    Returns ``(nodes, edges, skipped_size)`` where ``skipped_size`` is the
    file size in bytes when the file was skipped due to ``MAX_AST_FILE_BYTES``,
    else ``None``. Skipped files still get a file-level node in the caller;
    only AST-derived symbol/edge extraction is bypassed.
    """
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        return [], [], None

    if size > MAX_AST_FILE_BYTES:
        # Caller adds the file-level node regardless; only AST work is skipped.
        return [], [], size

    try:
        with open(abs_path, 'rb') as f:
            content = f.read()
    except (OSError, IOError):
        return [], [], None

    nodes, edges = analyze_file(rel_path, content, ext)
    return nodes, edges, None


def _resolve_references(
    nodes: List[SymbolNode],
    edges: List[SymbolEdge],
    repo_path: str,
) -> List[SymbolEdge]:
    """Resolve unresolved edge targets to actual node IDs.

    Handles:
    - Import path → file node ID resolution
    - Function call name → function node ID resolution
    - Class name → class node ID resolution (for inheritance)
    """
    # Build lookup indices
    node_by_id = {n.id: n for n in nodes}

    # Symbol name → list of node IDs (for call/inheritance resolution)
    symbols_by_name: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        if n.kind != 'file':
            symbols_by_name[n.name].append(n.id)

    # Import module path → file path mapping
    file_path_set = {n.id for n in nodes if n.kind == 'file'}

    resolved: List[SymbolEdge] = []
    for edge in edges:
        if edge.kind == 'imports':
            resolved_target = _resolve_import(
                edge.target_id, file_path_set, repo_path,
            )
            if resolved_target:
                resolved.append(SymbolEdge(
                    source_id=edge.source_id,
                    target_id=resolved_target,
                    kind='imports',
                ))
            # else: external import, skip

        elif edge.kind == 'calls':
            # Target is a function/method name — resolve to node ID
            candidates = symbols_by_name.get(edge.target_id, [])
            if len(candidates) == 1:
                resolved.append(SymbolEdge(
                    source_id=edge.source_id,
                    target_id=candidates[0],
                    kind='calls',
                ))
            elif len(candidates) > 1:
                # Prefer same-file match
                source_node = node_by_id.get(edge.source_id)
                source_file = (
                    source_node.file_path if source_node else None
                )
                best = _pick_best_candidate(
                    candidates, source_file, node_by_id,
                )
                if best:
                    resolved.append(SymbolEdge(
                        source_id=edge.source_id,
                        target_id=best,
                        kind='calls',
                    ))
            # else: no match = external call, skip

        elif edge.kind in ('inherits', 'implements'):
            # Target is a class/interface name
            candidates = symbols_by_name.get(edge.target_id, [])
            if candidates:
                # Pick first class/interface match
                for cid in candidates:
                    cn = node_by_id.get(cid)
                    if cn and cn.kind == 'class':
                        resolved.append(SymbolEdge(
                            source_id=edge.source_id,
                            target_id=cid,
                            kind=edge.kind,
                        ))
                        break
                else:
                    resolved.append(SymbolEdge(
                        source_id=edge.source_id,
                        target_id=candidates[0],
                        kind=edge.kind,
                    ))

        else:
            resolved.append(edge)

    # Free lookup tables before returning
    del node_by_id
    del symbols_by_name
    del file_path_set

    return resolved


def _resolve_import(
    import_path: str,
    file_path_set: Set[str],
    repo_path: str,
) -> Optional[str]:
    """Resolve an import path to a file node ID.

    Tries multiple resolution strategies:
    - Python dotted path: backend.modules.wiki → backend/modules/wiki.py
      or backend/modules/wiki/__init__.py
    - JS/TS relative: ./utils/helper → utils/helper.ts (or .js, .tsx, etc.)
    - Java: com.example.MyClass → com/example/MyClass.java
    """
    # Already a file path?
    if import_path in file_path_set:
        return import_path

    # Python dotted module → file path
    as_path = import_path.replace('.', '/')
    for candidate in [
        f"{as_path}.py",
        f"{as_path}/__init__.py",
        f"{as_path}.java",
        f"{as_path}.go",
        f"{as_path}.cs",
    ]:
        if candidate in file_path_set:
            return candidate

    # JS/TS relative import (strip leading ./)
    clean = import_path.lstrip('./')
    for ext in ('', '.ts', '.tsx', '.js', '.jsx', '/index.ts',
                '/index.tsx', '/index.js', '/index.jsx'):
        candidate = f"{clean}{ext}"
        if candidate in file_path_set:
            return candidate

    return None


def _pick_best_candidate(
    candidates: List[str],
    source_file: Optional[str],
    node_by_id: Dict[str, SymbolNode],
) -> Optional[str]:
    """Pick the best matching candidate when multiple symbols share a name.

    Priority: same file > same directory > any match.
    """
    if not candidates:
        return None
    if not source_file:
        return candidates[0]

    source_dir = os.path.dirname(source_file)

    # Same file
    for cid in candidates:
        cn = node_by_id.get(cid)
        if cn and cn.file_path == source_file:
            return cid

    # Same directory
    for cid in candidates:
        cn = node_by_id.get(cid)
        if cn and os.path.dirname(cn.file_path) == source_dir:
            return cid

    return candidates[0]


def _read_commit_hash(repo_path: str) -> Optional[str]:
    """Read HEAD commit hash from .git directory."""
    head_file = os.path.join(repo_path, '.git', 'HEAD')
    try:
        with open(head_file, 'r') as f:
            content = f.read().strip()
        if content.startswith('ref:'):
            ref_path = content.split('ref: ', 1)[1].strip()
            ref_file = os.path.join(repo_path, '.git', ref_path)
            with open(ref_file, 'r') as f:
                return f.read().strip()
        return content
    except (OSError, IOError, IndexError):
        return None


def _compute_importance_scores(
    nodes: List[SymbolNode],
    edges: List[SymbolEdge],
) -> None:
    """Compute importance scores for all nodes in-place.

    Scoring formula (normalised 0.0–1.0):
        0.4 * indegree   — heavily referenced = core utility
        0.2 * outdegree   — many outgoing refs = orchestrator / entry point
        0.2 * kind_bonus  — structural weight by symbol kind
        0.2 * inheritance — base classes / interfaces with many subclasses

    Nodes with higher scores are shown first when the frontend applies
    level-of-detail (LOD) zoom filtering.
    """
    if not nodes:
        return

    # Count degrees
    indegree: Dict[str, int] = defaultdict(int)
    outdegree: Dict[str, int] = defaultdict(int)
    inherit_indegree: Dict[str, int] = defaultdict(int)

    for edge in edges:
        indegree[edge.target_id] += 1
        outdegree[edge.source_id] += 1
        if edge.kind in ('inherits', 'implements'):
            inherit_indegree[edge.target_id] += 1

    max_in = max(indegree.values()) if indegree else 1
    max_out = max(outdegree.values()) if outdegree else 1
    max_inh = max(inherit_indegree.values()) if inherit_indegree else 1

    kind_bonus: Dict[str, float] = {
        'class': 1.0,
        'file': 0.6,
        'function': 0.3,
        'method': 0.1,
        'module': 0.5,
    }

    for node in nodes:
        in_norm = indegree.get(node.id, 0) / max_in
        out_norm = outdegree.get(node.id, 0) / max_out
        k_bonus = kind_bonus.get(node.kind, 0.1)
        inh_norm = inherit_indegree.get(node.id, 0) / max_inh

        node.importance_score = round(
            0.4 * in_norm + 0.2 * out_norm + 0.2 * k_bonus + 0.2 * inh_norm,
            4,
        )
