"""
CodeMap processing utilities for wiki generation.

Provides functions that leverage codemap graph data during the wiki
generation pipeline:
- expand_file_paths: Expand page file paths using codemap edges
- summarize_codemap: Build a compact summary for LLM prompts
- build_file_edge_index / expand_file_paths_from_index: memory-efficient
  pre-built index that lets the bulky CodeMapData be released before
  per-page expansion runs.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Set

from backend.modules.codemap.models import CodeMapData

logger = logging.getLogger(__name__)


def expand_file_paths(
    file_paths: List[str],
    codemap: CodeMapData,
    max_extra: int = 10,
) -> List[str]:
    """Expand file_paths with files connected via codemap edges.

    Walks imports, calls, and inheritance edges from the declared files
    to discover closely related files that the wiki page should cover.
    Returns the original paths plus up to max_extra additional paths.

    Args:
        file_paths: Declared relevant file paths for a wiki page
        codemap: CodeMapData graph with nodes and edges
        max_extra: Maximum additional files to include

    Returns:
        Original paths + up to max_extra connected paths
    """
    if not codemap or not codemap.edges or not file_paths:
        return file_paths

    file_set: Set[str] = set(file_paths)

    # Build a quick lookup: node_id → file_path
    node_to_file: Dict[str, str] = {}
    for node in codemap.nodes:
        node_to_file[node.id] = node.file_path

    # Collect target files from edges where source is in our file set
    related_files: Dict[str, int] = {}  # file_path -> edge count
    for edge in codemap.edges:
        src_file = node_to_file.get(edge.source_id, '')
        tgt_file = node_to_file.get(edge.target_id, '')
        if src_file in file_set and tgt_file and tgt_file not in file_set:
            related_files[tgt_file] = related_files.get(tgt_file, 0) + 1
        elif tgt_file in file_set and src_file and src_file not in file_set:
            related_files[src_file] = related_files.get(src_file, 0) + 1

    # Sort by edge count (most connected first) and cap
    extra = sorted(
        related_files, key=lambda f: related_files[f], reverse=True
    )
    extra = extra[:max_extra]

    return file_paths + extra


def build_file_edge_index(
    codemap: Optional[CodeMapData],
) -> Dict[str, Counter]:
    """Build a compact ``file_path -> Counter(connected_file_path -> edge_count)``
    map from codemap edges.

    Bidirectional: every edge contributes to both endpoints' counters, mirroring
    the symmetric traversal in ``expand_file_paths``.

    Why this exists: ``expand_file_paths`` traverses every edge and rebuilds
    ``node_to_file`` on each call. With the ``codemap`` reference held through
    the entire wiki generation loop (one call per page), the full
    ``CodeMapData`` (~100-300 MB on a large repo) is alive for minutes.
    Pre-building this index up-front lets the caller drop the codemap
    reference before the page loop starts. The resulting index is typically
    1-2 orders of magnitude smaller than the codemap itself.

    Returns:
        Dict mapping file_path to a ``Counter`` of connected files. Empty
        ``Counter`` semantics are preserved: missing keys yield empty results
        in ``expand_file_paths_from_index``.
    """
    if not codemap or not codemap.edges:
        return {}

    node_to_file: Dict[str, str] = {n.id: n.file_path for n in codemap.nodes}
    index: Dict[str, Counter] = {}
    for edge in codemap.edges:
        sf = node_to_file.get(edge.source_id)
        tf = node_to_file.get(edge.target_id)
        if not sf or not tf or sf == tf:
            continue
        index.setdefault(sf, Counter())[tf] += 1
        index.setdefault(tf, Counter())[sf] += 1
    return index


def expand_file_paths_from_index(
    file_paths: List[str],
    edge_index: Dict[str, Counter],
    max_extra: int = 10,
) -> List[str]:
    """Memory-efficient counterpart of ``expand_file_paths`` that reads from
    a pre-built file-edge index instead of a full ``CodeMapData``.

    Behaviour mirrors ``expand_file_paths`` exactly:
        - Returns ``file_paths`` (verbatim, in the original order) followed by
          up to ``max_extra`` additional connected files, ranked by edge count
          (most-connected first).
        - When the index is empty or no related files exist, returns the
          original list unchanged.

    Tie-handling note: when two candidate files have the same edge count,
    ``Counter.most_common`` may return them in a different order than the
    original ``sorted(... reverse=True)``. Both downstream retrieval paths
    (FAISS and Azure AI Search) collapse the file-path list to a set before
    use, so order on ties is not observable. See ``test_expand_file_paths_parity``
    for the parity guarantee.
    """
    if not edge_index or not file_paths:
        return list(file_paths)

    file_set: Set[str] = set(file_paths)
    related: Counter = Counter()
    for fp in file_paths:
        for other, count in edge_index.get(fp, Counter()).items():
            if other not in file_set:
                related[other] += count

    extra = [f for f, _ in related.most_common(max_extra)]
    return list(file_paths) + extra


def summarize_codemap(codemap: CodeMapData, max_lines: int = 80) -> str:
    """Build a compact codemap summary for LLM prompt injection.

    Shows top-level modules, key classes/functions, and dependency edges.
    The output is an XML-tagged block suitable for embedding in prompts.

    Args:
        codemap: CodeMapData graph with nodes, edges, and metadata
        max_lines: Maximum output lines (truncates if exceeded)

    Returns:
        Compact summary string, or '' if codemap is empty
    """
    if not codemap or not codemap.nodes:
        return ''

    # Collect file-level nodes grouped by top directory
    modules: Dict[str, List[str]] = {}
    for node in codemap.nodes:
        if node.kind != 'file':
            continue
        parts = node.file_path.split('/')
        top_dir = parts[0] if len(parts) > 1 else '(root)'
        if top_dir not in modules:
            modules[top_dir] = []
        modules[top_dir].append(node.file_path)

    # Collect key symbols (classes and top-level functions)
    key_symbols: List[str] = []
    for node in codemap.nodes:
        if node.kind in ('class', 'function') and not node.parent_id:
            key_symbols.append(
                f"  {node.kind}: {node.name} ({node.file_path})"
            )
            if len(key_symbols) >= 40:
                break

    # Collect import/dependency edges between files
    file_deps: Dict[str, Set[str]] = {}
    for edge in codemap.edges:
        if edge.kind == 'imports':
            src = (edge.source_id.split('::')[0]
                   if '::' in edge.source_id else edge.source_id)
            tgt = (edge.target_id.split('::')[0]
                   if '::' in edge.target_id else edge.target_id)
            if src != tgt:
                if src not in file_deps:
                    file_deps[src] = set()
                file_deps[src].add(tgt)

    lines = ['<codemap_summary>']
    lines.append(
        f'Total: {codemap.metadata.total_files} files, '
        f'{codemap.metadata.total_symbols} symbols, '
        f'{codemap.metadata.total_edges} edges'
    )
    lines.append('')

    # Module overview
    lines.append('Modules:')
    for mod, files in sorted(modules.items()):
        lines.append(f'  {mod}/ ({len(files)} files)')
    lines.append('')

    # Key symbols
    if key_symbols:
        lines.append('Key symbols:')
        lines.extend(key_symbols[:30])
        lines.append('')

    # Top dependencies
    if file_deps:
        lines.append('Key dependencies:')
        dep_count = 0
        for src, targets in sorted(
            file_deps.items(), key=lambda x: -len(x[1])
        ):
            if dep_count >= 15:
                break
            lines.append(
                f'  {src} → {", ".join(sorted(targets)[:5])}'
            )
            dep_count += 1

    lines.append('</codemap_summary>')

    # Cap total length
    result = '\n'.join(lines)
    if len(result.splitlines()) > max_lines:
        result = '\n'.join(result.splitlines()[:max_lines])
        result += '\n</codemap_summary>'
    return result
