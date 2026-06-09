"""
Server-side wiki generation — replaces the frontend's 1200-line orchestration.

Generates a complete wiki for a repository:
1. Build file tree from cloned repo
2. Generate wiki structure via LLM
3. Generate each page via LLM + RAG retrieval
4. Assemble into WikiCacheData
"""

import gc
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from adalflow.core.types import ModelType
from openai import BadRequestError

from backend.config import get_azure_deployment_name
from backend.clients.azureai_client import AzureAIClient
from backend.logger import get_content_filter_logger
from backend.utils.guard_checker import is_content_filter_error
from backend.modules.chat.service import format_context_text, get_language_info
from backend.modules.codemap.models import CodeMapData
from backend.processor.codemap_generator import (
    summarize_codemap as _summarize_codemap,
    build_file_edge_index as _build_file_edge_index,
    expand_file_paths_from_index as _expand_file_paths_from_index,
)
from backend.promptstore.codemap import build_codemap_prompt_section
from backend.modules.wiki.models import (
    WikiCacheData, WikiPage, WikiSection,
    WikiStructureModel, RepoInfo,
)
from backend.promptstore.wiki_page import (
    build_wiki_page_prompt, build_wiki_page_review_prompt,
    format_page_catalog,
)
from backend.promptstore.wiki_structure import (
    build_wiki_structure_prompt as _build_structure_prompt,
    file_tree_dirs_only as _file_tree_dirs_only,
    LANGUAGE_DISPLAY_NAMES,
)
from backend.modules.wiki.xml_repair import close_open_tags
from backend.utils.sanitizer import sanitize_for_content_filter
from backend.utils.model_capabilities import is_reasoning_model

logger = logging.getLogger(__name__)


def build_file_tree(repo_path: str, max_depth: int = 6,
                    max_entries: int = 10000) -> str:
    """Build a file tree string from a cloned repo directory.

    Honours `excluded.json` (dirs + file patterns) and the supported-extension
    whitelist from `included.json` so the structure-gen LLM only sees files
    that are actually candidates for embedding. Without this, the LLM happily
    references Makefiles, .tmplt, .csproj, .props etc. that never get indexed,
    producing wiki pages with declared file paths the retriever can't resolve.

    For large repos (>max_entries entries), switches to directory-only
    output mid-walk to avoid building a massive string then discarding it.
    """
    lines = []
    repo_path = repo_path.rstrip(os.sep)
    prefix_len = len(repo_path) + 1
    entry_count = 0

    # Load filter config (single source of truth) and build a FileFilter
    # equivalent to the one document.py uses in exclusion mode.
    from backend.config import get_file_filters_config, get_included_config
    from backend.types.processor_types import FileFilter

    file_filters = get_file_filters_config()
    excluded_dirs_set = set(file_filters["excluded_dirs"])
    file_filter = FileFilter(
        excluded_dirs=excluded_dirs_set,
        excluded_patterns=set(file_filters["excluded_files"]),
    )
    included_cfg = get_included_config()
    all_ext_set = set(included_cfg.get("code", [])) | set(
        included_cfg.get("doc", [])
    )

    for root, dirs, files in os.walk(repo_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs_set
                   and not d.startswith('.')]
        rel = root[prefix_len:]
        depth = rel.count(os.sep) if rel else 0
        if depth >= max_depth:
            dirs.clear()
            continue
        # Normalise to forward slash so the file tree matches the path
        # convention the structure-gen LLM (and retriever index) expect.
        rel_posix = rel.replace(os.sep, '/') if rel else ''
        if rel_posix:
            lines.append(rel_posix + '/')
            entry_count += 1
        for f in sorted(files):
            if f.startswith('.'):
                continue
            rel_file = f"{rel_posix}/{f}" if rel_posix else f
            # Drop files excluded by pattern (*.csproj, *.tmplt, ...).
            if not file_filter.should_process_file(rel_file):
                continue
            # Drop files whose extension isn't supported for embedding.
            ext = os.path.splitext(f)[1].lower()
            if ext not in all_ext_set:
                continue
            lines.append(rel_file)
            entry_count += 1
            if entry_count > max_entries:
                # Bail early — convert to dirs-only from what's collected
                logger.info(
                    f"Large repo (>{max_entries} entries), switching "
                    f"to directory-only tree mid-walk"
                )
                return _file_tree_dirs_only('\n'.join(lines))

    return '\n'.join(lines)


def read_readme(repo_path: str) -> str:
    """Read README file from repo, trying common names."""
    for name in ['README.md', 'README.rst', 'README.txt', 'README', 'readme.md']:
        path = os.path.join(repo_path, name)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                # Truncate very long READMEs
                if len(content) > 15000:
                    content = content[:15000] + '\n\n[... truncated for length]'
                return content
            except Exception:
                pass
    return '(No README found)'


def _call_llm(
    prompt: str,
    model_client: AzureAIClient,
    deployment: str,
    *,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    max_completion_tokens: int = 16384,
    temperature: Optional[float] = None,
) -> Tuple[str, str]:
    """Make a direct (non-streaming) LLM call.

    Reasoning-family deployments (gpt-5, o1, o3, o4) take
    ``reasoning_effort`` + ``verbosity`` and reject ``temperature``;
    chat deployments take ``temperature``. The deployment is the source
    of truth -- callers pass the knobs they want and this branches.

    Returns:
        Tuple of (content, req_id) where req_id is the Azure OpenAI
        server request ID for support ticket correlation.
    """
    api_kwargs: Dict[str, object] = {
        'model': deployment,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_completion_tokens': max_completion_tokens,
    }
    if is_reasoning_model(deployment):
        if reasoning_effort is not None:
            api_kwargs['reasoning_effort'] = reasoning_effort
        if verbosity is not None:
            api_kwargs['verbosity'] = verbosity
    else:
        # Chat-family deployment -- temperature applies, reasoning knobs
        # do not. gpt-5.1-chat in our infra still forces temperature=1.0
        # server-side, so this is mostly a future-proofing branch.
        if temperature is not None:
            api_kwargs['temperature'] = temperature
    try:
        response = model_client.call(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )
    except BadRequestError as e:
        if is_content_filter_error(e):
            _log_content_filter_prompt(
                prompt=prompt,
                deployment=deployment,
                error=e,
            )
        raise
    req_id = getattr(response, '_request_id', 'unknown')
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content or '', req_id
    return '', req_id


# ---- content-filter diagnostic helpers ----


def _log_content_filter_prompt(
    prompt: str,
    deployment: str,
    error: Exception,
) -> None:
    """Dump the full prompt to the diagnostic log on content filter errors."""
    cf_logger = get_content_filter_logger()
    req_id = getattr(error, '_request_id', None)
    if req_id is None:
        # Try extracting from response headers
        try:
            resp = getattr(error, 'response', None)
            if resp and hasattr(resp, 'headers'):
                req_id = (resp.headers.get('apim-request-id')
                          or resp.headers.get('x-request-id')
                          or 'unknown')
        except Exception:
            req_id = 'unknown'
    cf_logger.error(
        "=== CONTENT FILTER PROMPT DUMP ===\n"
        "deployment=%s | req_id=%s | prompt_length=%d\n"
        "error=%s\n"
        "--- PROMPT START ---\n%s\n--- PROMPT END ---",
        deployment, req_id, len(prompt), error, prompt,
    )


def _parse_structure_xml(xml_text: str) -> Tuple[
    str, str, List[dict], List[dict], List[str]
]:
    """Parse LLM's XML structure response into components.

    Returns (title, description, pages, sections, root_section_ids).
    Replicates the frontend's DOMParser logic with Python xml.etree.
    """
    # Strip markdown fences
    xml_text = re.sub(r'^```(?:xml)?\s*', '', xml_text, flags=re.MULTILINE)
    xml_text = re.sub(r'```\s*$', '', xml_text, flags=re.MULTILINE)

    # Extract <wiki_structure> block
    match = re.search(
        r'<wiki_structure>[\s\S]*?</wiki_structure>', xml_text
    )
    if not match:
        # Try to repair truncated XML
        if '<wiki_structure>' in xml_text:
            xml_text = close_open_tags(xml_text)
            match = re.search(
                r'<wiki_structure>[\s\S]*?</wiki_structure>', xml_text
            )
    if not match:
        raise ValueError("No <wiki_structure> block found in LLM response")

    raw = match.group(0)
    # Strip control characters
    raw = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', raw)

    def _escape_bare_ampersands(s: str) -> str:
        # XML/HTML entity references: &name; or &#123; or &#xAB;
        # Replace any & that is NOT followed by such a reference.
        return re.sub(r'&(?!(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);)',
                      '&amp;', s)

    def _dump_failed_xml(payload: str, reason: str) -> None:
        try:
            log_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            path = os.path.join(
                log_dir, f'wiki_structure_failed_{ts}.xml',
            )
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(payload)
            logger.warning(
                "Saved failed wiki structure XML to %s (%s)",
                path, reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not save failed XML for inspection: %s", exc,
            )

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        logger.warning(f"XML parse error, attempting repair: {e}")
        repaired = close_open_tags(raw)
        repaired = _escape_bare_ampersands(repaired)
        try:
            root = ET.fromstring(repaired)
        except ET.ParseError as e2:
            _dump_failed_xml(raw, f"first parse: {e}; second parse: {e2}")
            raise

    title = (root.findtext('title') or 'Wiki').strip()
    description = (root.findtext('description') or '').strip()

    # Parse pages
    pages = []
    seen_ids = set()
    for page_el in root.iter('page'):
        pid = page_el.get('id', f'page-{len(pages)+1}')
        # Normalize dash IDs: "2-2" → "2.2"
        if re.match(r'^\d+(-\d+)+$', pid):
            pid = pid.replace('-', '.')
        # Deduplicate
        if pid in seen_ids:
            parts = pid.split('.')
            last = int(parts[-1]) if parts[-1].isdigit() else 0
            while pid in seen_ids:
                last += 1
                parts[-1] = str(last)
                pid = '.'.join(parts)
        seen_ids.add(pid)

        page_title = (page_el.findtext('title') or '').strip()
        page_description = (page_el.findtext('description') or '').strip()
        importance = (page_el.findtext('importance') or 'medium').strip()
        if importance not in ('high', 'medium', 'low'):
            importance = 'medium'

        file_paths = [
            fp.text.strip() for fp in page_el.iter('file_path')
            if fp.text and fp.text.strip()
        ]
        related = [
            r.text.strip() for r in page_el.iter('related')
            if r.text and r.text.strip()
        ]
        pages.append({
            'id': pid, 'title': page_title, 'content': '',
            'description': page_description,
            'filePaths': file_paths, 'importance': importance,
            'relatedPages': related,
        })

    # Parse sections (recursive)
    sections = []
    root_section_ids = []

    def _parse_section(section_el: ET.Element) -> dict:
        sid = section_el.get('id', f'section-{len(sections)+1}')
        stitle = ''
        title_el = section_el.find('title')
        if title_el is not None and title_el.text:
            stitle = title_el.text.strip()

        page_refs = []
        pages_container = section_el.find('pages')
        if pages_container is not None:
            for ref in pages_container.iter('page_ref'):
                if ref.text:
                    r = ref.text.strip()
                    if re.match(r'^\d+(-\d+)+$', r):
                        r = r.replace('-', '.')
                    page_refs.append(r)

        subsections = []
        sub_container = section_el.find('subsections')
        if sub_container is not None:
            for child in sub_container:
                if child.tag == 'section':
                    child_section = _parse_section(child)
                    subsections.append(child_section)

        section_data = {
            'id': sid, 'title': stitle, 'pages': page_refs,
            'subsections': subsections if subsections else None,
        }
        sections.append(section_data)
        return section_data

    sections_container = root.find('sections')
    if sections_container is not None:
        for section_el in sections_container:
            if section_el.tag == 'section':
                _parse_section(section_el)
                root_section_ids.append(section_el.get('id', ''))

    # Sync section titles with overview pages
    for s in sections:
        overview = next((p for p in pages if p['id'] == s['id']), None)
        if overview and overview['title'] and s['title'] != overview['title']:
            s['title'] = overview['title']

    return title, description, pages, sections, root_section_ids


def generate_wiki(
    repo_url: str,
    branch: str,
    repo_type: str,
    repo_path: str,
    retriever,
    commit_hash: str,
    language: str = 'en',
    comprehensive: bool = True,
    owner: str = '',
    repo: str = '',
    codemap: Optional[CodeMapData] = None,
    enable_review_pass: bool = True,
) -> WikiCacheData:
    """Generate a complete wiki for a repository.

    This replaces the frontend's WebSocket-based orchestration loop.

    Args:
        repo_url: Full repository URL
        branch: Branch name
        repo_type: 'azuredevops' (only supported type)
        repo_path: Local path to cloned repo
        retriever: RAG retriever with prepared FAISS index
        commit_hash: HEAD commit hash for citation URLs
        language: Language code (default 'en')
        comprehensive: True for 15-25 pages, False for 4-6
        owner: Repo owner (derived from URL if empty)
        repo: Repo name (derived from URL if empty)
        codemap: Optional CodeMapData for structure-aware generation
        enable_review_pass: If True, run a second LLM pass to verify
            factual accuracy and fix Mermaid diagrams (~50% more cost)

    Returns:
        WikiCacheData with structure + all generated pages
    """
    logger.info(f"Starting wiki generation for {owner}/{repo} ({branch})")

    _, language_name = get_language_info(language)

    # Step 1: Build file tree + read README
    logger.info(f"Building file tree for: {owner}/{repo} ({branch})")

    file_tree = build_file_tree(repo_path)
    readme = read_readme(repo_path)
    file_count = len(file_tree.splitlines())
    logger.info(f"File tree: {file_count} entries, README: {len(readme)} chars")

    # Step 2: Get LLM client
    from backend.config import get_azure_ai_client
    model_client = get_azure_ai_client(task='reasoning')
    deployment = get_azure_deployment_name(task='reasoning')

    # Step 3: Generate wiki structure via LLM
    logger.info("Generating wiki structure")

    # Build codemap summary for structure-aware page organization
    codemap_summary = _summarize_codemap(codemap) if codemap else ''
    additional_context = build_codemap_prompt_section(codemap_summary)
    if additional_context:
        logger.info(
            f"Codemap summary: {len(codemap_summary)} chars "
            f"injected into structure prompt"
        )

    # Build a compact file-edge index up-front so we can release the
    # bulky CodeMapData reference before the per-page loop starts.
    # Without this, the full codemap (~100–300 MB on large repos) stays
    # alive for the entire wiki generation window.
    edge_index = _build_file_edge_index(codemap)
    codemap = None  # Allow garbage collection of the full graph
    gc.collect()
    if edge_index:
        logger.info(
            f"Codemap edge index: {len(edge_index)} files "
            f"(full codemap released)"
        )

    readme_safe = sanitize_for_content_filter(readme)
    structure_prompt = _build_structure_prompt(
        file_tree=file_tree, readme=readme_safe,
        owner=owner, repo=repo,
        language=language, comprehensive=comprehensive,
        additional_context=additional_context,
    )

    structure_xml, structure_req_id = _call_llm(
        structure_prompt, model_client, deployment,
        reasoning_effort='high',
        verbosity='medium',
    )
    logger.info(
        f"Structure generated by {deployment} "
        f"(req_id={structure_req_id})"
    )

    # Content filter retry: if response is too short, retry with dir-only tree
    if len(structure_xml) < 200 or '<wiki_structure>' not in structure_xml:
        logger.warning(
            "Structure response too short or missing XML, "
            "retrying with directory-only file tree"
        )
        dir_tree = _file_tree_dirs_only(file_tree)
        structure_prompt = _build_structure_prompt(
            file_tree=dir_tree,
            readme='(README omitted for content safety)',
            owner=owner, repo=repo,
            language=language, comprehensive=comprehensive,
            additional_context=additional_context,
        )
        structure_xml, structure_req_id = _call_llm(
            structure_prompt, model_client, deployment,
            reasoning_effort='high',
            verbosity='medium',
        )
        logger.info(
            f"Structure retry generated by {deployment} "
            f"(req_id={structure_req_id})"
        )

    # Parse XML structure
    title, description, pages_data, sections_data, root_sections = (
        _parse_structure_xml(structure_xml)
    )
    # Release large strings — no longer needed after parsing
    del structure_prompt, structure_xml
    gc.collect()
    logger.info(
        f"Structure: title={title}, pages={len(pages_data)}, "
        f"sections={len(sections_data)}, root_sections={root_sections}"
    )

    # Step 3b: Validate pages have declared files that exist in repo
    # AND are actually indexed by the retriever (the structure-gen LLM
    # often references build files / templates that aren't embedded).
    indexed_files = set(getattr(retriever, '_file_path_index', {}).keys())
    _validated_pages = []
    total_declared = 0
    total_dropped_missing = 0
    total_dropped_unindexed = 0
    for page in pages_data:
        declared = page.get('filePaths', [])
        total_declared += len(declared)
        # Normalise to forward slash to match retriever index keys.
        normalised = [fp.replace('\\', '/') for fp in declared]
        on_disk = [
            fp for fp in normalised
            if os.path.isfile(os.path.join(repo_path, fp))
        ]
        total_dropped_missing += len(normalised) - len(on_disk)
        if indexed_files:
            valid_files = [fp for fp in on_disk if fp in indexed_files]
            total_dropped_unindexed += len(on_disk) - len(valid_files)
        else:
            # Cloud mode or empty index: skip the indexed-files check.
            valid_files = on_disk
        page['filePaths'] = valid_files
        _validated_pages.append(page)
    logger.info(
        f"Page file validation: {total_declared} declared, "
        f"dropped {total_dropped_missing} not-on-disk, "
        f"{total_dropped_unindexed} not-indexed"
    )
    # Log pages with no valid files (they'll rely on semantic search)
    no_file_pages = [
        p['id'] for p in _validated_pages if not p['filePaths']
    ]
    if no_file_pages:
        logger.info(
            f"Pages with no matching files (semantic-only): "
            f"{no_file_pages}"
        )
    pages_data = _validated_pages

    # Build page catalog for cross-page links
    page_catalog = format_page_catalog(
        [{'id': p['id'], 'title': p['title']} for p in pages_data]
    )

    # Build (page_id -> owning section title) lookup so each page prompt
    # can be told which section it belongs to. ``sections_data`` is
    # already flat (``_parse_section`` appends every node, root + nested)
    # so a single pass is enough.
    section_title_by_page: Dict[str, str] = {}
    for sec in sections_data:
        stitle = sec.get('title') or ''
        for pid in sec.get('pages') or []:
            # First-write wins -- a page can only sit in one section.
            section_title_by_page.setdefault(pid, stitle)

    # Step 4: Generate each page via LLM + retrieval
    logger.info(f"Generating {len(pages_data)} pages")
    generated_pages: Dict[str, WikiPage] = {}
    wiki_top_k = 40  # Same as current system

    for i, page_data in enumerate(pages_data, 1):
        page_id = page_data['id']
        page_title = page_data['title']
        page_file_paths = page_data.get('filePaths', [])

        # Expand file_paths with codemap-connected files (imports, calls).
        # Uses the pre-built edge index rather than the full CodeMapData,
        # which has already been released for memory.
        retrieval_file_paths = _expand_file_paths_from_index(
            page_file_paths, edge_index
        )

        # Build expanded query: title + description + related page titles
        # instead of bare title, for dramatically better retrieval.
        query_parts = [page_title]
        page_desc = page_data.get('description', '')
        if page_desc:
            query_parts.append(page_desc)
        # Add titles of related pages for cross-cutting context
        for rel_id in page_data.get('relatedPages', []):
            rel_page = next(
                (p for p in pages_data if p['id'] == rel_id), None
            )
            if rel_page:
                query_parts.append(rel_page['title'])
        expanded_query = ' — '.join(query_parts)

        # RAG retrieval with file-path priority
        try:
            retrieved_docs = retriever.call_with_file_filter(
                query=expanded_query,
                file_paths=retrieval_file_paths,
                top_k=wiki_top_k,
                language=language,
            )
        except Exception as e:
            logger.warning(f"Retrieval failed for page {page_id}: {e}")
            retrieved_docs = []

        # Format context
        context_text = format_context_text(
            retrieved_docs,
            repo_url=repo_url,
            commit_hash=commit_hash,
            repo_type=repo_type,
        )

        # Sanitize context to avoid content-filter triggers from
        # credentials, GUIDs, or internal URLs in source code.
        context_text = sanitize_for_content_filter(context_text)

        # Log context size for the page prompt (no truncation —
        # quality is critical; let the model handle its full context).
        logger.info(
            f"Page [{i}/{len(pages_data)}] {page_id}: {page_title} — "
            f"context={len(context_text)} chars, "
            f"files={len(page_file_paths)} declared"
        )

        # Build page prompt
        page_importance = page_data.get('importance', 'medium')
        related_page_titles = [
            rp['title']
            for rp in (
                next(
                    (p for p in pages_data if p['id'] == rid), None
                )
                for rid in page_data.get('relatedPages', [])
            )
            if rp is not None and rp.get('title')
        ]
        wiki_prompt = build_wiki_page_prompt(
            page_title=page_title,
            page_id=page_id,
            file_paths=page_file_paths,
            context_text=context_text,
            repo_url=repo_url,
            commit_hash=commit_hash,
            page_catalog=page_catalog,
            language_name=language_name,
            repo_type=repo_type,
            page_description=page_data.get('description', ''),
            page_importance=page_importance,
            section_title=section_title_by_page.get(page_id, ''),
            related_page_titles=related_page_titles,
        )

        # Generate page content. ``verbosity`` follows page importance --
        # high-importance pages get longer / more detailed output.
        page_verbosity = (
            'high' if page_importance == 'high'
            else 'low' if page_importance == 'low'
            else 'medium'
        )
        try:
            content, page_req_id = _call_llm(
                wiki_prompt, model_client, deployment,
                reasoning_effort='medium',
                verbosity=page_verbosity,
            )
        except Exception as e:
            logger.error(f"LLM call failed for page {page_id}: {e}")
            content = f"Error generating page: {e}"
            page_req_id = 'failed'

        # Optional review pass: verify accuracy and fix diagrams.
        # Verbosity matches the page so the reviewer can return the full
        # rewritten body. Using a lower verbosity here makes the reviewer
        # compress the page and trip the 50% retention guard, throwing
        # away the review on long high-importance pages.
        if enable_review_pass and not content.startswith('Error'):
            try:
                review_prompt = build_wiki_page_review_prompt(
                    generated_page=content,
                    context_text=context_text,
                    language_name=language_name,
                )
                reviewed, review_req_id = _call_llm(
                    review_prompt, model_client, deployment,
                    reasoning_effort='low',
                    verbosity=page_verbosity,
                )
                if reviewed and len(reviewed) > len(content) * 0.5:
                    content = reviewed
                    logger.info(
                        f"  Review pass applied for {page_id}"
                    )
                else:
                    logger.warning(
                        f"  Review pass returned short content for "
                        f"{page_id}, keeping original"
                    )
            except Exception as e:
                logger.warning(
                    f"  Review pass failed for {page_id}: {e}"
                )

        generated_pages[page_id] = WikiPage(
            id=page_id,
            title=page_title,
            content=content,
            filePaths=page_file_paths,
            importance=page_data.get('importance', 'medium'),
            relatedPages=page_data.get('relatedPages', []),
        )
        logger.info(
            f"Generated page -> "
            f"[{i}/{len(pages_data)}] {page_id}: {page_title} "
            f"({len(content)} chars, model={deployment}, "
            f"req_id={page_req_id})"
        )

    # Step 5: Assemble WikiCacheData
    wiki_pages_for_structure = [
        WikiPage(
            id=p['id'], title=p['title'], content='',
            filePaths=p.get('filePaths', []),
            importance=p.get('importance', 'medium'),
            relatedPages=p.get('relatedPages', []),
        )
        for p in pages_data
    ]

    wiki_sections = [
        WikiSection(
            id=s['id'], title=s['title'], pages=s.get('pages', []),
            subsections=s.get('subsections'),
        )
        for s in sections_data
    ]

    wiki_structure = WikiStructureModel(
        id='wiki',
        title=title,
        description=description,
        pages=wiki_pages_for_structure,
        sections=wiki_sections,
        rootSections=root_sections,
    )

    cache_data = WikiCacheData(
        wiki_structure=wiki_structure,
        generated_pages=generated_pages,
        repo=RepoInfo(
            owner=owner, repo=repo, type=repo_type,
            branch=branch, repoUrl=repo_url,
        ),
        provider='azure',
        model=deployment,
        comprehensive=comprehensive,
        is_partial=False,
        commit_hash=commit_hash,
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        f"Wiki generation complete: title={title}, "
        f"pages={len(generated_pages)}, "
        f"commit={commit_hash[:7] if commit_hash else 'N/A'}"
    )

    return cache_data
