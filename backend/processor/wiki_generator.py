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
from typing import Dict, List, Tuple

from adalflow.core.types import ModelType

from backend.config import get_azure_deployment_name
from backend.clients.azureai_client import AzureAIClient
from backend.modules.chat.service import format_context_text, get_language_info
from backend.modules.wiki.models import (
    WikiCacheData, WikiPage, WikiSection,
    WikiStructureModel, RepoInfo,
)
from backend.promptstore.wiki_page import (
    build_wiki_page_prompt, format_page_catalog,
)
from backend.promptstore.wiki_structure import (
    build_wiki_structure_prompt as _build_structure_prompt,
    file_tree_dirs_only as _file_tree_dirs_only,
    LANGUAGE_DISPLAY_NAMES,
)
from backend.modules.wiki.xml_repair import close_open_tags

logger = logging.getLogger(__name__)


def build_file_tree(repo_path: str, max_depth: int = 6,
                    max_entries: int = 10000) -> str:
    """Build a file tree string from a cloned repo directory.

    For large repos (>max_entries entries), switches to directory-only
    output mid-walk to avoid building a massive string then discarding it.
    """
    lines = []
    repo_path = repo_path.rstrip(os.sep)
    prefix_len = len(repo_path) + 1
    entry_count = 0

    # Load excluded directories from excluded.json (single source of truth)
    from backend.config import get_file_filters_config
    file_filters = get_file_filters_config()
    excluded_dirs_set = set(file_filters["excluded_dirs"])

    for root, dirs, files in os.walk(repo_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs_set
                   and not d.startswith('.')]
        rel = root[prefix_len:]
        depth = rel.count(os.sep) if rel else 0
        if depth >= max_depth:
            dirs.clear()
            continue
        if rel:
            lines.append(rel + '/')
            entry_count += 1
        for f in sorted(files):
            if not f.startswith('.'):
                lines.append(os.path.join(rel, f) if rel else f)
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


def _call_llm(prompt: str, model_client: AzureAIClient,
              deployment: str, temperature: float = 1.0,
              max_tokens: int = 16384) -> str:
    """Make a direct (non-streaming) LLM call and return the full text."""
    api_kwargs = {
        'model': deployment,
        'messages': [{'role': 'user', 'content': f'/no_think {prompt}'}],
        'temperature': temperature,
        'max_completion_tokens': max_tokens,
    }
    response = model_client.call(
        api_kwargs=api_kwargs, model_type=ModelType.LLM
    )
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content or ''
    return ''


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

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        logger.warning(f"XML parse error, attempting repair: {e}")
        raw = close_open_tags(raw)
        root = ET.fromstring(raw)

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

    Returns:
        WikiCacheData with structure + all generated pages
    """
    logger.info(f"Starting wiki generation for {owner}/{repo} ({branch})")

    _, language_name = get_language_info(language)

    # Step 1: Build file tree + read README
    logger.info(f"GENERATING WIKI: {owner}/{repo} ({branch})")

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

    structure_prompt = _build_structure_prompt(
        file_tree=file_tree, readme=readme,
        owner=owner, repo=repo,
        language=language, comprehensive=comprehensive,
    )

    structure_xml = _call_llm(structure_prompt, model_client, deployment)

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
        )
        structure_xml = _call_llm(structure_prompt, model_client, deployment)

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

    # Build page catalog for cross-page links
    page_catalog = format_page_catalog(
        [{'id': p['id'], 'title': p['title']} for p in pages_data]
    )

    # Step 4: Generate each page via LLM + retrieval
    logger.info(f"Generating {len(pages_data)} pages")
    generated_pages: Dict[str, WikiPage] = {}
    wiki_top_k = 40  # Same as current system

    for i, page_data in enumerate(pages_data, 1):
        page_id = page_data['id']
        page_title = page_data['title']
        page_file_paths = page_data.get('filePaths', [])

        # RAG retrieval with file-path priority
        try:
            retrieved_docs = retriever.call_with_file_filter(
                query=page_title,
                file_paths=page_file_paths,
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

        # Cap context to avoid oversized prompts and memory spikes.
        # Truncate at a clean file-path boundary so partial files
        # are not sent to the LLM.
        _MAX_CONTEXT_CHARS = 120000
        if len(context_text) > _MAX_CONTEXT_CHARS:
            truncated = context_text[:_MAX_CONTEXT_CHARS]
            # Find last complete file section boundary
            boundary = truncated.rfind('\n## File Path:')
            if boundary > 0:
                context_text = truncated[:boundary]
            else:
                context_text = truncated
            logger.info(
                f"Context truncated to {len(context_text)} chars "
                f"(was {len(context_text)})"
            )

        # Build page prompt
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
        )

        # Generate page content
        try:
            content = _call_llm(wiki_prompt, model_client, deployment)
        except Exception as e:
            logger.error(f"LLM call failed for page {page_id}: {e}")
            content = f"Error generating page: {e}"

        generated_pages[page_id] = WikiPage(
            id=page_id,
            title=page_title,
            content=content,
            filePaths=page_file_paths,
            importance=page_data.get('importance', 'medium'),
            relatedPages=page_data.get('relatedPages', []),
        )
        logger.info(
            f"[{i}/{len(pages_data)}] {page_id}: {page_title} "
            f"({len(content)} chars)"
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
