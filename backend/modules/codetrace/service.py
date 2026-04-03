"""
CodeTrace service — orchestrates RAG retrieval + LLM call for code tracing.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from adalflow.core.types import ModelType

from backend.config import get_azure_ai_client, get_azure_deployment_name
from backend.modules.chat.service import (
    format_context_text,
    get_language_info,
)
from backend.modules.codetrace.models import (
    CodeReference,
    CodeTraceResult,
    CodeTraceSection,
)
from backend.promptstore.code_trace import (
    CODE_TRACE_SYSTEM_PROMPT,
    CODE_TRACE_USER_PROMPT,
)

logger = logging.getLogger(__name__)


def generate_code_trace(
    question: str,
    rag,
    repo_url: str,
    repo_type: str = "azuredevops",
    repo_name: str = "",
    commit_hash: str = "",
    language: str = "en",
) -> CodeTraceResult:
    """Generate a code trace for a given question using RAG + LLM.

    Args:
        question: User's natural language question.
        rag: RAG instance with prepared retriever.
        repo_url: Repository URL.
        repo_type: Repository type.
        repo_name: Display name of the repository.
        commit_hash: HEAD commit hash for citation URLs.
        language: Language code for response.

    Returns:
        CodeTraceResult with structured trace sections.
    """
    _, language_name = get_language_info(language)

    # Step 1: Retrieve relevant code via RAG
    logger.info(f"[CodeTrace] Retrieving code for: {question[:80]}")
    retrieved = rag.call(question, language=language)

    # Step 2: Format context from retrieved documents
    context = format_context_text(
        retrieved,
        repo_url=repo_url,
        commit_hash=commit_hash,
        repo_type=repo_type,
    )

    if not context.strip():
        return CodeTraceResult(
            query=question,
            title="No relevant code found",
            sections=[CodeTraceSection(
                id="1",
                title="No Results",
                details="Could not find relevant code for this question. "
                        "Try rephrasing or asking about specific files.",
            )],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # Step 3: Build prompts
    system_prompt = CODE_TRACE_SYSTEM_PROMPT.format(
        repo_type=repo_type,
        repo_url=repo_url,
        repo_name=repo_name,
        language_name=language_name,
    )

    user_prompt = CODE_TRACE_USER_PROMPT.format(
        question=question,
        context=context[:100000],  # Cap context at 100KB
    )

    # Step 4: Call LLM
    logger.info("[CodeTrace] Calling LLM for code trace generation")
    model_client = get_azure_ai_client(task='reasoning')
    deployment = get_azure_deployment_name(task='reasoning')

    api_kwargs = {
        'model': deployment,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'/no_think {user_prompt}'},
        ],
        'temperature': 0.7,
        'max_completion_tokens': 16384,
    }

    response = model_client.call(
        api_kwargs=api_kwargs, model_type=ModelType.LLM,
    )

    if not hasattr(response, 'choices') or not response.choices:
        logger.error("[CodeTrace] LLM returned empty response")
        return CodeTraceResult(
            query=question,
            title="Generation failed",
            sections=[],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    raw_text = response.choices[0].message.content or ''

    # Step 5: Parse XML response
    result = _parse_trace_xml(raw_text, question)

    # Step 6: Extract source files and source contents from RAG docs
    source_files = set()
    for section in result.sections:
        for ref in section.code_refs:
            source_files.add(ref.file_path)
    result.source_files = sorted(source_files)

    # Build source_contents from retrieved documents
    result.source_contents = _extract_source_contents(retrieved)

    result.generated_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"[CodeTrace] Generated {len(result.sections)} sections, "
        f"{len(result.source_files)} source files"
    )
    return result


def _extract_source_contents(retrieved) -> dict:
    """Extract source code chunks grouped by file from RAG results.

    Returns dict of file_path -> list of chunk dicts with
    start_line, end_line, content, language.
    """
    contents: dict = {}
    try:
        docs = retrieved[0].documents if retrieved else []
    except (IndexError, AttributeError):
        return contents

    for doc in docs:
        meta = doc.meta_data or {}
        file_path = meta.get('file_path', '')
        if not file_path:
            continue

        # Extract raw code from enriched text
        header_len = meta.get('_header_len', 0)
        raw_text = doc.text[header_len:] if header_len else doc.text

        start_line = meta.get('start_line', 0)
        end_line = meta.get('end_line', 0)
        ext = meta.get('type', '')

        # Language mapping for syntax highlighting
        lang_map = {
            'py': 'python', 'js': 'javascript', 'ts': 'typescript',
            'jsx': 'javascript', 'tsx': 'typescript',
            'java': 'java', 'go': 'go', 'cs': 'csharp',
            'cpp': 'cpp', 'c': 'c', 'h': 'c',
            'rb': 'ruby', 'rs': 'rust', 'php': 'php',
            'sh': 'bash', 'yaml': 'yaml', 'yml': 'yaml',
            'json': 'json', 'xml': 'xml', 'sql': 'sql',
            'md': 'markdown', 'html': 'html', 'css': 'css',
        }
        lang = lang_map.get(ext, ext)

        if file_path not in contents:
            contents[file_path] = []

        contents[file_path].append({
            'file_path': file_path,
            'start_line': start_line + 1,  # Convert 0-based to 1-based
            'end_line': end_line + 1,
            'content': raw_text.strip(),
            'language': lang,
        })

    # Sort chunks within each file by start_line
    for fp in contents:
        contents[fp].sort(key=lambda c: c['start_line'])

    return contents


def _parse_trace_xml(
    raw_text: str, question: str,
) -> CodeTraceResult:
    """Parse the LLM's XML response into a CodeTraceResult."""
    # Strip markdown fences
    raw_text = re.sub(r'^```(?:xml)?\s*', '', raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r'```\s*$', '', raw_text, flags=re.MULTILINE)

    # Extract <code_trace> block
    match = re.search(
        r'<code_trace>[\s\S]*?</code_trace>', raw_text,
    )
    if not match:
        # Fallback: return raw text as a single section
        logger.warning("[CodeTrace] No <code_trace> XML found, using fallback")
        return CodeTraceResult(
            query=question,
            title="Code Trace",
            sections=[CodeTraceSection(
                id="1",
                title="Analysis",
                details=raw_text.strip(),
            )],
        )

    xml_text = match.group(0)
    # Strip control characters
    xml_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_text)

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"[CodeTrace] XML parse error: {e}")
        return CodeTraceResult(
            query=question,
            title="Code Trace",
            sections=[CodeTraceSection(
                id="1",
                title="Analysis",
                details=raw_text.strip(),
            )],
        )

    title = (root.findtext('title') or 'Code Trace').strip()

    sections = []
    for sec_el in root.findall('section'):
        sec_id = sec_el.get('id', str(len(sections) + 1))
        sec_title = (sec_el.findtext('title') or '').strip()
        motivation = (sec_el.findtext('motivation') or '').strip()
        details = (sec_el.findtext('details') or '').strip()

        code_refs = []
        for ref_el in sec_el.findall('code_ref'):
            ref_id = ref_el.get('id', '')
            fp = (ref_el.findtext('file_path') or '').strip()
            try:
                sl = int(ref_el.findtext('start_line') or '0')
                el = int(ref_el.findtext('end_line') or '0')
            except ValueError:
                sl, el = 0, 0
            annotation = (ref_el.findtext('annotation') or '').strip()
            snippet = (ref_el.findtext('snippet') or '').strip()
            if fp:
                code_refs.append(CodeReference(
                    ref_id=ref_id, file_path=fp,
                    start_line=sl, end_line=el,
                    snippet=snippet, annotation=annotation,
                ))

        connections = []
        for conn_el in sec_el.findall('connects_to'):
            if conn_el.text:
                connections.append(conn_el.text.strip())

        sections.append(CodeTraceSection(
            id=sec_id, title=sec_title,
            motivation=motivation, details=details,
            code_refs=code_refs, connections=connections,
        ))

    return CodeTraceResult(
        query=question, title=title, sections=sections,
    )
