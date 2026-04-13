"""
Code-aware text splitter for enhanced embedding quality.

Splits code files at logical boundaries (functions, classes, blocks)
rather than arbitrary token positions. Enriches chunks with structural
context for improved embedding retrieval.

Inspired by the processor design approach:
- Logical boundary detection (no mid-function splits)
- Structural metadata extraction (functions, classes, imports)
- Context-enriched embedding text (file path, language, section type)

Key differences from the full processor design:
- No LLM calls (zero additional cost)
- Uses regex-based boundary detection (language-agnostic)
- Prepends structural context prefix to embedding text
"""

import re
import logging
from typing import List, Dict, Any

from adalflow.core.types import Document
from backend.modules.embedder.tokenizer import count_tokens

logger = logging.getLogger(__name__)

# ============================================================================
# Boundary Detection Patterns (language-agnostic)
# ============================================================================

# Patterns that indicate the start of a new logical code block
BOUNDARY_PATTERNS = [
    # Python
    r'^(?:async\s+)?def\s+\w+',
    r'^class\s+\w+',
    # JavaScript / TypeScript
    r'^(?:export\s+)?(?:async\s+)?function\s+\w+',
    r'^(?:export\s+)?(?:default\s+)?class\s+\w+',
    r'^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(',
    r'^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*\(.*?\)\s*=>',
    # Java / C#
    (r'^\s*(?:public|private|protected|internal|static|abstract|'
     r'final|override|virtual)\s+(?:static\s+)?(?:async\s+)?'
     r'(?:class|interface|enum|struct|record)\s+\w+'),
    (r'^\s*(?:public|private|protected|internal|static|abstract|'
     r'final|override|virtual)\s+(?:static\s+)?(?:async\s+)?'
     r'\w+(?:<.*?>)?\s+\w+\s*\('),
    # Go
    r'^func\s+(?:\(\s*\w+\s+\*?\w+\)\s+)?\w+',
    r'^type\s+\w+\s+(?:struct|interface)',
    # Rust
    r'^(?:pub\s+)?(?:async\s+)?fn\s+\w+',
    r'^(?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+',
    # Ruby
    r'^(?:def|class|module)\s+\w+',
    # PHP
    r'^\s*(?:public|private|protected|static)?\s*function\s+\w+',
]

COMPILED_BOUNDARIES = [re.compile(p, re.MULTILINE) for p in BOUNDARY_PATTERNS]

# Language type descriptions for embedding context
LANGUAGE_MAP = {
    'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript',
    'java': 'Java', 'go': 'Go', 'rs': 'Rust', 'cpp': 'C++',
    'c': 'C', 'cs': 'C#', 'rb': 'Ruby', 'php': 'PHP',
    'swift': 'Swift', 'kt': 'Kotlin', 'jsx': 'React JSX',
    'tsx': 'React TSX', 'html': 'HTML', 'css': 'CSS',
    'md': 'Markdown', 'json': 'JSON', 'yaml': 'YAML',
    'yml': 'YAML', 'sql': 'SQL', 'sh': 'Shell', 'ps1': 'PowerShell',
    'h': 'C/C++ Header', 'hpp': 'C++ Header',
}


# ============================================================================
# Boundary Detection
# ============================================================================

def find_logical_boundaries(lines: List[str]) -> List[int]:
    """
    Find line indices that represent logical boundaries in code.

    Detects:
    - Function/class/method definitions (via regex patterns)
    - Blank line separators (2+ consecutive blank lines)
    - Decorator blocks (treated as part of the next definition)

    Args:
        lines: List of source code lines

    Returns:
        Sorted list of line indices where new logical blocks start
    """
    boundaries = set()
    boundaries.add(0)  # Start of file is always a boundary

    consecutive_blanks = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track consecutive blank lines
        if stripped == '':
            consecutive_blanks += 1
            continue

        # After 2+ blank lines, the next non-blank line is a boundary
        if consecutive_blanks >= 2:
            boundaries.add(i)
        consecutive_blanks = 0

        # Check for function/class definitions
        for pattern in COMPILED_BOUNDARIES:
            if pattern.match(stripped):
                # Look back for decorators (e.g., @app.route, @staticmethod)
                decorator_start = i
                j = i - 1
                while j >= 0 and lines[j].strip().startswith('@'):
                    decorator_start = j
                    j -= 1
                boundaries.add(decorator_start)
                break

    return sorted(boundaries)


# ============================================================================
# Code Element Extraction
# ============================================================================

def extract_code_elements(text: str) -> Dict[str, List[str]]:
    """
    Extract key code elements from text using regex (no LLM).

    Returns dict with:
    - functions: List of function/method names
    - classes: List of class/struct/interface names
    - imports: List of import statements (summarized)
    """
    elements = {
        'functions': [],
        'classes': [],
        'imports': [],
    }

    for line in text.split('\n'):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith(('#', '//', '/*', '*', '--')):
            continue

        # Functions
        func_match = re.match(
            r'(?:pub\s+)?(?:async\s+)?(?:static\s+)?'
            r'(?:def|function|func|fn)\s+(\w+)',
            stripped
        )
        if func_match:
            elements['functions'].append(func_match.group(1))
            continue

        # Arrow functions / const functions (JS/TS)
        arrow_match = re.match(
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*'
            r'(?:async\s+)?(?:\(|function)',
            stripped
        )
        if arrow_match:
            elements['functions'].append(arrow_match.group(1))
            continue

        # Classes / interfaces / structs
        class_match = re.match(
            r'(?:export\s+)?(?:pub\s+)?(?:abstract\s+)?'
            r'(?:class|interface|struct|enum|trait|impl|type)\s+(\w+)',
            stripped
        )
        if class_match:
            elements['classes'].append(class_match.group(1))
            continue

        # Java/C# methods with access modifiers
        method_match = re.match(
            r'(?:public|private|protected|internal)\s+'
            r'(?:static\s+)?(?:async\s+)?'
            r'(?:void|int|str|string|bool|float|double|'
            r'Task|Promise|Optional|List|Dict|Map|Set|\w+(?:<.*?>)?)\s+'
            r'(\w+)\s*\(',
            stripped
        )
        if method_match:
            elements['functions'].append(method_match.group(1))
            continue

        # Imports (collect first few for context)
        if len(elements['imports']) < 5:
            if stripped.startswith(('import ', 'from ', 'require(',
                                    '#include', 'using ')):
                # Shorten long imports
                if len(stripped) > 80:
                    stripped = stripped[:77] + '...'
                elements['imports'].append(stripped)

    return elements


def _detect_section_type(text: str) -> str:
    """
    Detect the type of code section from its content.

    Returns one of: 'imports', 'class', 'function', 'declaration',
                    'configuration', 'documentation', 'code'
    """
    first_lines = text.strip().split('\n')[:5]
    first_text = '\n'.join(first_lines)

    # Check for imports
    import_lines = sum(
        1 for line in first_lines
        if line.strip().startswith(('import ', 'from ', 'require(',
                                    '#include', 'using '))
    )
    if import_lines >= 2:
        return 'imports'

    # Check for class definition
    if re.search(r'(?:^|\s)(?:export\s+)?(?:abstract\s+)?class\s+\w+',
                 first_text, re.MULTILINE):
        return 'class'

    # Check for function definition
    if re.search(
        r'(?:^|\s)(?:pub\s+)?(?:async\s+)?'
        r'(?:def|function|func|fn)\s+\w+',
        first_text, re.MULTILINE
    ):
        return 'function'

    # Check for constant/variable declarations
    if any(line.strip().startswith(('const ', 'let ', 'var ',
                                    'export const ', 'export let ',
                                    'final '))
           for line in first_lines):
        return 'declaration'

    # Check for uppercase constant assignments (e.g. MAX_RETRIES = 5)
    if any(re.match(r'^[A-Z][A-Z0-9_]+ *=', line.strip())
           for line in first_lines):
        return 'declaration'

    # Check for documentation
    if any(line.strip().startswith(('"""', "'''", '/**', '///'))
           for line in first_lines):
        return 'documentation'

    return 'code'


# ============================================================================
# Code-Aware Splitting
# ============================================================================

def split_code_at_boundaries(
    text: str,
    file_path: str,
    target_tokens: int = 2000,
    max_tokens: int = 2800,
    min_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split code text at logical boundaries rather than arbitrary positions.

    Strategy:
    1. Find logical boundaries (function/class definitions, blank line groups)
    2. Group consecutive sections into chunks targeting ~target_tokens
    3. If a section exceeds max_tokens, split it at line boundaries

    Args:
        text: Source code text to split
        file_path: Source file path (for logging)
        target_tokens: Target chunk size in tokens (~2000)
        max_tokens: Maximum chunk size before forced split (~2800)
        min_tokens: Minimum chunk size (merge smaller sections)

    Returns:
        List of chunk dicts with keys:
        - text: The chunk text
        - start_line: Starting line number (0-based)
        - end_line: Ending line number (0-based)
        - section_type: Detected type (function, class, imports, etc.)
    """
    lines = text.split('\n')
    if not lines:
        return []

    # For very small files, return as single chunk
    total_tokens = count_tokens(text)
    if total_tokens <= target_tokens:
        return [{
            'text': text,
            'start_line': 0,
            'end_line': len(lines) - 1,
            'section_type': _detect_section_type(text),
        }]

    # Find logical boundaries
    boundaries = find_logical_boundaries(lines)

    # Create sections between boundaries
    sections = []
    for i, boundary in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        section_text = '\n'.join(lines[boundary:end])
        if section_text.strip():
            sections.append({
                'text': section_text,
                'start_line': boundary,
                'end_line': end - 1,
                'tokens': count_tokens(section_text),
            })

    if not sections:
        return [{
            'text': text,
            'start_line': 0,
            'end_line': len(lines) - 1,
            'section_type': _detect_section_type(text),
        }]

    # Merge small sections and split large ones into chunks
    chunks = []
    current = {
        'text': '',
        'start_line': sections[0]['start_line'],
        'end_line': 0,
        'tokens': 0
    }

    for section in sections:
        combined_tokens = current['tokens'] + section['tokens']

        # If adding this section exceeds max and current has content, finalize
        if combined_tokens > max_tokens and current['tokens'] >= min_tokens:
            current['section_type'] = _detect_section_type(current['text'])
            chunks.append(current)
            current = {
                'text': '',
                'start_line': section['start_line'],
                'end_line': 0,
                'tokens': 0
            }

        # If a single section exceeds max_tokens, split it at nested
        # block boundaries (indentation changes, inner defs) rather than
        # arbitrary line counts. Falls back to line splitting if needed.
        if section['tokens'] > max_tokens:
            # Finalize current chunk if it has content
            if current['text'].strip():
                current['section_type'] = _detect_section_type(
                    current['text']
                )
                chunks.append(current)
                current = {
                    'text': '',
                    'start_line': section['start_line'],
                    'end_line': 0,
                    'tokens': 0
                }

            # Find nested block boundaries via indentation drops
            # and inner function/class definitions
            section_lines = section['text'].split('\n')
            nested_boundaries = [0]
            if len(section_lines) > 4:
                # Get base indentation of first non-blank line
                base_indent = 0
                for ln in section_lines[:5]:
                    stripped = ln.lstrip()
                    if stripped:
                        base_indent = len(ln) - len(stripped)
                        break

                for j, line in enumerate(section_lines[1:], 1):
                    stripped = line.lstrip()
                    if not stripped:
                        continue
                    indent = len(line) - len(stripped)
                    # Split at inner definitions (nested functions,
                    # classes, methods) at deeper indentation
                    if indent > base_indent and any(
                        stripped.startswith(kw) for kw in (
                            'def ', 'class ', 'async def ',
                            'function ', 'public ', 'private ',
                            'protected ', 'static ',
                        )
                    ):
                        nested_boundaries.append(j)
                    # Split at blank-line groups (2+ consecutive blanks)
                    elif (j >= 2
                          and not section_lines[j-1].strip()
                          and not section_lines[j-2].strip()
                          and stripped):
                        nested_boundaries.append(j)

            # If we found real nested boundaries, use them
            if len(nested_boundaries) > 1:
                nested_boundaries.append(len(section_lines))
                sub_lines = []
                sub_tokens = 0
                sub_start = section['start_line']
                for bi in range(len(nested_boundaries) - 1):
                    block = section_lines[
                        nested_boundaries[bi]:nested_boundaries[bi + 1]
                    ]
                    block_tokens = count_tokens('\n'.join(block))

                    if (sub_tokens + block_tokens > target_tokens
                            and sub_lines):
                        chunk_text = '\n'.join(sub_lines)
                        chunks.append({
                            'text': chunk_text,
                            'start_line': sub_start,
                            'end_line': sub_start + len(sub_lines) - 1,
                            'section_type': _detect_section_type(
                                chunk_text
                            ),
                        })
                        sub_lines = []
                        sub_tokens = 0
                        sub_start = (section['start_line']
                                     + nested_boundaries[bi])

                    sub_lines.extend(block)
                    sub_tokens += block_tokens

                if sub_lines:
                    chunk_text = '\n'.join(sub_lines)
                    chunks.append({
                        'text': chunk_text,
                        'start_line': sub_start,
                        'end_line': sub_start + len(sub_lines) - 1,
                        'section_type': _detect_section_type(chunk_text),
                    })
            else:
                # Fallback: split at line boundaries
                sub_lines = []
                sub_tokens = 0
                sub_start = section['start_line']

                for j, line in enumerate(section_lines):
                    line_tokens = count_tokens(line)
                    if (sub_tokens + line_tokens > target_tokens
                            and sub_lines):
                        chunk_text = '\n'.join(sub_lines)
                        chunks.append({
                            'text': chunk_text,
                            'start_line': sub_start,
                            'end_line': (sub_start
                                         + len(sub_lines) - 1),
                            'section_type': _detect_section_type(
                                chunk_text
                            ),
                        })
                        sub_lines = []
                        sub_tokens = 0
                        sub_start = section['start_line'] + j

                    sub_lines.append(line)
                    sub_tokens += line_tokens

                if sub_lines:
                    chunk_text = '\n'.join(sub_lines)
                    chunks.append({
                        'text': chunk_text,
                        'start_line': sub_start,
                        'end_line': (sub_start
                                     + len(sub_lines) - 1),
                        'section_type': _detect_section_type(
                            chunk_text
                        ),
                    })
            continue

        # Add section to current chunk
        if current['text']:
            current['text'] += '\n' + section['text']
        else:
            current['text'] = section['text']
        current['end_line'] = section['end_line']
        current['tokens'] = count_tokens(current['text'])

        # If we've reached target size, finalize the chunk
        if current['tokens'] >= target_tokens:
            current['section_type'] = _detect_section_type(current['text'])
            chunks.append(current)
            next_start = section['end_line'] + 1
            current = {
                'text': '',
                'start_line': next_start,
                'end_line': 0,
                'tokens': 0
            }

    # Finalize last chunk
    if current['text'].strip():
        current['section_type'] = _detect_section_type(current['text'])
        chunks.append(current)

    return chunks


# ============================================================================
# Context Enrichment for Embedding
# ============================================================================

def build_enriched_chunk_text(
    chunk_text: str,
    file_path: str,
    section_type: str,
    start_line: int,
    end_line: int,
    elements: Dict[str, List[str]] = None,
) -> str:
    """
    Build context-enriched text for embedding.

    Prepends structural context to the code chunk so that the embedding
    captures file identity and code structure. This significantly improves
    FAISS retrieval when wiki prompts reference specific files or components.

    The raw chunk_text is preserved in metadata for display purposes.

    Args:
        chunk_text: Raw code chunk text
        file_path: Source file path
        section_type: Type of code section (function, class, imports, etc.)
        start_line: Start line in source file (0-based)
        end_line: End line in source file (0-based)
        elements: Extracted code elements (functions, classes, imports)

    Returns:
        Enriched text for embedding
    """
    ext = file_path.rsplit('.', 1)[-1] if '.' in file_path else ''
    lang_name = LANGUAGE_MAP.get(ext, ext.upper() if ext else 'Code')

    # Build context header
    header = f"[File: {file_path} | Language: {lang_name}"
    if section_type and section_type != 'code':
        header += f" | Section: {section_type}"
    header += f" | Lines: {start_line + 1}-{end_line + 1}]"

    # Add element summary for richer embedding context
    element_parts = []
    if elements:
        if elements.get('classes'):
            element_parts.append(
                f"Defines classes: {', '.join(elements['classes'][:5])}"
            )
        if elements.get('functions'):
            element_parts.append(
                f"Defines functions: {', '.join(elements['functions'][:8])}"
            )

    if element_parts:
        context = header + '\n' + '\n'.join(element_parts)
    else:
        context = header

    return f"{context}\n\n{chunk_text}"


def build_enriched_doc_text(
    chunk_text: str,
    file_path: str,
) -> str:
    """
    Build context-enriched text for documentation file chunks.

    Lighter enrichment than code files - just adds file path context.

    Args:
        chunk_text: Raw documentation chunk text
        file_path: Source file path

    Returns:
        Enriched text for embedding
    """
    ext = file_path.rsplit('.', 1)[-1] if '.' in file_path else ''
    type_name = LANGUAGE_MAP.get(ext, 'Documentation')
    header = f"[File: {file_path} | Type: {type_name}]"
    return f"{header}\n\n{chunk_text}"


# ============================================================================
# Main Entry Point
# ============================================================================

def split_and_enrich_documents(
    documents: List[Document],
    target_tokens: int = 2000,
) -> List[Document]:
    """
    Split documents using code-aware logic and enrich with structural context.

    This is the main entry point that replaces naive TextSplitter for code files.

    For code files:
    - Uses code-aware boundary detection
    - Extracts structural metadata (functions, classes, imports)
    - Builds enriched embedding text with file context

    For documentation files:
    - Uses simple paragraph/section-based splitting
    - Adds file path context to embedding text

    Args:
        documents: List of raw Document objects (full files)
        target_tokens: Target chunk size in tokens

    Returns:
        List of pre-split, enriched Document objects ready for embedding.
        Each Document has:
        - text: Enriched text for embedding (includes context prefix)
        - meta_data: Original metadata + structural fields:
            - raw_chunk_text: Original chunk text (for display)
            - section_type: function/class/imports/code/etc.
            - start_line, end_line: Line range in source file
            - functions, classes: Extracted element names
            - chunk_index, total_chunks_in_file: Position info
    """
    all_chunks = []

    for doc in documents:
        file_path = doc.meta_data.get('file_path', 'unknown')
        is_code = doc.meta_data.get('is_code', False)

        # Collect chunks for this file (for neighbor context)
        file_chunks = []

        if is_code:
            # Code-aware splitting
            raw_chunks = split_code_at_boundaries(
                doc.text,
                file_path,
                target_tokens=target_tokens,
            )

            for i, chunk in enumerate(raw_chunks):
                elements = extract_code_elements(chunk['text'])
                enriched_text = build_enriched_chunk_text(
                    chunk['text'],
                    file_path,
                    chunk.get('section_type', 'code'),
                    chunk.get('start_line', 0),
                    chunk.get('end_line', 0),
                    elements,
                )

                # Exclude bulky keys from parent meta_data to avoid
                # duplicating full file content into every chunk
                chunk_meta = {
                    k: v for k, v in doc.meta_data.items()
                    if k not in ('raw_content', 'token_count')
                }
                chunk_doc = Document(
                    text=enriched_text,
                    meta_data={
                        **chunk_meta,
                        '_header_len': len(enriched_text) - len(chunk['text']),
                        'section_type': chunk.get('section_type', 'code'),
                        'start_line': chunk.get('start_line', 0),
                        'end_line': chunk.get('end_line', 0),
                        'functions': elements.get('functions', []),
                        'classes': elements.get('classes', []),
                        'chunk_index': i,
                        'total_chunks_in_file': len(raw_chunks),
                    },
                )
                file_chunks.append(chunk_doc)
        else:
            # Documentation / configuration files: split by paragraphs
            doc_chunks = _split_doc_text(doc.text, target_tokens)

            # Distinguish config files from documentation per design
            config_extensions = {
                'json', 'yaml', 'yml', 'toml', 'ini',
            }
            ext = (
                file_path.rsplit('.', 1)[-1].lower()
                if '.' in file_path else ''
            )
            file_section_type = (
                'configuration' if ext in config_extensions
                else 'documentation'
            )

            for i, chunk_text in enumerate(doc_chunks):
                enriched_text = build_enriched_doc_text(
                    chunk_text, file_path
                )
                chunk_meta = {
                    k: v for k, v in doc.meta_data.items()
                    if k not in ('raw_content', 'token_count')
                }
                chunk_doc = Document(
                    text=enriched_text,
                    meta_data={
                        **chunk_meta,
                        '_header_len': len(enriched_text) - len(chunk_text),
                        'section_type': file_section_type,
                        'start_line': 0,
                        'end_line': 0,
                        'functions': [],
                        'classes': [],
                        'chunk_index': i,
                        'total_chunks_in_file': len(doc_chunks),
                    },
                )
                file_chunks.append(chunk_doc)

        all_chunks.extend(file_chunks)

    logger.info(
        f"Split {len(documents)} files into {len(all_chunks)} "
        f"enriched chunks"
    )
    return all_chunks


def _split_doc_text(
    text: str,
    target_tokens: int = 2000,
) -> List[str]:
    """
    Split documentation text at paragraph/section boundaries.

    Uses heading markers (# , ## , etc.) and blank line groups
    as natural split points.

    Args:
        text: Documentation text
        target_tokens: Target tokens per chunk

    Returns:
        List of chunk texts
    """
    total_tokens = count_tokens(text)
    if total_tokens <= target_tokens:
        return [text]

    # Split at headings and blank line groups
    sections = re.split(r'\n(?=#{1,4}\s)', text)

    chunks = []
    current = ''
    current_tokens = 0

    for section in sections:
        section_tokens = count_tokens(section)

        if current_tokens + section_tokens > target_tokens and current:
            chunks.append(current.strip())
            current = ''
            current_tokens = 0

        # If single section too large, split at paragraphs
        if section_tokens > target_tokens:
            if current.strip():
                chunks.append(current.strip())
                current = ''
                current_tokens = 0

            paragraphs = section.split('\n\n')
            for para in paragraphs:
                para_tokens = count_tokens(para)
                if current_tokens + para_tokens > target_tokens and current:
                    chunks.append(current.strip())
                    current = ''
                    current_tokens = 0
                current += '\n\n' + para if current else para
                current_tokens += para_tokens
        else:
            current += '\n' + section if current else section
            current_tokens += section_tokens

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]
