"""
AST-based code analyzer using tree-sitter.

Extracts symbol definitions (functions, classes, methods) and relationships
(imports, calls, inheritance) from source files across multiple languages.
"""

import logging
from typing import List, Tuple, Optional

from tree_sitter import Language, Parser, Node

from backend.modules.codemap.models import SymbolNode, SymbolEdge

logger = logging.getLogger(__name__)

# ============================================================================
# Language setup — lazy-loaded parsers
# ============================================================================

_PARSERS: dict = {}

# Extension → tree-sitter language key
SUPPORTED_EXTENSIONS = {
    'py', 'js', 'jsx', 'ts', 'tsx', 'java', 'go', 'cs',
    'c', 'h', 'cpp', 'cc', 'cxx', 'hpp', 'hxx',
}

# Extension → display name (subset of code_splitter.LANGUAGE_MAP)
LANGUAGE_DISPLAY = {
    'py': 'Python', 'js': 'JavaScript', 'jsx': 'JavaScript',
    'ts': 'TypeScript', 'tsx': 'TypeScript',
    'java': 'Java', 'go': 'Go', 'cs': 'C#',
    'c': 'C', 'h': 'C/C++',
    'cpp': 'C++', 'cc': 'C++', 'cxx': 'C++',
    'hpp': 'C++', 'hxx': 'C++',
}


def _get_parser(ext: str) -> Optional[Parser]:
    """Get or create a tree-sitter parser for the given file extension."""
    if ext in _PARSERS:
        return _PARSERS[ext]

    try:
        lang = _load_language(ext)
        if not lang:
            return None
        parser = Parser(lang)
        _PARSERS[ext] = parser
        return parser
    except Exception as e:
        logger.debug(f"Failed to create parser for .{ext}: {e}")
        _PARSERS[ext] = None
        return None


def _load_language(ext: str) -> Optional[Language]:
    """Load a tree-sitter Language for the given extension."""
    if ext == 'py':
        import tree_sitter_python
        return Language(tree_sitter_python.language())
    elif ext in ('js', 'jsx'):
        import tree_sitter_javascript
        return Language(tree_sitter_javascript.language())
    elif ext == 'ts':
        import tree_sitter_typescript
        return Language(tree_sitter_typescript.language_typescript())
    elif ext == 'tsx':
        import tree_sitter_typescript
        return Language(tree_sitter_typescript.language_tsx())
    elif ext == 'java':
        import tree_sitter_java
        return Language(tree_sitter_java.language())
    elif ext == 'go':
        import tree_sitter_go
        return Language(tree_sitter_go.language())
    elif ext == 'cs':
        import tree_sitter_c_sharp
        return Language(tree_sitter_c_sharp.language())
    elif ext in ('c', 'h'):
        import tree_sitter_c
        return Language(tree_sitter_c.language())
    elif ext in ('cpp', 'cc', 'cxx', 'hpp', 'hxx'):
        import tree_sitter_cpp
        return Language(tree_sitter_cpp.language())
    return None


def _node_text(node: Node) -> str:
    """Extract text from a tree-sitter node."""
    return node.text.decode('utf-8', errors='replace')


def _make_id(file_path: str, line: int, name: str) -> str:
    """Create a stable symbol ID."""
    return f"{file_path}:{line}:{name}"


# ============================================================================
# Public API
# ============================================================================

def analyze_file(
    file_path: str,
    content: bytes,
    ext: str,
) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
    """Analyze a source file and extract symbols and relationships.

    Args:
        file_path: Relative path from repo root (e.g., "src/main.py").
        content: Raw file content as bytes.
        ext: File extension without dot (e.g., "py").

    Returns:
        Tuple of (nodes, edges) extracted from the file.
    """
    parser = _get_parser(ext)
    if not parser:
        return [], []

    try:
        tree = parser.parse(content)
    except Exception as e:
        logger.debug(f"Parse failed for {file_path}: {e}")
        return [], []

    lang = LANGUAGE_DISPLAY.get(ext, ext)
    nodes: List[SymbolNode] = []
    edges: List[SymbolEdge] = []

    if ext == 'py':
        _extract_python(tree.root_node, file_path, lang, nodes, edges)
    elif ext in ('js', 'jsx'):
        _extract_javascript(tree.root_node, file_path, lang, nodes, edges)
    elif ext in ('ts', 'tsx'):
        _extract_typescript(tree.root_node, file_path, lang, nodes, edges)
    elif ext == 'java':
        _extract_java(tree.root_node, file_path, lang, nodes, edges)
    elif ext == 'go':
        _extract_go(tree.root_node, file_path, lang, nodes, edges)
    elif ext == 'cs':
        _extract_csharp(tree.root_node, file_path, lang, nodes, edges)
    elif ext in ('c', 'h'):
        _extract_c(tree.root_node, file_path, lang, nodes, edges)
    elif ext in ('cpp', 'cc', 'cxx', 'hpp', 'hxx'):
        _extract_cpp(tree.root_node, file_path, lang, nodes, edges)

    return nodes, edges


# ============================================================================
# Python extractor
# ============================================================================

def _extract_python(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a Python AST."""
    for node in root.children:
        if node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            kind = 'method' if parent_id else 'function'
            sym_id = _make_id(file_path, node.start_point[0] + 1, name)
            params = node.child_by_field_name('parameters')
            sig = f"{name}({_node_text(params)})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind=kind,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            # Extract calls within this function
            body = node.child_by_field_name('body')
            if body:
                _extract_calls(body, file_path, sym_id, edges)

        elif node.type == 'decorated_definition':
            # Decorators wrap the actual definition
            for child in node.children:
                if child.type in (
                    'function_definition', 'class_definition'
                ):
                    _extract_python(
                        node, file_path, lang,
                        nodes, edges, parent_id,
                    )
                    break

        elif node.type == 'class_definition':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(file_path, node.start_point[0] + 1, name)
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            # Inheritance
            superclasses = node.child_by_field_name('superclasses')
            if not superclasses:
                # Python uses argument_list for base classes
                for child in node.children:
                    if child.type == 'argument_list':
                        superclasses = child
                        break
            if superclasses:
                for arg in superclasses.children:
                    if arg.type == 'identifier':
                        edges.append(SymbolEdge(
                            source_id=sym_id,
                            target_id=_node_text(arg),
                            kind='inherits',
                        ))
            # Recurse into class body for methods
            body = node.child_by_field_name('body')
            if body:
                _extract_python(
                    body, file_path, lang,
                    nodes, edges, parent_id=sym_id,
                )

        elif node.type == 'import_from_statement':
            # from X import Y
            module_node = None
            for child in node.children:
                if child.type == 'dotted_name' and module_node is None:
                    module_node = child
                    break
                elif child.type == 'relative_import':
                    module_node = child
                    break
            if module_node:
                module_path = _node_text(module_node)
                edges.append(SymbolEdge(
                    source_id=file_path,
                    target_id=module_path,
                    kind='imports',
                ))

        elif node.type == 'import_statement':
            for child in node.children:
                if child.type == 'dotted_name':
                    edges.append(SymbolEdge(
                        source_id=file_path,
                        target_id=_node_text(child),
                        kind='imports',
                    ))


def _extract_calls(
    node: Node, file_path: str, enclosing_id: str,
    edges: List[SymbolEdge],
):
    """Recursively extract function/method calls from a node."""
    if node.type == 'call':
        func_node = node.child_by_field_name('function')
        if func_node:
            if func_node.type == 'identifier':
                callee = _node_text(func_node)
                edges.append(SymbolEdge(
                    source_id=enclosing_id,
                    target_id=callee,
                    kind='calls',
                ))
            elif func_node.type == 'attribute':
                # e.g., obj.method() — extract method name
                attr = func_node.child_by_field_name('attribute')
                if attr:
                    edges.append(SymbolEdge(
                        source_id=enclosing_id,
                        target_id=_node_text(attr),
                        kind='calls',
                    ))
    for child in node.children:
        _extract_calls(child, file_path, enclosing_id, edges)


# ============================================================================
# JavaScript extractor
# ============================================================================

def _extract_javascript(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a JavaScript AST."""
    for node in root.children:
        if node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(file_path, node.start_point[0] + 1, name)
            params = node.child_by_field_name('parameters')
            sig = f"{name}({_node_text(params)})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='function',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_js(body, file_path, sym_id, edges)

        elif node.type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(file_path, node.start_point[0] + 1, name)
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            # Inheritance via class_heritage
            for child in node.children:
                if child.type == 'class_heritage':
                    for hc in child.children:
                        if hc.type == 'identifier':
                            edges.append(SymbolEdge(
                                source_id=sym_id,
                                target_id=_node_text(hc),
                                kind='inherits',
                            ))
            # Class body — extract methods
            body = node.child_by_field_name('body')
            if body:
                _extract_js_class_body(
                    body, file_path, lang, nodes, edges, sym_id,
                )

        elif node.type == 'lexical_declaration':
            # const foo = () => {} or const foo = function() {}
            for child in node.children:
                if child.type == 'variable_declarator':
                    _extract_js_var_decl(
                        child, file_path, lang,
                        nodes, edges, parent_id,
                    )

        elif node.type == 'import_statement':
            source_node = node.child_by_field_name('source')
            if source_node:
                # Strip quotes from string
                import_path = _node_text(source_node).strip("'\"")
                edges.append(SymbolEdge(
                    source_id=file_path,
                    target_id=import_path,
                    kind='imports',
                ))

        elif node.type == 'export_statement':
            # Recurse into exported declarations
            for child in node.children:
                if child.type in (
                    'function_declaration', 'class_declaration',
                    'lexical_declaration',
                ):
                    _extract_javascript(
                        node, file_path, lang,
                        nodes, edges, parent_id,
                    )
                    break


def _extract_js_var_decl(
    node: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str],
):
    """Extract arrow functions and function expressions from variable declarations."""
    name_node = node.child_by_field_name('name')
    value_node = node.child_by_field_name('value')
    if not name_node or not value_node:
        return
    if value_node.type not in ('arrow_function', 'function'):
        return
    name = _node_text(name_node)
    sym_id = _make_id(file_path, node.start_point[0] + 1, name)
    params = value_node.child_by_field_name('parameters')
    sig = f"{name}({_node_text(params)})" if params else name
    nodes.append(SymbolNode(
        id=sym_id, name=name, kind='function',
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        parent_id=parent_id, language=lang,
        signature=sig,
    ))
    body = value_node.child_by_field_name('body')
    if body:
        _extract_calls_js(body, file_path, sym_id, edges)


def _extract_js_class_body(
    body: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    class_id: str,
):
    """Extract methods from a JS/TS class body."""
    for node in body.children:
        if node.type == 'method_definition':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            params = node.child_by_field_name('parameters')
            sig = f"{name}({_node_text(params)})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='method',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=class_id, language=lang,
                signature=sig,
            ))
            stmt_body = node.child_by_field_name('body')
            if stmt_body:
                _extract_calls_js(
                    stmt_body, file_path, sym_id, edges,
                )


def _extract_calls_js(
    node: Node, file_path: str, enclosing_id: str,
    edges: List[SymbolEdge],
):
    """Recursively extract call expressions from JS/TS nodes."""
    if node.type == 'call_expression':
        func = node.child_by_field_name('function')
        if func:
            if func.type == 'identifier':
                edges.append(SymbolEdge(
                    source_id=enclosing_id,
                    target_id=_node_text(func),
                    kind='calls',
                ))
            elif func.type == 'member_expression':
                prop = func.child_by_field_name('property')
                if prop:
                    edges.append(SymbolEdge(
                        source_id=enclosing_id,
                        target_id=_node_text(prop),
                        kind='calls',
                    ))
    for child in node.children:
        _extract_calls_js(child, file_path, enclosing_id, edges)


# ============================================================================
# TypeScript extractor (extends JS with interfaces and type aliases)
# ============================================================================

def _extract_typescript(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a TypeScript AST."""
    for node in root.children:
        if node.type == 'interface_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            # extends clause
            for child in node.children:
                if child.type == 'extends_type_clause':
                    for tc in child.children:
                        if tc.type == 'type_identifier':
                            edges.append(SymbolEdge(
                                source_id=sym_id,
                                target_id=_node_text(tc),
                                kind='inherits',
                            ))

        elif node.type == 'type_alias_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))

        else:
            # Delegate to JS extractor for shared node types
            _extract_javascript(
                node if node.type in ('program',) else root,
                file_path, lang, nodes, edges, parent_id,
            )
            return  # JS extractor handles full root iteration


# ============================================================================
# Java extractor
# ============================================================================

def _extract_java(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a Java AST."""
    for node in root.children:
        if node.type in ('class_declaration', 'interface_declaration'):
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            # Superclass
            sc = node.child_by_field_name('superclass')
            if sc:
                for child in sc.children:
                    if child.type == 'type_identifier':
                        edges.append(SymbolEdge(
                            source_id=sym_id,
                            target_id=_node_text(child),
                            kind='inherits',
                        ))
            # Interfaces
            ifaces = node.child_by_field_name('interfaces')
            if ifaces:
                for child in ifaces.children:
                    if child.type == 'type_identifier':
                        edges.append(SymbolEdge(
                            source_id=sym_id,
                            target_id=_node_text(child),
                            kind='implements',
                        ))
            # Body
            body = node.child_by_field_name('body')
            if body:
                _extract_java(
                    body, file_path, lang,
                    nodes, edges, parent_id=sym_id,
                )

        elif node.type == 'method_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            kind = 'method' if parent_id else 'function'
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            params = node.child_by_field_name('parameters')
            sig = f"{name}({_node_text(params)})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind=kind,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_java(body, file_path, sym_id, edges)

        elif node.type == 'constructor_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='method',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=f"{name}(...)",
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_java(body, file_path, sym_id, edges)

        elif node.type == 'import_declaration':
            for child in node.children:
                if child.type == 'scoped_identifier':
                    edges.append(SymbolEdge(
                        source_id=file_path,
                        target_id=_node_text(child),
                        kind='imports',
                    ))
                    break

        elif node.type == 'program':
            _extract_java(
                node, file_path, lang,
                nodes, edges, parent_id,
            )


def _extract_calls_java(
    node: Node, file_path: str, enclosing_id: str,
    edges: List[SymbolEdge],
):
    """Recursively extract method invocations from Java nodes."""
    if node.type == 'method_invocation':
        name_node = node.child_by_field_name('name')
        if name_node:
            edges.append(SymbolEdge(
                source_id=enclosing_id,
                target_id=_node_text(name_node),
                kind='calls',
            ))
    for child in node.children:
        _extract_calls_java(child, file_path, enclosing_id, edges)


# ============================================================================
# Go extractor
# ============================================================================

def _extract_go(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a Go AST."""
    for node in root.children:
        if node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            params = node.child_by_field_name('parameters')
            sig = f"{name}({_node_text(params)})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='function',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_go(body, file_path, sym_id, edges)

        elif node.type == 'method_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            # Try to find receiver type for parent_id
            receiver = node.child_by_field_name('receiver')
            recv_type = None
            if receiver:
                for child in _walk(receiver):
                    if child.type == 'type_identifier':
                        recv_type = _node_text(child)
                        break
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='method',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=recv_type,  # Resolved later by graph_builder
                language=lang,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_go(body, file_path, sym_id, edges)

        elif node.type == 'type_declaration':
            for child in node.children:
                if child.type == 'type_spec':
                    name_node = child.child_by_field_name('name')
                    if not name_node:
                        continue
                    name = _node_text(name_node)
                    sym_id = _make_id(
                        file_path, child.start_point[0] + 1, name,
                    )
                    nodes.append(SymbolNode(
                        id=sym_id, name=name, kind='class',
                        file_path=file_path,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        parent_id=parent_id, language=lang,
                    ))

        elif node.type == 'import_declaration':
            for child in _walk(node):
                if child.type == 'import_spec':
                    path_node = child.child_by_field_name('path')
                    if path_node:
                        imp = _node_text(path_node).strip('"')
                        edges.append(SymbolEdge(
                            source_id=file_path,
                            target_id=imp,
                            kind='imports',
                        ))
                elif child.type == 'interpreted_string_literal':
                    # Single import: import "fmt"
                    if child.parent and child.parent.type == 'import_declaration':
                        imp = _node_text(child).strip('"')
                        edges.append(SymbolEdge(
                            source_id=file_path,
                            target_id=imp,
                            kind='imports',
                        ))

        elif node.type == 'source_file':
            _extract_go(
                node, file_path, lang,
                nodes, edges, parent_id,
            )


def _extract_calls_go(
    node: Node, file_path: str, enclosing_id: str,
    edges: List[SymbolEdge],
):
    """Recursively extract call expressions from Go nodes."""
    if node.type == 'call_expression':
        func = node.child_by_field_name('function')
        if func:
            if func.type == 'identifier':
                edges.append(SymbolEdge(
                    source_id=enclosing_id,
                    target_id=_node_text(func),
                    kind='calls',
                ))
            elif func.type == 'selector_expression':
                field = func.child_by_field_name('field')
                if field:
                    edges.append(SymbolEdge(
                        source_id=enclosing_id,
                        target_id=_node_text(field),
                        kind='calls',
                    ))
    for child in node.children:
        _extract_calls_go(child, file_path, enclosing_id, edges)


# ============================================================================
# C# extractor
# ============================================================================

def _extract_csharp(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a C# AST."""
    for node in root.children:
        if node.type in (
            'class_declaration', 'interface_declaration',
            'struct_declaration',
        ):
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            # Base list (inheritance)
            for child in node.children:
                if child.type == 'base_list':
                    for bc in child.children:
                        if bc.type in (
                            'identifier', 'generic_name',
                            'qualified_name',
                        ):
                            edges.append(SymbolEdge(
                                source_id=sym_id,
                                target_id=_node_text(bc),
                                kind='inherits',
                            ))
            # Body
            body = node.child_by_field_name('body')
            if body:
                _extract_csharp(
                    body, file_path, lang,
                    nodes, edges, parent_id=sym_id,
                )

        elif node.type == 'method_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            kind = 'method' if parent_id else 'function'
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            params = node.child_by_field_name('parameters')
            sig = f"{name}({_node_text(params)})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind=kind,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_cs(body, file_path, sym_id, edges)

        elif node.type == 'constructor_declaration':
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='method',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_cs(body, file_path, sym_id, edges)

        elif node.type == 'using_directive':
            for child in _walk(node):
                if child.type in (
                    'identifier', 'qualified_name',
                ):
                    edges.append(SymbolEdge(
                        source_id=file_path,
                        target_id=_node_text(child),
                        kind='imports',
                    ))
                    break

        elif node.type in (
            'namespace_declaration',
            'file_scoped_namespace_declaration',
            'compilation_unit',
            'declaration_list',
        ):
            _extract_csharp(
                node, file_path, lang,
                nodes, edges, parent_id,
            )


def _extract_calls_cs(
    node: Node, file_path: str, enclosing_id: str,
    edges: List[SymbolEdge],
):
    """Recursively extract invocation expressions from C# nodes."""
    if node.type == 'invocation_expression':
        func = node.child_by_field_name('function')
        if func:
            if func.type == 'identifier':
                edges.append(SymbolEdge(
                    source_id=enclosing_id,
                    target_id=_node_text(func),
                    kind='calls',
                ))
            elif func.type == 'member_access_expression':
                name_node = func.child_by_field_name('name')
                if name_node:
                    edges.append(SymbolEdge(
                        source_id=enclosing_id,
                        target_id=_node_text(name_node),
                        kind='calls',
                    ))
    for child in node.children:
        _extract_calls_cs(child, file_path, enclosing_id, edges)


# ============================================================================
# C extractor
# ============================================================================

def _extract_c(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a C AST."""
    for node in root.children:
        if node.type == 'function_definition':
            # Get name from function_declarator child
            decl = node.child_by_field_name('declarator')
            name = _c_declarator_name(decl)
            if not name:
                continue
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            params = _c_declarator_params(decl)
            sig = f"{name}({params})" if params else name
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='function',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_c(body, file_path, sym_id, edges)

        elif node.type == 'declaration':
            # Function declarations (prototypes)
            decl = node.child_by_field_name('declarator')
            if decl and decl.type == 'function_declarator':
                name = _c_declarator_name(decl)
                if name:
                    sym_id = _make_id(
                        file_path, node.start_point[0] + 1, name,
                    )
                    nodes.append(SymbolNode(
                        id=sym_id, name=name, kind='function',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_id=parent_id, language=lang,
                    ))

        elif node.type == 'struct_specifier':
            name_node = node.child_by_field_name('name')
            if name_node:
                name = _node_text(name_node)
                sym_id = _make_id(
                    file_path, node.start_point[0] + 1, name,
                )
                nodes.append(SymbolNode(
                    id=sym_id, name=name, kind='class',
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_id=parent_id, language=lang,
                ))

        elif node.type == 'type_definition':
            # typedef struct { ... } Name;
            for child in node.children:
                if child.type == 'type_identifier':
                    name = _node_text(child)
                    sym_id = _make_id(
                        file_path, node.start_point[0] + 1, name,
                    )
                    nodes.append(SymbolNode(
                        id=sym_id, name=name, kind='class',
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_id=parent_id, language=lang,
                    ))
                    break

        elif node.type == 'preproc_include':
            # #include "file.h" or #include <header.h>
            for child in node.children:
                if child.type == 'string_literal':
                    # Local include: "file.h"
                    inc_text = _node_text(child).strip('"')
                    edges.append(SymbolEdge(
                        source_id=file_path,
                        target_id=inc_text,
                        kind='imports',
                    ))
                    break
                elif child.type == 'system_lib_string':
                    # System include: <stdio.h> — skip
                    break


def _extract_calls_c(
    node: Node, file_path: str, enclosing_id: str,
    edges: List[SymbolEdge],
):
    """Recursively extract call expressions from C nodes."""
    if node.type == 'call_expression':
        func = node.child_by_field_name('function')
        if func:
            if func.type == 'identifier':
                edges.append(SymbolEdge(
                    source_id=enclosing_id,
                    target_id=_node_text(func),
                    kind='calls',
                ))
            elif func.type == 'field_expression':
                field = func.child_by_field_name('field')
                if field:
                    edges.append(SymbolEdge(
                        source_id=enclosing_id,
                        target_id=_node_text(field),
                        kind='calls',
                    ))
    for child in node.children:
        _extract_calls_c(child, file_path, enclosing_id, edges)


def _c_declarator_name(decl: Optional[Node]) -> Optional[str]:
    """Extract function name from a C/C++ declarator node."""
    if not decl:
        return None
    if decl.type == 'function_declarator':
        inner = decl.child_by_field_name('declarator')
        if inner and inner.type == 'identifier':
            return _node_text(inner)
        # Pointer declarator: (*func_ptr)(args)
        if inner:
            return _c_declarator_name(inner)
    elif decl.type == 'identifier':
        return _node_text(decl)
    elif decl.type == 'pointer_declarator':
        inner = decl.child_by_field_name('declarator')
        return _c_declarator_name(inner)
    return None


def _c_declarator_params(decl: Optional[Node]) -> Optional[str]:
    """Extract parameter text from a C/C++ function declarator."""
    if not decl:
        return None
    if decl.type == 'function_declarator':
        params = decl.child_by_field_name('parameters')
        if params:
            return _node_text(params)
    return None


# ============================================================================
# C++ extractor (extends C with classes, namespaces, methods)
# ============================================================================

def _extract_cpp(
    root: Node, file_path: str, lang: str,
    nodes: List[SymbolNode], edges: List[SymbolEdge],
    parent_id: Optional[str] = None,
):
    """Extract symbols from a C++ AST."""
    for node in root.children:
        if node.type in ('class_specifier', 'struct_specifier'):
            name_node = node.child_by_field_name('name')
            if not name_node:
                continue
            name = _node_text(name_node)
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, name,
            )
            nodes.append(SymbolNode(
                id=sym_id, name=name, kind='class',
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
            ))
            # Base classes
            for child in node.children:
                if child.type == 'base_class_clause':
                    for bc in child.children:
                        if bc.type == 'type_identifier':
                            edges.append(SymbolEdge(
                                source_id=sym_id,
                                target_id=_node_text(bc),
                                kind='inherits',
                            ))
            # Class body (methods, nested types)
            body = node.child_by_field_name('body')
            if body:
                _extract_cpp(
                    body, file_path, lang,
                    nodes, edges, parent_id=sym_id,
                )

        elif node.type == 'function_definition':
            decl = node.child_by_field_name('declarator')
            fname = _cpp_function_name(decl)
            if not fname:
                continue
            kind = 'method' if parent_id else 'function'
            sym_id = _make_id(
                file_path, node.start_point[0] + 1, fname,
            )
            params = _c_declarator_params(decl)
            sig = f"{fname}({params})" if params else fname
            nodes.append(SymbolNode(
                id=sym_id, name=fname, kind=kind,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_id=parent_id, language=lang,
                signature=sig,
            ))
            body = node.child_by_field_name('body')
            if body:
                _extract_calls_c(body, file_path, sym_id, edges)

        elif node.type == 'declaration':
            decl = node.child_by_field_name('declarator')
            if decl and decl.type == 'function_declarator':
                fname = _cpp_function_name(decl)
                if fname:
                    kind = 'method' if parent_id else 'function'
                    sym_id = _make_id(
                        file_path, node.start_point[0] + 1,
                        fname,
                    )
                    nodes.append(SymbolNode(
                        id=sym_id, name=fname, kind=kind,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_id=parent_id, language=lang,
                    ))

        elif node.type == 'namespace_definition':
            # Recurse into namespace body
            name_node = None
            for child in node.children:
                if child.type == 'namespace_identifier':
                    name_node = child
                    break
            body = node.child_by_field_name('body')
            if body:
                _extract_cpp(
                    body, file_path, lang,
                    nodes, edges, parent_id,
                )

        elif node.type == 'preproc_include':
            for child in node.children:
                if child.type == 'string_literal':
                    inc_text = _node_text(child).strip('"')
                    edges.append(SymbolEdge(
                        source_id=file_path,
                        target_id=inc_text,
                        kind='imports',
                    ))
                    break
                elif child.type == 'system_lib_string':
                    break

        elif node.type in (
            'field_declaration_list', 'declaration_list',
            'translation_unit',
        ):
            _extract_cpp(
                node, file_path, lang,
                nodes, edges, parent_id,
            )


def _cpp_function_name(
    decl: Optional[Node],
) -> Optional[str]:
    """Extract function name from a C++ declarator.

    Handles qualified names like ClassName::method.
    """
    if not decl:
        return None
    if decl.type == 'function_declarator':
        inner = decl.child_by_field_name('declarator')
        if not inner:
            return None
        if inner.type == 'identifier':
            return _node_text(inner)
        if inner.type == 'qualified_identifier':
            # ClassName::method — extract the rightmost name
            name = inner.child_by_field_name('name')
            if name:
                return _node_text(name)
        if inner.type == 'field_identifier':
            return _node_text(inner)
        return _c_declarator_name(inner)
    return _c_declarator_name(decl)


# ============================================================================
# Shared utilities
# ============================================================================

def _walk(node: Node):
    """Iterate all descendants of a node (breadth-first)."""
    queue = list(node.children)
    while queue:
        child = queue.pop(0)
        yield child
        queue.extend(child.children)
