"""
Pydantic models for codemap operations.

Defines the graph data structure: nodes (symbols) and edges (relationships).
"""

from typing import List, Optional, Dict
from pydantic import BaseModel


class SymbolNode(BaseModel):
    """A code symbol (file, class, function, method) as a graph node."""
    id: str
    name: str
    kind: str       # "file" | "class" | "function" | "method" | "module"
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    parent_id: Optional[str] = None
    language: Optional[str] = None
    signature: Optional[str] = None


class SymbolEdge(BaseModel):
    """A relationship between two symbols."""
    source_id: str
    target_id: str
    kind: str       # "imports" | "calls" | "inherits" | "implements"


class CodeMapMetadata(BaseModel):
    """Metadata about the codemap generation."""
    owner: str = ""
    repo: str = ""
    repo_type: str = ""
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    generated_at: Optional[str] = None
    total_files: int = 0
    total_symbols: int = 0
    total_edges: int = 0
    language_stats: Dict[str, int] = {}


class CodeMapData(BaseModel):
    """Root model for the complete code map graph."""
    nodes: List[SymbolNode] = []
    edges: List[SymbolEdge] = []
    metadata: CodeMapMetadata = CodeMapMetadata()
