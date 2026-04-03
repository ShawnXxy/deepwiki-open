"""
File processing types for consistent file filtering and metadata.
"""
from pathlib import Path
from typing import Optional, Set
from pydantic import BaseModel, Field, validator
from datetime import datetime


class FileFilter(BaseModel):
    """
    File filtering configuration for repository processing.
    
    Supports both inclusion and exclusion modes:
    - Exclusion mode: Process all files except those matching filters
    - Inclusion mode: Only process files matching filters
    
    Examples:
        >>> filter = FileFilter(
        ...     excluded_dirs={'node_modules', '.git'},
        ...     excluded_patterns={'*.pyc', '*.log'}
        ... )
    """
    included_dirs: Set[str] = Field(default_factory=set, description="Directories to include exclusively")
    excluded_dirs: Set[str] = Field(default_factory=set, description="Directories to exclude")
    included_patterns: Set[str] = Field(default_factory=set, description="File patterns to include (e.g., *.py)")
    excluded_patterns: Set[str] = Field(default_factory=set, description="File patterns to exclude")
    
    @validator('included_dirs', 'excluded_dirs', pre=True, always=True)
    def normalize_paths(cls, v):
        """Normalize path separators to forward slashes. Returns empty set if None."""
        if v is None:
            return set()
        if isinstance(v, (list, set)):
            return {str(Path(p)).replace('\\', '/') for p in v}
        return v
    
    @validator('included_patterns', 'excluded_patterns', pre=True, always=True)
    def normalize_patterns(cls, v):
        """Convert lists to sets and normalize. Returns empty set if None."""
        if v is None:
            return set()
        if isinstance(v, list):
            return set(v)
        return v
    
    def is_inclusion_mode(self) -> bool:
        """Check if filter is in inclusion mode."""
        return bool(self.included_dirs or self.included_patterns)
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on filters.
        
        Args:
            file_path: Relative path of the file
            
        Returns:
            True if file should be processed, False otherwise
        """
        path = Path(file_path)
        path_parts = set(path.parts)
        
        # Ensure sets are not None (defensive check)
        included_dirs = self.included_dirs or set()
        excluded_dirs = self.excluded_dirs or set()
        included_patterns = self.included_patterns or set()
        excluded_patterns = self.excluded_patterns or set()
        
        if self.is_inclusion_mode():
            # Inclusion mode: file must match included dirs or patterns
            dir_match = any(included in path_parts for included in included_dirs)
            pattern_match = any(path.match(pattern) for pattern in included_patterns)
            return dir_match or pattern_match
        else:
            # Exclusion mode: file must not match excluded dirs or patterns
            dir_match = any(excluded in path_parts for excluded in excluded_dirs)
            if dir_match:
                return False
            
            pattern_match = any(path.match(pattern) for pattern in excluded_patterns)
            return not pattern_match


class ProcessedFile(BaseModel):
    """
    Metadata for a processed file in the repository.
    
    Tracks file details, processing metrics, and content analysis.
    """
    path: str = Field(..., description="Relative path from repository root")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    token_count: int = Field(..., ge=0, description="Number of tokens in file")
    language: Optional[str] = Field(None, description="Programming language detected")
    encoding: str = Field(default="utf-8", description="File encoding")
    processing_time_ms: float = Field(..., ge=0, description="Time taken to process")
    chunk_count: Optional[int] = Field(None, ge=0, description="Number of chunks created")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "path": "src/main.py",
                "size_bytes": 2048,
                "token_count": 512,
                "language": "python",
                "encoding": "utf-8",
                "processing_time_ms": 145.5,
                "chunk_count": 3
            }
        }


class FileProcessingStats(BaseModel):
    """
    Aggregate statistics for file processing operations.
    
    Provides overview of processing results and performance metrics.
    """
    total_files: int = Field(..., ge=0)
    processed_files: int = Field(..., ge=0)
    failed_files: int = Field(default=0, ge=0)
    skipped_files: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    total_processing_time_ms: float = Field(..., ge=0)
    languages: dict[str, int] = Field(default_factory=dict, description="Count of files per language")
    
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per file."""
        if self.processed_files == 0:
            return 0.0
        return self.total_processing_time_ms / self.processed_files
    
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100


class DocumentChunk(BaseModel):
    """
    A chunk of document content for embedding.
    
    Used when large files are split into smaller pieces for processing.
    """
    chunk_id: str = Field(..., description="Unique identifier for chunk")
    file_path: str = Field(..., description="Source file path")
    content: str = Field(..., description="Chunk content")
    token_count: int = Field(..., ge=0, description="Tokens in this chunk")
    chunk_index: int = Field(..., ge=0, description="Position in file (0-based)")
    total_chunks: int = Field(..., ge=1, description="Total chunks in file")
    overlap_tokens: int = Field(default=0, ge=0, description="Overlapping tokens with adjacent chunks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "main.py_chunk_0",
                "file_path": "src/main.py",
                "content": "import os\n\ndef main():\n    ...",
                "token_count": 128,
                "chunk_index": 0,
                "total_chunks": 3,
                "overlap_tokens": 20
            }
        }


class RepositoryMetadata(BaseModel):
    """
    Metadata about a processed repository.
    
    Contains high-level information about repository structure and content.
    """
    owner: str
    repo: str
    repo_type: str
    branch: str
    total_files: int
    total_size_bytes: int
    total_tokens: int
    languages: dict[str, int]
    processing_date: datetime = Field(default_factory=datetime.utcnow)
    filters_applied: Optional[FileFilter] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "owner": "AsyncFuncAI",
                "repo": "deepwiki-open",
                "repo_type": "github",
                "branch": "main",
                "total_files": 150,
                "total_size_bytes": 2048000,
                "total_tokens": 512000,
                "languages": {"Python": 80, "TypeScript": 60, "JSON": 10},
                "processing_date": "2026-01-07T12:00:00Z"
            }
        }
