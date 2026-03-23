"""
Wiki Module

This module provides wiki cache management and export including:
- Wiki cache read/write/delete operations
- Wiki export to Markdown/JSON
- Processed projects listing

Exports:
    - WikiPage, WikiSection, WikiStructureModel: Wiki structure models
    - WikiCacheData, WikiCacheRequest: Cache models
    - ProcessedProjectEntry: Project listing model
    - Wiki cache functions and routes
"""

from backend.modules.wiki.models import (
    WikiPage,
    WikiSection,
    WikiStructureModel,
    WikiCacheData,
    WikiCacheRequest,
    WikiExportRequest,
    ProcessedProjectEntry,
    Model,
    Provider,
    ModelConfig,
    AuthorizationConfig,
    FrontendLogRequest,
    FrontendLogBatchRequest,
)
from backend.modules.wiki.cache import (
    read_wiki_cache,
    save_wiki_cache,
    get_wiki_cache_filename,
    get_wiki_cache_path,
    get_wiki_cache_blob_path,
    WIKI_CACHE_DIR,
    WIKI_CACHE_BLOB_PREFIX,
)
from backend.modules.wiki.export import (
    generate_markdown_export,
    generate_json_export,
)
from backend.modules.wiki.xml_repair import (
    close_open_tags,
    repair_wiki_structure_xml,
)

__all__ = [
    # Models
    "WikiPage",
    "WikiSection",
    "WikiStructureModel",
    "WikiCacheData",
    "WikiCacheRequest",
    "WikiExportRequest",
    "ProcessedProjectEntry",
    "Model",
    "Provider",
    "ModelConfig",
    "AuthorizationConfig",
    "FrontendLogRequest",
    "FrontendLogBatchRequest",
    # Cache functions
    "read_wiki_cache",
    "save_wiki_cache",
    "get_wiki_cache_filename",
    "get_wiki_cache_path",
    "get_wiki_cache_blob_path",
    "WIKI_CACHE_DIR",
    "WIKI_CACHE_BLOB_PREFIX",
    # Export functions
    "generate_markdown_export",
    "generate_json_export",
    # XML repair
    "close_open_tags",
    "repair_wiki_structure_xml",
]
