# Wiki Module

Wiki cache management and export.

## Responsibility

Handles generated wiki data lifecycle:
- Read/write wiki cache JSON files (local disk or Azure Blob)
- Export wiki content as Markdown or JSON
- Manage cache file naming conventions (with branch + legacy support)
- Repair truncated XML from content-filtered LLM responses

## Files

| File | Purpose |
|------|---------|
| `models.py` | `WikiPage`, `WikiSection`, `WikiStructureModel`, `WikiCacheData` — data models |
| `cache.py` | `read_wiki_cache()`, `save_wiki_cache()` — storage operations |
| `export.py` | `generate_markdown_export()`, `generate_json_export()` |
| `xml_repair.py` | `repair_wiki_structure_xml()`, `close_open_tags()` — truncation recovery |

## Dependencies

- **Invokes:** `clients/blob_client` (Azure Blob storage), `repository/models` (RepoInfo)
- **Invoked by:** `processor/` (save cache after generation), frontend Next.js API routes (read cache)
