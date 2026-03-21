# Processor Module

Standalone CLI pipeline that generates AI wikis from code repositories.

> For the full architecture deep-dive, see [DESIGN.md](DESIGN.md).

## Responsibility

Runs as a one-shot CLI command — no server required. Transforms a code repository into a structured, AI-generated wiki saved as JSON:

```
python -m backend.processor.code_processor --repo=URL --branch=main --mode=local
```

**Pipeline:** Clone repo → Embed documents → Generate wiki via LLM → Save JSON cache

**Output:** `~/.adalflow/wikicache/*.json` — read directly by the Next.js frontend.

## Files

| File | Purpose |
|------|---------|
| `code_processor.py` | CLI entry point, argument parsing, auth resolution, mode dispatch |
| `wiki_generator.py` | LLM-based structure + content generation, XML parsing |
| `cloud_setup.py` | Azure AI Search index + AML pipeline management |
| `DESIGN.md` | Comprehensive design document (architecture, data flow, decisions) |

## How It Works

### Pipeline Stages

```
Stage 1: Clone Repository
    git clone → ~/.adalflow/repos/{owner}_{repo}/
    Extract HEAD commit hash for citation URLs

Stage 2: Embed Documents (RAG Preparation)
    read_all_documents() → code-aware splitting → Azure OpenAI embeddings
    Save as JSON vectors → Build FAISS index

Stage 3: Generate Wiki
    File tree + README → LLM generates XML structure (pages + sections)
    For each page: RAG retrieval → LLM generates Markdown content

Stage 4: Save Wiki Cache
    Assemble WikiCacheData → Save to ~/.adalflow/wikicache/*.json

Stage 5: Push to AI Search (cloud mode only)
    Load vectors → Push to Azure AI Search index
```

### Execution Modes

| Mode | Command | Auth | Storage |
|------|---------|------|---------|
| **local** | `--mode=local` | PAT from `.env` or `az login` | Local disk |
| **docker** | `--mode=docker` | PAT from `backend/.env` (required) | Mounted volume |
| **cloud** | `--mode=cloud` | MSI (DefaultAzureCredential) | Azure Blob + AI Search |

### Authentication

Resolves credentials in order:
1. `REPO_ACCESS_TOKEN` / `ADO_PAT` / `AZURE_DEVOPS_PAT` environment variables
2. `DefaultAzureCredential` (MSI → Azure CLI → VS Code → Environment)
3. Exit with error if no auth available

### Wiki Structure Generation

The LLM receives the repo's file tree and README, then outputs XML:

```xml
<wiki_structure>
  <title>Repository Wiki</title>
  <sections>
    <section id="1">
      <title>Getting Started</title>
      <pages><page_ref>1</page_ref><page_ref>1.1</page_ref></pages>
    </section>
  </sections>
  <pages>
    <page id="1">
      <title>Overview</title>
      <file_path>README.md</file_path>
      <importance>high</importance>
    </page>
  </pages>
</wiki_structure>
```

The XML is parsed by `_parse_structure_xml()` with:
- Dash-to-dot ID normalization (`"2-2"` → `"2.2"`)
- Duplicate ID resolution
- Content filter retry (directory-only tree fallback)
- Truncated XML repair via `xml_repair.py`

### Page Content Generation

For each page, the generator:
1. Calls `RAG.call_with_file_filter()` to get relevant code chunks (file-priority + semantic)
2. Formats context with commit-pinned source URLs
3. Sends prompt to LLM with page title, files, cross-page catalog
4. LLM generates Markdown with:
   - `<details>` block listing relevant source files
   - Mermaid diagrams (architecture, data flow, state machines)
   - `[[deepwiki://2.1]]` cross-page links
   - Code explanations (summarized, not verbatim)

### Cloud Mode

`cloud_setup.py` manages per-repo Azure resources:

- **AI Search Index** — One per `{owner}_{repo}_{branch}`, stores vector embeddings for cloud-based RAG
- **AML Compute Cluster** — Shared, auto-scales 0→4 instances, 600s idle timeout
- **AML Scheduled Pipeline** — One per repo+branch, runs the processor on a recurrence schedule (default: every 480 hours)
- **Teardown** — `teardown_cloud_resources()` deletes index + pipeline

## Usage

```bash
# Local mode with PAT
export REPO_ACCESS_TOKEN="your-pat"
python -m backend.processor.code_processor \
    --repo="https://dev.azure.com/org/proj/_git/repo" \
    --branch=main --mode=local

# With config file
python -m backend.processor.code_processor --config=run.json

# Docker mode (builds Dockerfile.processor)
python -m backend.processor.code_processor \
    --repo="https://dev.azure.com/org/proj/_git/repo" \
    --branch=main --mode=docker

# Cloud mode (creates AI Search + AML pipeline)
python -m backend.processor.code_processor \
    --repo="https://dev.azure.com/org/proj/_git/repo" \
    --branch=main --mode=cloud
```

Config file format (`run.json`):
```json
{
  "repo": "https://dev.azure.com/org/proj/_git/repo",
  "branch": "main",
  "mode": "local",
  "language": "en"
}
```

CLI args override config file values.

## Dependencies

- **Invokes:** `repository/git_ops` (clone), `embedder/` (RAG), `wiki/cache` (save), `clients/azureai_client` (LLM), `clients/search_client` (AI Search), `promptstore/` (templates)
- **Invoked by:** CLI directly, Docker container, AML pipeline job
