"""
Azure AI Search client for DeepWiki cloud mode.

Manages per-repo search indexes: create, push documents, query, delete.
Each repo+branch gets its own index: "deepwiki-{owner}-{repo}-{branch}".
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from backend.config import get_search_config

logger = logging.getLogger(__name__)


def _sanitize_index_name(name: str) -> str:
    """Sanitize a string for use in AI Search index/datasource names.

    Rules: lowercase, alphanumeric + hyphens only, max 128 chars.
    """
    sanitized = re.sub(r'[^a-z0-9-]', '-', name.lower())
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    return sanitized[:128]


def get_index_name(owner: str, repo: str, branch: str) -> str:
    """Derive AI Search index name for a repo+branch."""
    return _sanitize_index_name(f"deepwiki-{owner}-{repo}-{branch}")


def _get_search_client():
    """Create an authenticated SearchIndexClient."""
    from azure.search.documents.indexes import SearchIndexClient
    from azure.identity import DefaultAzureCredential

    config = get_search_config()
    credential = DefaultAzureCredential()
    return SearchIndexClient(
        endpoint=config.endpoint,
        credential=credential,
    )


def _get_search_documents_client(index_name: str):
    """Create an authenticated SearchClient for a specific index."""
    from azure.search.documents import SearchClient
    from azure.identity import DefaultAzureCredential

    config = get_search_config()
    credential = DefaultAzureCredential()
    return SearchClient(
        endpoint=config.endpoint,
        index_name=index_name,
        credential=credential,
    )


def _load_index_schema() -> dict:
    """Load index schema from Deployments/index/code_index_schema.json."""
    schema_path = (
        Path(__file__).parent.parent.parent
        / 'Deployments' / 'index' / 'code_index_schema.json'
    )
    if not schema_path.exists():
        raise FileNotFoundError(f"Index schema not found at {schema_path}")
    with open(schema_path, 'r') as f:
        return json.load(f)


def create_or_update_index(index_name: str) -> None:
    """Create or update an AI Search index with the DeepWiki schema."""
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile,
        SemanticConfiguration,
        SemanticSearch,
        SemanticPrioritizedFields,
        SemanticField,
    )

    schema = _load_index_schema()

    # Build fields from schema
    fields = []
    for field_def in schema.get('fields', []):
        name = field_def['name']
        field_type = field_def.get('field_type', 'SimpleField')

        if 'vector' in name.lower() and field_def.get('vector_search_dimensions'):
            fields.append(
                SearchField(
                    name=name,
                    type=SearchFieldDataType.Collection(
                        SearchFieldDataType.Single
                    ),
                    searchable=True,
                    vector_search_dimensions=field_def['vector_search_dimensions'],
                    vector_search_profile_name="myHnswProfile",
                )
            )
        elif field_type == 'SearchableField':
            fields.append(
                SearchableField(
                    name=name,
                    type=field_def.get('type', 'Edm.String'),
                    filterable=field_def.get('filterable', False),
                    key=field_def.get('key', False),
                )
            )
        else:
            fields.append(
                SimpleField(
                    name=name,
                    type=field_def.get('type', 'Edm.String'),
                    filterable=field_def.get('filterable', False),
                    key=field_def.get('key', False),
                )
            )

    # Vector search config
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw"),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
        ],
    )

    # Semantic config
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    client = _get_search_client()
    client.create_or_update_index(index)
    logger.info(f"Created/updated AI Search index: {index_name}")


def push_documents(
    index_name: str,
    documents: list,
    repo_name: str,
    branch: str,
) -> int:
    """Push vector documents to AI Search index.

    Args:
        index_name: Target index name
        documents: List of adalflow Document objects (with text, vector, meta_data)
        repo_name: Repository identifier (owner_repo)
        branch: Branch name

    Returns:
        Number of documents pushed
    """
    client = _get_search_documents_client(index_name)

    batch = []
    for i, doc in enumerate(documents):
        meta = doc.meta_data or {}
        search_doc = {
            "id": f"{repo_name}_{branch}_{i}",
            "title": meta.get('file_path', ''),
            "filepath": meta.get('file_path', ''),
            "content": doc.text or '',
            "raw_content": meta.get('raw_content', ''),
            "service_id": repo_name,
            "content_vector": doc.vector if doc.vector else [],
        }
        batch.append(search_doc)

        # Upload in batches of 1000
        if len(batch) >= 1000:
            client.upload_documents(documents=batch)
            logger.info(
                f"Pushed batch of {len(batch)} docs to {index_name}"
            )
            batch = []

    # Final batch
    if batch:
        client.upload_documents(documents=batch)
        logger.info(f"Pushed final batch of {len(batch)} docs to {index_name}")

    total = len(documents)
    logger.info(f"Total {total} documents pushed to index {index_name}")
    return total


def search(
    index_name: str,
    query: str,
    top_k: int = 40,
    vector: Optional[List[float]] = None,
) -> list:
    """Search an AI Search index with hybrid (text + vector) query.

    Args:
        index_name: Index to search
        query: Text query
        top_k: Number of results
        vector: Optional query embedding vector (3072-dim)

    Returns:
        List of search result dicts with text, score, metadata
    """
    from azure.search.documents.models import VectorizedQuery

    client = _get_search_documents_client(index_name)

    vector_queries = []
    if vector:
        vector_queries.append(
            VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=top_k,
                fields="content_vector",
            )
        )

    results = client.search(
        search_text=query,
        vector_queries=vector_queries if vector_queries else None,
        top=top_k,
        select=["id", "title", "filepath", "content", "raw_content"],
    )

    docs = []
    for result in results:
        docs.append({
            'id': result['id'],
            'file_path': result.get('filepath', ''),
            'text': result.get('content', ''),
            'raw_content': result.get('raw_content', ''),
            'score': result['@search.score'],
        })

    return docs


def delete_index(index_name: str) -> None:
    """Delete an AI Search index."""
    client = _get_search_client()
    try:
        client.delete_index(index_name)
        logger.info(f"Deleted AI Search index: {index_name}")
    except Exception as e:
        logger.warning(f"Could not delete index {index_name}: {e}")


def index_exists(index_name: str) -> bool:
    """Check if an AI Search index exists."""
    client = _get_search_client()
    try:
        client.get_index(index_name)
        return True
    except Exception:
        return False
