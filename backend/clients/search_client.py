"""
Azure AI Search client for DeepWiki cloud mode.

Manages per-repo search indexes, data sources, and indexers.
Each repo+branch gets its own index: "deepwiki-{owner}-{repo}-{branch}".

Resources created per repo:
    Index:       deepwiki-{owner}-{repo}-{branch}
    Data source: deepwiki-{owner}-{repo}-{branch}-datasource
    Indexer:     deepwiki-{owner}-{repo}-{branch}-indexer
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from backend.config import get_search_config, get_infra_config

logger = logging.getLogger(__name__)


def _sanitize_index_name(name: str) -> str:
    """Sanitize a string for use in AI Search index/datasource names.

    Rules: lowercase, alphanumeric + hyphens only, max 128 chars.
    """
    sanitized = re.sub(r'[^a-z0-9-]', '-', name.lower())
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    return sanitized[:128]


def _sanitize_document_key(key: str) -> str:
    """Sanitize a string for use as an AI Search document key.

    Keys can only contain letters, digits, underscore (_), dash (-),
    or equal sign (=). Replace anything else with underscore.
    """
    return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)


def get_index_name(owner: str, repo: str, branch: str) -> str:
    """Derive AI Search index name for a repo+branch."""
    return _sanitize_index_name(f"deepwiki-{owner}-{repo}-{branch}")


def _get_search_credential():
    """Get Azure credential for AI Search.

    If managed_identity.client_id is configured, use MSI.
    Otherwise fall back to Azure CLI (local dev), skipping MSI
    to avoid noisy timeouts on machines without managed identity.
    """
    from azure.identity import DefaultAzureCredential
    from backend.config import get_infra_config

    infra = get_infra_config()
    client_id = infra.managed_identity.client_id

    if client_id:
        return DefaultAzureCredential(managed_identity_client_id=client_id)

    return DefaultAzureCredential(exclude_managed_identity_credential=True)


def _get_search_client():
    """Create an authenticated SearchIndexClient."""
    from azure.search.documents.indexes import SearchIndexClient

    config = get_search_config()
    credential = _get_search_credential()
    return SearchIndexClient(
        endpoint=config.endpoint,
        credential=credential,
    )


def _get_search_documents_client(index_name: str):
    """Create an authenticated SearchClient for a specific index."""
    from azure.search.documents import SearchClient

    config = get_search_config()
    credential = _get_search_credential()
    return SearchClient(
        endpoint=config.endpoint,
        index_name=index_name,
        credential=credential,
    )


def _get_indexer_client():
    """Create an authenticated SearchIndexerClient."""
    from azure.search.documents.indexes import SearchIndexerClient

    config = get_search_config()
    credential = _get_search_credential()
    return SearchIndexerClient(
        endpoint=config.endpoint,
        credential=credential,
    )


def _build_blob_connection_string() -> str:
    """Build ResourceId-format connection string for blob data source.

    Uses account.subscription_id, account.resource_group, and
    azure_blob_storage.account_name from infra.json.
    """
    infra = get_infra_config()
    return (
        f"ResourceId=/subscriptions/{infra.account.subscription_id}"
        f"/resourceGroups/{infra.account.resource_group}"
        f"/providers/Microsoft.Storage"
        f"/storageAccounts/{infra.azure_blob_storage.account_name};"
    )


def _load_index_schema() -> dict:
    """Load index schema from backend/processor/code_index_schema.json."""
    schema_path = (
        Path(__file__).parent.parent
        / 'processor' / 'code_index_schema.json'
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


def _is_payload_too_large(e: Exception) -> bool:
    """Check if exception indicates Azure Search payload too large (413)."""
    err_str = str(e)
    return (
        '413' in err_str
        or 'Too Large' in err_str
        or (isinstance(e, KeyError) and 'error_map' in err_str)
    )


def push_documents(
    index_name: str,
    documents: list,
    repo_name: str,
    branch: str,
) -> int:
    """Push vector documents to AI Search index.

    Uses adaptive batch sizing: starts at 500, halves on 413 errors.
    Once a smaller size succeeds, all remaining batches use that size.

    Args:
        index_name: Target index name
        documents: List of adalflow Document objects (with text, vector, meta_data)
        repo_name: Repository identifier (owner_repo)
        branch: Branch name

    Returns:
        Number of documents pushed
    """
    client = _get_search_documents_client(index_name)

    # Build search docs
    search_docs = []
    for i, doc in enumerate(documents):
        meta = doc.meta_data or {}
        search_doc = {
            "id": _sanitize_document_key(f"{repo_name}_{branch}_{i}"),
            "title": meta.get('file_path', ''),
            "filepath": meta.get('file_path', ''),
            "content": doc.text or '',
            "raw_content": (
                doc.text[meta['_header_len']:]
                if '_header_len' in meta
                else meta.get('raw_chunk_text', doc.text or '')
            ),
            "service_id": repo_name,
            "content_vector": doc.vector if doc.vector else [],
        }
        search_docs.append(search_doc)

    # Adaptive batch upload: start large, halve on 413 errors
    batch_size = 500
    offset = 0
    total = len(search_docs)

    while offset < total:
        batch = search_docs[offset:offset + batch_size]
        try:
            client.upload_documents(documents=batch)
            offset += len(batch)
        except Exception as e:
            if _is_payload_too_large(e) and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    f"Payload too large for {index_name}, "
                    f"reducing batch size to {batch_size}"
                )
                # Don't advance offset — retry same docs with smaller batch
            else:
                raise

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


def search_as_documents(
    index_name: str,
    query: str,
    top_k: int = 40,
    vector: Optional[List[float]] = None,
    filter_expr: Optional[str] = None,
) -> list:
    """Hybrid search returning adalflow Document objects.

    Wraps search() and converts results to Document objects compatible
    with the RAG pipeline (same shape as FAISS retrieval output).

    Args:
        index_name: Index to search
        query: Text query
        top_k: Number of results
        vector: Optional query embedding vector (3072-dim)
        filter_expr: Optional OData filter (e.g. "filepath eq 'src/main.py'")

    Returns:
        List of adalflow Document objects with text, vector=None, meta_data
    """
    from azure.search.documents.models import VectorizedQuery
    from adalflow.core.types import Document

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
        filter=filter_expr,
        top=top_k,
        select=["id", "title", "filepath", "content", "raw_content"],
    )

    docs = []
    for result in results:
        meta = {
            'file_path': result.get('filepath', ''),
            'raw_content': result.get('raw_content', ''),
            'search_score': result.get('@search.score', 0),
        }
        doc = Document(
            text=result.get('content', ''),
            meta_data=meta,
        )
        docs.append(doc)

    return docs


def create_data_source(
    index_name: str, repo_name: str, branch: str
) -> str:
    """Create or update blob data source for a repo's vectors.

    Points to the blob folder: vectors/{repo_name}_{branch}/
    inside the configured blob container (e.g. deepwiki-data).

    Args:
        index_name: Associated index name (used to derive data source name)
        repo_name: Repository name (owner_repo format)
        branch: Branch name

    Returns:
        Data source name
    """
    from azure.search.documents.indexes.models import (
        SearchIndexerDataSourceConnection,
        SearchIndexerDataContainer,
    )

    infra = get_infra_config()
    ds_name = f"{index_name}-datasource"
    branch_suffix = branch.strip() if branch and branch.strip() else 'main'
    blob_folder = f"vectors/{repo_name}_{branch_suffix}"

    container = SearchIndexerDataContainer(
        name=infra.azure_blob_storage.container_name,
        query=blob_folder,
    )

    connection = SearchIndexerDataSourceConnection(
        name=ds_name,
        type="azureblob",
        connection_string=_build_blob_connection_string(),
        container=container,
    )

    client = _get_indexer_client()
    client.create_or_update_data_source_connection(connection)
    logger.info(f"Created/updated data source: {ds_name} → {blob_folder}")
    return ds_name


def create_indexer(
    index_name: str,
    data_source_name: str,
    interval: str = "PT24H",
) -> str:
    """Create or update a scheduled indexer that syncs blob → index.

    Args:
        index_name: Target search index name
        data_source_name: Data source to read from
        interval: ISO 8601 schedule interval (default PT24H = daily)

    Returns:
        Indexer name
    """
    from azure.search.documents.indexes.models import (
        SearchIndexer,
        IndexingSchedule,
        IndexingParameters,
        FieldMapping,
    )

    indexer_name = f"{index_name}-indexer"

    # Field mappings: map JSON fields from blob to index schema
    field_mappings = [
        FieldMapping(
            source_field_name="file_path",
            target_field_name="filepath",
        ),
        FieldMapping(
            source_field_name="text",
            target_field_name="content",
        ),
    ]

    indexer = SearchIndexer(
        name=indexer_name,
        description="DeepWiki vector indexer — syncs JSON vectors from blob",
        target_index_name=index_name,
        data_source_name=data_source_name,
        schedule=IndexingSchedule(interval=interval),
        parameters=IndexingParameters(
            batch_size=10,
            max_failed_items=-1,
            configuration={
                "dataToExtract": "contentAndMetadata",
                "parsingMode": "json",
            },
        ),
        field_mappings=field_mappings,
    )

    client = _get_indexer_client()
    client.create_or_update_indexer(indexer)
    logger.info(
        f"Created/updated indexer: {indexer_name} "
        f"(schedule={interval})"
    )
    return indexer_name


def run_indexer(index_name: str) -> None:
    """Manually trigger an indexer run for immediate sync.

    Call this after the processor writes vectors to blob
    so they become searchable without waiting for the next schedule.
    """
    indexer_name = f"{index_name}-indexer"
    client = _get_indexer_client()
    try:
        client.run_indexer(indexer_name)
        logger.info(f"Triggered indexer run: {indexer_name}")
    except Exception as e:
        logger.warning(f"Could not trigger indexer {indexer_name}: {e}")


def wait_for_indexer(
    index_name: str,
    timeout_seconds: int = 300,
    poll_interval: int = 10,
) -> bool:
    """Wait for the indexer to complete its current run.

    Polls indexer status until success, failure, or timeout.

    Args:
        index_name: Index name (indexer name derived as {index_name}-indexer)
        timeout_seconds: Maximum wait time (default 5 minutes)
        poll_interval: Seconds between status checks (default 10s)

    Returns:
        True if indexer completed successfully, False on timeout/failure
    """
    import time

    indexer_name = f"{index_name}-indexer"
    client = _get_indexer_client()
    start = time.time()

    while time.time() - start < timeout_seconds:
        try:
            status = client.get_indexer_status(indexer_name)
            last_result = status.last_result
            if last_result is None:
                logger.debug(
                    f"Indexer {indexer_name}: no run yet, waiting..."
                )
                time.sleep(poll_interval)
                continue

            run_status = str(last_result.status)
            if run_status in ("success", "Success"):
                elapsed = time.time() - start
                item_count = getattr(last_result, 'item_count', '?')
                logger.info(
                    f"Indexer {indexer_name} completed in "
                    f"{elapsed:.0f}s ({item_count} items indexed)"
                )
                return True
            elif run_status in ("inProgress", "InProgress"):
                logger.debug(
                    f"Indexer {indexer_name}: in progress, "
                    f"waiting {poll_interval}s..."
                )
                time.sleep(poll_interval)
            else:
                errors = getattr(last_result, 'errors', '')
                logger.warning(
                    f"Indexer {indexer_name} status: {run_status}"
                    f" — {errors}"
                )
                return False
        except Exception as e:
            logger.warning(
                f"Error polling indexer {indexer_name}: {e}, "
                f"retrying..."
            )
            time.sleep(poll_interval)

    logger.error(
        f"Indexer {indexer_name} timed out after {timeout_seconds}s"
    )
    return False


def delete_indexer(index_name: str) -> None:
    """Delete indexer and data source for an index."""
    indexer_name = f"{index_name}-indexer"
    ds_name = f"{index_name}-datasource"
    client = _get_indexer_client()
    try:
        client.delete_indexer(indexer_name)
        logger.info(f"Deleted indexer: {indexer_name}")
    except Exception as e:
        logger.warning(f"Could not delete indexer {indexer_name}: {e}")
    try:
        client.delete_data_source_connection(ds_name)
        logger.info(f"Deleted data source: {ds_name}")
    except Exception as e:
        logger.warning(f"Could not delete data source {ds_name}: {e}")


def delete_index(index_name: str) -> None:
    """Delete an AI Search index."""
    client = _get_search_client()
    try:
        client.delete_index(index_name)
        logger.info(f"Deleted AI Search index: {index_name}")
    except Exception as e:
        logger.warning(f"Could not delete index {index_name}: {e}")


def index_exists(index_name: str) -> bool:
    """Check if an AI Search index exists.

    Only returns False for actual 404 (not found). Auth errors and
    other failures are raised so they aren't silently masked as
    'index not found'.
    """
    from azure.core.exceptions import ResourceNotFoundError

    client = _get_search_client()
    try:
        client.get_index(index_name)
        return True
    except ResourceNotFoundError:
        return False
