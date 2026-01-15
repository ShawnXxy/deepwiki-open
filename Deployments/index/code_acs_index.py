"""
Utility for Azure Cognitive Search index (Code repositories).

Copyright (c) Microsoft Corporation.  All rights reserved.
"""

import time

from azure.search.documents.indexes.models import (
    IndexingSchedule,
    SearchIndexer,
    SearchIndex,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)
from .acs_index_utils import (
    wait_for_indexer_completion,
    get_fields_from_json,
    IndexManager
)


class CodeIndexManager(IndexManager):
    """Code Index Manager for code repository indexing."""

    def __init__(self, blob_container, blob_connection_string, service_endpoint,
                 credential, index_schema, index_name='code-acs-index',
                 index_blob_folder=''):
        """
        Initialize Code Index Manager.

        Parameters:
            blob_container: Azure blob container name
            blob_connection_string: Connection string (ResourceId format)
            service_endpoint: Azure Cognitive Search endpoint
            credential: Azure credential
            index_schema: Index schema dictionary from JSON
            index_name: Name of the search index
            index_blob_folder: Folder path in blob (empty for root)
        """
        super().__init__(
            index_name=index_name,
            index_blob_folder=index_blob_folder,
            blob_container=blob_container,
            blob_connection_string=blob_connection_string,
            service_endpoint=service_endpoint,
            credential=credential
        )
        self.indexer_name = f"{self.index_name}-indexer"
        self.fields, self.field_mappings = get_fields_from_json(index_schema)
        self.semanticConfiguration = index_schema['semanticConfiguration']
        self.prioritized_fields = index_schema['prioritized_fields']

    def create_code_search_index(self):
        """Create or update the search index."""
        semantic_config = SemanticConfiguration(
            name=self.semanticConfiguration,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(
                    field_name=self.prioritized_fields['title_field']
                ),
                content_fields=[
                    SemanticField(
                        field_name=self.prioritized_fields[
                            'prioritized_content_fields'
                        ]
                    )
                ]
            )
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        index = SearchIndex(
            name=self.index_name,
            fields=self.fields,
            vector_search=self.vector_search,
            semantic_search=semantic_search
        )

        result = self.index_client.create_or_update_index(index)
        print(f"✓ Index '{result.name}' created or updated")
        return result

    def create_code_search_indexer(self, interval='PT4H'):
        """
        Create or update the search indexer.

        Parameters:
            interval: Indexing schedule interval (ISO 8601 duration)
                     PT4H = every 4 hours, PT24H = daily
        """
        indexer = SearchIndexer(
            name=self.indexer_name,
            description="Indexer for code repository documents",
            target_index_name=self.index_name,
            data_source_name=self.data_source.name,
            parameters=self.index_parameters,
            schedule=IndexingSchedule(interval=interval),
            field_mappings=self.field_mappings
        )

        result = self.indexer_client.create_or_update_indexer(indexer)
        print(f"✓ Indexer '{self.indexer_name}' created or updated")

        # Wait for initial indexing to complete
        wait_for_indexer_completion(self.indexer_client, self.indexer_name)
        return result

    def create_code_index(self, interval='PT24H', recreate=False):
        """
        Create complete code index with data source and indexer.

        Parameters:
            interval: Indexing schedule interval
            recreate: If True, delete existing index before creating
        """
        print("=" * 60)
        print(f"Creating Code Index: {self.index_name}")
        print("=" * 60)

        if recreate:
            self.delete_index()

        self.create_data_source()
        self.create_code_search_index()
        self.create_code_search_indexer(interval=interval)

        print(f"\n✓ Code index setup complete")
        print(f"  - Index: {self.index_name}")
        print(f"  - Indexer: {self.indexer_name}")
        print(f"  - Schedule: {interval}")

