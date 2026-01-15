"""
Utility for Azure Cognitive Search indexes.

Copyright (c) Microsoft Corporation.  All rights reserved.
"""

import time

from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    FieldMapping,
    SearchField,
    SearchableField,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    IndexingParameters
)


# Helpers
# -----------------------------------------------------------------------------


def wait_for_indexer_completion(indexer_client, indexer_name, timeout_seconds=600):
    """Poll status and wait until indexer is complete."""
    start_time = time.time()
    status = "not started"
    
    while time.time() - start_time < timeout_seconds:
        indexer_status = indexer_client.get_indexer_status(indexer_name)
        
        if indexer_status.last_result is None:
            print(f"Indexing status: waiting to start...")
            time.sleep(5)
            continue
            
        status = indexer_status.last_result.status
        print(f"Indexing status: {status}")
        
        if status == "success":
            print(f"✓ Indexer {indexer_name} completed successfully")
            return
        elif status == "transientFailure":
            print(f"⚠ Indexer {indexer_name} encountered transient failure, continuing...")
            return
        elif status == "inProgress":
            time.sleep(5)
        else:
            time.sleep(5)
    
    print(f"⚠ Indexer status check timed out after {timeout_seconds} seconds")


def get_fields_from_json(index_schema):
    """Parse fields from index schema JSON."""
    fields = []
    field_mappings = []

    def _get_field_type(type_str):
        """Convert string type to SearchFieldDataType."""
        if type_str == "SearchFieldDataType.Collection(SearchFieldDataType.Single)":
            return SearchFieldDataType.Collection(SearchFieldDataType.Single)
        else:
            return type_str

    if not index_schema.get('fields'):
        raise ValueError("No fields found in index schema")

    for item in index_schema['fields']:
        field_type_str = item.get('field_type', 'SearchField')
        type_value = _get_field_type(item['type'])
        
        # Add field mapping if source field is specified
        if item.get("source_field_name"):
            field_mappings.append(FieldMapping(
                target_field_name=item['name'], 
                source_field_name=item['source_field_name']
            ))
        
        # Create field based on field_type
        if field_type_str == "SimpleField":
            fields.append(SimpleField(
                name=item['name'],
                type=type_value,
                key=item.get('key', False),
                filterable=item.get('filterable', False),
                sortable=item.get('sortable', False),
                facetable=item.get('facetable', False)
            ))
        elif field_type_str == "SearchableField":
            fields.append(SearchableField(
                name=item['name'],
                type=type_value,
                searchable=item.get('searchable', True),
                filterable=item.get('filterable', False),
                sortable=item.get('sortable', False),
                facetable=item.get('facetable', False)
            ))
        elif field_type_str == "SearchField":
            # Vector field
            if 'vector_search_dimensions' in item:
                fields.append(SearchField(
                    name=item['name'],
                    type=type_value,
                    searchable=True,
                    vector_search_dimensions=item['vector_search_dimensions'],
                    vector_search_profile_name=item['vector_search_configuration']
                ))
            else:
                # Regular search field
                fields.append(SearchField(
                    name=item['name'],
                    type=type_value,
                    searchable=item.get('searchable', True),
                    filterable=item.get('filterable', False),
                    sortable=item.get('sortable', False),
                    facetable=item.get('facetable', False)
                ))
        else:
            raise ValueError(f"Unsupported field_type: {field_type_str}")
    
    return fields, field_mappings


# Main Index Manager
# -----------------------------------------------------------------------------


class IndexManager:
    """Base index manager class for Azure Cognitive Search."""

    def __init__(self, index_name, index_blob_folder, blob_container, blob_connection_string, 
                 service_endpoint, credential, batch_size=10):
        """
        Initialize index manager.

        Parameters:
            index_name: Name of the search index
            index_blob_folder: Folder path in blob container for indexing
            blob_container: Azure blob container name
            blob_connection_string: Connection string for blob storage (use ResourceId format for managed identity)
            service_endpoint: Azure Cognitive Search endpoint URL
            credential: Azure credential for authentication
            batch_size: Number of items indexed in a batch
        """
        self.index_name = index_name
        self.index_blob_folder = index_blob_folder
        self.blob_container = blob_container
        self.blob_connection_string = blob_connection_string
        self.service_endpoint = service_endpoint
        self.credential = credential
        
        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=self.service_endpoint, 
            credential=self.credential
        )
        self.indexer_client = SearchIndexerClient(
            endpoint=self.service_endpoint, 
            credential=self.credential
        )
        
        # Configure vector search with HNSW algorithm
        self.vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw"
                )
            ]
        )
        
        # Configure indexing parameters
        self.indexer_config = {
            "dataToExtract": "contentAndMetadata",
            "parsingMode": "json"
        }
        self.index_parameters = IndexingParameters(
            batch_size=batch_size,
            max_failed_items=-1,
            configuration=self.indexer_config
        )

    def create_data_source(self, data_source_name=None):
        """Create or update data source connection."""
        from azure.search.documents.indexes.models import (
            SearchIndexerDataSourceConnection,
            SearchIndexerDataContainer
        )
        
        if data_source_name is None:
            data_source_name = f"{self.index_name}-datasource"
        
        container = SearchIndexerDataContainer(
            name=self.blob_container,
            query=self.index_blob_folder
        )
        
        data_source_connection = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type="azureblob",
            connection_string=self.blob_connection_string,
            container=container
        )
        
        self.data_source = self.indexer_client.create_or_update_data_source_connection(
            data_source_connection
        )
        print(f"✓ Data source '{self.data_source.name}' created or updated")
        return self.data_source

    def delete_index(self):
        """Delete the search index."""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"✓ Index '{self.index_name}' deleted")
        except Exception as e:
            print(f"⚠ Could not delete index '{self.index_name}': {e}")
    
    def get_index(self):
        """Get the search index."""
        try:
            return self.index_client.get_index(self.index_name)
        except Exception:
            return None
