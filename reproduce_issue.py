
import logging
import os
import shutil
import sys
from unittest.mock import MagicMock, patch
from adalflow.core import ModelClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class MockAzureAIClient(ModelClient):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.sync_client = MagicMock()
        # Mock embeddings.create
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1]*3072)]
        self.sync_client.embeddings.create.return_value = mock_response

    def call(self, api_kwargs={}, model_type=None):
        # Return mock response for embeddings
        if model_type == 1: # ModelType.EMBEDDER (assuming enum value) or check kwargs
             return self.sync_client.embeddings.create(**api_kwargs)
        return MagicMock()

    def convert_inputs_to_api_kwargs(self, input, model_kwargs, model_type):
        return {"input": input, **model_kwargs}

# Mock AzureAIClient before importing api modules
sys.modules["api.azureai_client"] = MagicMock()
sys.modules["api.azureai_client"].AzureAIClient = MockAzureAIClient


# Mock get_embedder to return a mock embedder
# We don't need to mock get_embedder if we mock AzureAIClient correctly, 
# but let's keep it simple and let RAG use the mocked client.
# So I will remove the get_embedder mock.

# Mock the response from embeddings.create
# (Removed)

from api.rag import RAG
# from api.data_pipeline import DataPipeline

def main():
    logger.info("Starting reproduction script")
    
    # Create a dummy repo
    repo_path = os.path.abspath("test_repo")
    if not os.path.exists(repo_path):
        os.makedirs(repo_path)
        with open(os.path.join(repo_path, "test.txt"), "w") as f:
            f.write("This is a test document for embedding.")

    # Clean up cache
    cache_path = os.path.join(os.environ.get("APPDATA", ""), "adalflow", "databases", "test_repo.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Removing cache file: {cache_path}")
        os.remove(cache_path)

    try:
        rag = RAG()
        logger.info(f"Preparing retriever for {repo_path}")
        rag.prepare_retriever(repo_path)
        logger.info("Retriever prepared successfully")
    except Exception as e:
        logger.error(f"Error preparing retriever: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
