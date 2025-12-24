#!/usr/bin/env python3
"""
Test suite for Azure OpenAI embedder.
This test file validates the Azure OpenAI embedder system.
"""

import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Simple test runner without pytest dependency."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def run_test(self, test_func, test_name=None):
        """Run a single test function."""
        if test_name is None:
            test_name = test_func.__name__
        
        self.tests_run += 1
        try:
            logger.info(f"Running test: {test_name}")
            test_func()
            self.tests_passed += 1
            logger.info(f"✅ {test_name} PASSED")
            return True
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
            logger.error(f"❌ {test_name} FAILED: {e}")
            return False
    
    def run_test_class(self, test_class):
        """Run all test methods in a test class."""
        instance = test_class()
        test_methods = [
            getattr(instance, method) 
            for method in dir(instance) 
            if method.startswith('test_') and callable(getattr(instance, method))
        ]
        
        for test_method in test_methods:
            test_name = f"{test_class.__name__}.{test_method.__name__}"
            self.run_test(test_method, test_name)
    
    def summary(self):
        """Print test summary."""
        logger.info("\n📊 Test Summary:")
        logger.info(f"Tests run: {self.tests_run}")
        logger.info(f"Passed: {self.tests_passed}")
        logger.info(f"Failed: {self.tests_failed}")
        
        if self.failures:
            logger.error("\n❌ Failed tests:")
            for test_name, error in self.failures:
                logger.error(f"  - {test_name}: {error}")
        
        return self.tests_failed == 0


class TestAzureConfiguration:
    """Test Azure OpenAI configuration system."""
    
    def test_config_loading(self):
        """Test that Azure configuration loads properly."""
        from api.config import configs, CLIENT_CLASSES
        
        # Check embedder configuration exists
        assert 'embedder' in configs, "Embedder config missing"
        
        # Check AzureAIClient is available
        assert 'AzureAIClient' in CLIENT_CLASSES, \
            "AzureAIClient missing from CLIENT_CLASSES"
    
    def test_embedder_type_detection(self):
        """Test embedder type detection returns azure."""
        from api.config import get_embedder_type, is_ollama_embedder
        
        # Type should be azure
        current_type = get_embedder_type()
        assert current_type == 'azure', f"Embedder type should be azure, got: {current_type}"
        
        # is_ollama_embedder should return False
        is_ollama = is_ollama_embedder()
        assert is_ollama is False, "is_ollama_embedder should return False"
    
    def test_azure_openai_configured(self):
        """Test Azure OpenAI configuration detection."""
        from api.config import is_azure_openai_configured
        
        is_configured = is_azure_openai_configured()
        assert isinstance(is_configured, bool), \
            "is_azure_openai_configured should return boolean"
        
        if not is_configured:
            logger.warning(
                "Azure OpenAI is not configured. "
                "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc."
            )
    
    def test_get_embedder_config(self):
        """Test getting embedder config."""
        from api.config import get_embedder_config
        
        config = get_embedder_config()
        assert isinstance(config, dict), "Config should be dict"
        assert 'model_kwargs' in config, "Config should have model_kwargs"


class TestEmbedderFactory:
    """Test the embedder factory function."""
    
    def test_get_embedder(self):
        """Test get_embedder creates Azure embedder."""
        from api.tools.embedder import get_embedder
        
        embedder = get_embedder()
        assert embedder is not None, "Embedder should be created"
    
    def test_get_embedder_with_legacy_params(self):
        """Test get_embedder ignores legacy parameters."""
        from api.tools.embedder import get_embedder
        
        # These parameters should be ignored (backward compatibility)
        embedder = get_embedder(embedder_type='ollama')
        assert embedder is not None, "Embedder should be created (params ignored)"


class TestAzureAIClient:
    """Test Azure AI client."""
    
    def test_client_initialization(self):
        """Test AzureAIClient can be initialized."""
        from api.azureai_client import AzureAIClient
        
        client = AzureAIClient()
        assert client is not None, "AzureAIClient should be created"
    
    def test_client_with_config(self):
        """Test AzureAIClient with embedding config."""
        from api.azureai_client import AzureAIClient
        from api.config import get_azure_openai_embedding_config
        
        config = get_azure_openai_embedding_config()
        client = AzureAIClient(**config)
        assert client is not None, "AzureAIClient should be created with config"


class TestDataPipelineFunctions:
    """Test data pipeline functions."""
    
    def test_count_tokens(self):
        """Test token counting function."""
        from api.data_pipeline import count_tokens
        
        test_text = "This is a test string for token counting."
        
        # Test basic counting
        token_count = count_tokens(test_text)
        assert isinstance(token_count, int), "Token count should be an integer"
        assert token_count > 0, "Token count should be positive"
        
        # Test with legacy parameters (should be ignored)
        token_count_2 = count_tokens(test_text, is_ollama_embedder=True)
        assert isinstance(token_count_2, int), "Token count should be an integer"
    
    def test_prepare_data_pipeline(self):
        """Test data pipeline preparation."""
        from api.data_pipeline import prepare_data_pipeline
        
        pipeline = prepare_data_pipeline()
        assert pipeline is not None, "Data pipeline should be created"


class TestRAGComponent:
    """Test RAG component."""
    
    def test_rag_initialization(self):
        """Test RAG component can be initialized."""
        from api.rag import RAG
        
        rag = RAG()
        assert rag is not None, "RAG should be created"
        assert rag.provider == "azure", "RAG provider should be azure"
        assert rag.is_ollama_embedder is False, "is_ollama_embedder should be False"
    
    def test_rag_with_model(self):
        """Test RAG component with model parameter."""
        from api.rag import RAG
        
        rag = RAG(model="gpt-4o")
        assert rag is not None, "RAG should be created"
        assert rag.model == "gpt-4o", "Model should be set"


def run_all_tests():
    """Run all test classes."""
    runner = TestRunner()
    
    test_classes = [
        TestAzureConfiguration,
        TestEmbedderFactory,
        TestAzureAIClient,
        TestDataPipelineFunctions,
        TestRAGComponent,
    ]
    
    for test_class in test_classes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running {test_class.__name__}")
        logger.info('=' * 60)
        runner.run_test_class(test_class)
    
    return runner.summary()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
