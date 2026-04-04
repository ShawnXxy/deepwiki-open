"""
Configuration module for DeepWiki using type-safe classes.
Provides type-safe access to configuration files using Pydantic models.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

from backend.types.config_types import (
    InfraConfig,
    EmbedderConfig,
    GeneratorConfig,
    FileFiltersConfig,
    IncludedConfig,
    LanguageConfig,
    AzureAccountConfig,
    AzureAISearchConfig,
    AzureMLConfig,
)
from backend.types.converter import from_dict

logger = logging.getLogger(__name__)

# ============================================================================
# Cached Configuration Objects
# ============================================================================

_infra_config: Optional[InfraConfig] = None
_embedder_config: Optional[EmbedderConfig] = None
_generator_config: Optional[GeneratorConfig] = None
_file_filters_config: Optional[FileFiltersConfig] = None
_included_config: Optional[IncludedConfig] = None
_lang_config: Optional[LanguageConfig] = None
_client_classes: Optional[Dict[str, Any]] = None
_azure_ai_client: Optional[Any] = None

# ============================================================================
# Environment Configuration
# ============================================================================

# Wiki authentication settings
raw_auth_mode = os.environ.get('DEEPWIKI_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']
WIKI_AUTH_CODE = os.environ.get('DEEPWIKI_AUTH_CODE', '')

# Configuration directory — set via set_config_dir() or env var
CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', None)


def set_config_dir(path: str) -> None:
    """Override the configuration directory at runtime.

    Called by ``main()`` when ``--mode cloud`` to switch to
    ``backend/config/.cloud/`` which has Azure services force-enabled.

    Clears all cached configs so they reload from the new directory.
    """
    global CONFIG_DIR
    global _infra_config, _embedder_config, _generator_config
    global _file_filters_config, _included_config, _lang_config
    global _client_classes, _azure_ai_client

    CONFIG_DIR = path

    # Clear all cached configs — next access reloads from new dir
    _infra_config = None
    _embedder_config = None
    _generator_config = None
    _file_filters_config = None
    _included_config = None
    _lang_config = None
    _client_classes = None
    _azure_ai_client = None


# ============================================================================
# Utility Functions
# ============================================================================

def replace_env_placeholders(
    config: Any
) -> Any:
    """
    Recursively replace placeholders like "${ENV_VAR}" in string values
    within a nested configuration structure (dicts, lists, strings)
    with environment variable values.
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"Environment variable placeholder '{original_placeholder}' not found. "
                f"The placeholder string will be used as is."
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        return config


def load_json_config(filename: str) -> Dict[str, Any]:
    """Load JSON configuration file from config directory.

    Resolution order:
    1. ``CONFIG_DIR`` (set via ``set_config_dir()`` or env var)
    2. ``backend/config/`` (default)
    """
    try:
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            config_path = Path(__file__).parent / "config" / filename

        logger.debug(f"Loading configuration from {config_path}")

        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} does not exist")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config = replace_env_placeholders(config)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration file {filename}: {str(e)}")
        return {}


# ============================================================================
# Infrastructure Configuration Loaders
# ============================================================================

def get_infra_config() -> InfraConfig:
    """
    Get the infrastructure configuration from infra.json.
    Loads the config on first access and caches it.
    
    Returns:
        InfraConfig object with type-safe access to configuration
    """
    global _infra_config
    if _infra_config is None:
        config_dict = load_json_config("infra.json")
        if config_dict:
            try:
                _infra_config = from_dict(InfraConfig, config_dict)
                logger.info("Successfully loaded and validated infra.json")
            except Exception as e:
                logger.error(f"Failed to parse infra.json: {e}")
                raise
        else:
            raise ValueError("infra.json is required but could not be loaded")
    return _infra_config


def get_managed_identity_client_id() -> Optional[str]:
    """Get the managed identity client ID from infra.json."""
    infra = get_infra_config()
    return infra.managed_identity.client_id


def get_account_config() -> AzureAccountConfig:
    """Get Azure account info (subscription_id, resource_group) from infra.json."""
    infra = get_infra_config()
    return infra.account


def enable_cloud_services() -> None:
    """Force-enable cloud services (AI Search, AML, Blob) for cloud mode.

    Called by _run_cloud_mode() so that cloud resources are created
    even when infra.json has enabled=false (the default for local dev).
    Mutates the cached InfraConfig in place.
    """
    infra = get_infra_config()
    infra.azure_ai_search.enabled = True
    infra.azure_blob_storage.enabled = True
    infra.azure_ml.enabled = True
    logger.info("Cloud services force-enabled: AI Search, Blob, AML")


def get_azure_openai_config(task: str = 'chat') -> Dict[str, str]:
    """
    Get Azure OpenAI configuration for a specific task.

    Args:
        task: 'chat' or 'reasoning'

    Returns:
        Dict containing endpoint, api_version, deployment
    """
    infra = get_infra_config()
    section = infra.azure_openai.chat if task == 'chat' else infra.azure_openai.reasoning
    return {
        "endpoint": section.endpoint,
        "api_version": section.api_version,
        "deployment": section.deployment
    }


def get_azure_openai_embedding_config_from_infra() -> Dict[str, str]:
    """
    Get Azure OpenAI embedding configuration from infra.json.
    
    Returns:
        Dict containing endpoint, api_version, deployment
    """
    infra = get_infra_config()
    return {
        "endpoint": infra.azure_openai.embedding.endpoint,
        "api_version": infra.azure_openai.embedding.api_version,
        "deployment": infra.azure_openai.embedding.deployment
    }


def is_azure_openai_configured() -> bool:
    """
    Check if Azure OpenAI is configured in infra.json.
    
    Returns:
        bool: True if Azure OpenAI is properly configured
    """
    try:
        infra = get_infra_config()
        
        # Check for basic Azure OpenAI configuration
        has_basic_config = bool(
            infra.azure_openai.chat.endpoint and 
            infra.azure_openai.chat.api_version
        )
        
        # Check for Azure endpoint pattern
        has_azure_pattern = ".openai.azure.com" in infra.azure_openai.chat.endpoint
        
        # Check for API key in environment (for local dev)
        has_api_key = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
        
        # Check for MSI client ID in infra.json (for production)
        has_msi_config = bool(infra.managed_identity.client_id)
        
        # Either API key or MSI config is sufficient
        return has_basic_config and has_azure_pattern and (has_api_key or has_msi_config)
    except Exception:
        return False


def get_azure_openai_text_config(task: str = 'chat') -> Dict[str, Any]:
    """
    Get Azure OpenAI configuration for text generation with MSI authentication.

    Args:
        task: 'chat' or 'reasoning'

    Returns:
        Dict containing azure_endpoint, api_version, and managed_identity_client_id
    """
    infra = get_infra_config()
    section = infra.azure_openai.chat if task == 'chat' else infra.azure_openai.reasoning
    return {
        "azure_endpoint": section.endpoint,
        "api_version": section.api_version,
        "managed_identity_client_id": infra.managed_identity.client_id
    }


def get_azure_deployment_name(task: str = 'chat', model_name: str = None) -> str:
    """
    Get the Azure OpenAI deployment name for a specific task.

    Args:
        task: 'chat' or 'reasoning'
        model_name: Ignored - always uses infra.json deployment for the task
    
    Returns:
        The deployment name from infra.json for the specified task
    """
    infra = get_infra_config()
    section = infra.azure_openai.chat if task == 'chat' else infra.azure_openai.reasoning
    deployment = section.deployment
    if model_name and model_name != deployment:
        logger.info(f"Using {task} deployment '{deployment}' (ignoring '{model_name}')")
    return deployment


def get_azure_openai_embedding_config() -> Dict[str, Any]:
    """
    Get Azure OpenAI configuration for embeddings with MSI authentication.
    
    Returns:
        Dict containing azure_endpoint, api_version, and managed_identity_client_id
    """
    infra = get_infra_config()
    return {
        "azure_endpoint": infra.azure_openai.embedding.endpoint,
        "api_version": infra.azure_openai.embedding.api_version,
        "managed_identity_client_id": infra.managed_identity.client_id
    }


# ============================================================================
# Embedder Configuration
# ============================================================================

def get_embedder_config_obj() -> EmbedderConfig:
    """
    Get the complete embedder configuration.
    Loads on first access, injects values from infra config, and caches.
    """
    global _embedder_config
    if _embedder_config is None:
        config_dict = load_json_config("embedder.json")
        if config_dict:
            try:
                _embedder_config = from_dict(EmbedderConfig, config_dict)
                
                # Inject values from infra.json
                infra = get_infra_config()
                _embedder_config.embedder.model_kwargs.model = infra.azure_openai.embedding.deployment
                _embedder_config.embedder.model_kwargs.dimensions = infra.azure_openai.embedding.dimensions
                _embedder_config.embedder.initialize_kwargs = get_azure_openai_embedding_config()
                
                logger.info("Successfully loaded and validated embedder.json")
            except Exception as e:
                logger.error(f"Failed to parse embedder.json: {e}")
                raise
    return _embedder_config


def get_embedder_config() -> Dict[str, Any]:
    """Get the embedder configuration as a dictionary (for backward compatibility)."""
    config = get_embedder_config_obj()
    # Exclude None values to prevent passing invalid params to embedder (e.g., temperature=None)
    return config.embedder.model_dump(exclude_none=True)


def get_retriever_config() -> Dict[str, Any]:
    """Get the retriever configuration."""
    config = get_embedder_config_obj()
    return config.retriever.model_dump()


def get_text_splitter_config() -> Dict[str, Any]:
    """Get the text splitter configuration."""
    config = get_embedder_config_obj()
    # Exclude None values to avoid passing separators=None to TextSplitter
    return config.text_splitter.model_dump(exclude_none=True)


# ============================================================================
# Generator Configuration
# ============================================================================

def get_generator_full_config() -> GeneratorConfig:
    """
    Get the complete generator configuration.
    Built from infra.json (no separate generator.json needed).
    """
    global _generator_config
    if _generator_config is None:
        infra = get_infra_config()
        _generator_config = GeneratorConfig(
            client_class="AzureAIClient",
        )
        _generator_config.model_kwargs.model = infra.azure_openai.chat.deployment
        _generator_config.model_kwargs.temperature = infra.azure_openai.chat.temperature
        _generator_config.initialize_kwargs = get_azure_openai_text_config('chat')
        logger.info("Generator config built from infra.json")
    return _generator_config


def get_generator_config() -> Dict[str, Any]:
    """Get the generator configuration as a dictionary (for backward compatibility)."""
    config = get_generator_full_config()
    # Exclude None values to keep the dict clean
    return config.model_dump(exclude_none=True)


# ============================================================================
# Repository Configuration
# ============================================================================

def get_file_filters_config_obj() -> FileFiltersConfig:
    """
    Get the file filters configuration.
    Loads on first access and caches.
    """
    global _file_filters_config
    if _file_filters_config is None:
        config_dict = load_json_config("excluded.json")
        if config_dict:
            try:
                file_filters_data = config_dict.get("file_filters", {})
                _file_filters_config = from_dict(FileFiltersConfig, file_filters_data)
                logger.info("Successfully loaded and validated file_filters from excluded.json")
            except Exception as e:
                logger.error(f"Failed to parse file_filters from excluded.json: {e}")
                raise
    return _file_filters_config


def get_file_filters_config() -> Dict[str, Any]:
    """Get file filters configuration."""
    config = get_file_filters_config_obj()
    return {
        "excluded_dirs": config.excluded_dirs,
        "excluded_files": config.excluded_files
    }


def get_included_config_obj() -> IncludedConfig:
    """Load supported file extensions from included.json. Caches on first access."""
    global _included_config
    if _included_config is None:
        config_dict = load_json_config("included.json")
        if config_dict:
            try:
                ext_data = config_dict.get("supported_extensions", {})
                _included_config = from_dict(IncludedConfig, ext_data)
                logger.info("Successfully loaded included.json")
            except Exception as e:
                logger.error(f"Failed to parse included.json: {e}")
                raise
        else:
            _included_config = IncludedConfig()  # Use defaults
            logger.info("included.json not found, using default extensions")
    return _included_config


def get_included_config() -> Dict[str, Any]:
    """Get included extensions as a dict."""
    config = get_included_config_obj()
    return {"code": config.code, "doc": config.doc}


# ============================================================================
# Language Configuration
# ============================================================================

def get_lang_full_config() -> LanguageConfig:
    """
    Get the complete language configuration.
    Loads on first access and caches.
    """
    global _lang_config
    if _lang_config is None:
        config_dict = load_json_config("lang.json")
        if config_dict:
            try:
                _lang_config = from_dict(LanguageConfig, config_dict)
                logger.info("Successfully loaded and validated lang.json")
            except Exception as e:
                logger.error(f"Failed to parse lang.json: {e}")
                raise
    return _lang_config


def get_lang_config() -> Dict[str, Any]:
    """Get language configuration as a dictionary (for backward compatibility)."""
    config = get_lang_full_config()
    return config.model_dump()


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages mapping."""
    config = get_lang_full_config()
    return config.supported_languages


def get_default_language() -> str:
    """Get default language code."""
    config = get_lang_full_config()
    return config.default


# ============================================================================
# Client Management
# ============================================================================

def get_client_classes() -> Dict[str, Any]:
    """
    Get the client class mapping. Imports AzureAIClient lazily to avoid circular imports.
    
    Returns:
        Dict mapping class name strings to actual class objects
    """
    global _client_classes
    if _client_classes is None:
        from backend.clients.azureai_client import AzureAIClient
        _client_classes = {
            "AzureAIClient": AzureAIClient,
        }
    return _client_classes


def get_azure_ai_client(task: str = 'chat', model: Optional[str] = None) -> Any:
    """
    Get a shared Azure AI client instance (singleton pattern).
    
    Args:
        task: 'chat' or 'reasoning' (used to get initialize_kwargs)
        model: Optional model name (ignored)
        
    Returns:
        Cached AzureAIClient instance
    """
    global _azure_ai_client
    if _azure_ai_client is None:
        from backend.clients.azureai_client import AzureAIClient
        initialize_kwargs = get_azure_openai_text_config(task)
        _azure_ai_client = AzureAIClient(**initialize_kwargs)
        logger.info("[Config] Created shared AzureAIClient instance")
    return _azure_ai_client


def get_model_config(provider: str = None, model: str = None, task: str = 'chat') -> Dict[str, Any]:
    """
    Get configuration for Azure OpenAI model.

    Parameters:
        provider: Model provider (ignored, always uses 'azure')
        model: Model name (ignored, always uses deployment from infra.json)
        task: 'chat' or 'reasoning'

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    infra = get_infra_config()
    section = infra.azure_openai.chat if task == 'chat' else infra.azure_openai.reasoning
    deployment = section.deployment
    temperature = section.temperature

    logger.info(f"Using Azure OpenAI {task} deployment: {deployment}")

    client_classes = get_client_classes()

    return {
        "model_client": client_classes.get("AzureAIClient"),
        "initialize_kwargs": get_azure_openai_text_config(task),
        "model_kwargs": {
            "model": deployment,
            "temperature": temperature
        }
    }


# ============================================================================
# Legacy Support - configs dictionary
# ============================================================================

def get_configs_dict() -> Dict[str, Any]:
    """
    Get all configurations as a dictionary (for backward compatibility).
    This maintains the same structure as the original config.py.
    """
    configs = {
        "default_provider": "azure",
        "generator": get_generator_config(),
        "embedder": get_embedder_config(),
        "retriever": get_retriever_config(),
        "text_splitter": get_text_splitter_config(),
        "file_filters": get_file_filters_config(),
        "lang_config": get_lang_config(),
    }
    
    # Add model_client to embedder config
    client_classes = get_client_classes()
    configs["embedder"]["model_client"] = client_classes.get("AzureAIClient")
    
    return configs


# For backward compatibility, expose a configs dictionary
# This allows existing code to work without modification
configs = get_configs_dict()


# ============================================================================
# Legacy Function Aliases
# ============================================================================

def get_embedder_type() -> str:
    """Always returns 'azure' as only Azure OpenAI is supported."""
    return 'azure'


def is_ollama_embedder() -> bool:
    """Always returns False as Ollama is not supported."""
    return False


def load_generator_config() -> Dict[str, Any]:
    """Legacy function - use get_generator_config() instead."""
    # Return in old format for backward compatibility
    return {"generator": get_generator_config()}


def load_embedder_config() -> Dict[str, Any]:
    """Legacy function - use get_embedder_config() instead."""
    return {
        "embedder": get_embedder_config(),
        "retriever": get_retriever_config(),
        "text_splitter": get_text_splitter_config()
    }


def load_repo_config() -> Dict[str, Any]:
    """Legacy function - use get_file_filters_config() instead."""
    return {
        "file_filters": get_file_filters_config_obj().model_dump(),
    }


def load_lang_config() -> Dict[str, Any]:
    """Legacy function - use get_lang_config() instead."""
    return get_lang_config()


def get_search_config() -> 'AzureAISearchConfig':
    """Get Azure AI Search configuration from infra.json.

    Returns:
        AzureAISearchConfig Pydantic model
    """
    infra = get_infra_config()
    search = getattr(infra, 'azure_ai_search', None)
    if search is None:
        return AzureAISearchConfig()
    return search


def is_search_configured() -> bool:
    """Check if Azure AI Search is enabled and configured."""
    cfg = get_search_config()
    return cfg.enabled and bool(cfg.endpoint)


def get_aml_config() -> 'AzureMLConfig':
    """Get Azure ML configuration from infra.json.

    Returns:
        AzureMLConfig Pydantic model (check .enabled before using)
    """
    infra = get_infra_config()
    aml = getattr(infra, 'azure_ml', None)
    if aml is None:
        return AzureMLConfig()
    return aml
