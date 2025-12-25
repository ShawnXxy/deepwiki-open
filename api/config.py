"""
Configuration module for DeepWiki.
This module handles loading and managing configuration for Azure OpenAI services.
All configuration is read from infra.json - no .env file needed.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import Azure AI client
from api.azureai_client import AzureAIClient

# Infrastructure configuration (loaded later via load_json_config)
_infra_config: Optional[Dict[str, Any]] = None


def get_infra_config() -> Dict[str, Any]:
    """
    Get the infrastructure configuration from infra.json.
    Loads the config on first access and caches it.
    
    Returns:
        Dict containing infrastructure configuration
    """
    global _infra_config
    if _infra_config is None:
        _infra_config = load_json_config("infra.json")
    return _infra_config


def get_managed_identity_client_id() -> Optional[str]:
    """
    Get the managed identity client ID from infra.json.
    
    Returns:
        The client ID string or None if not configured
    """
    infra = get_infra_config()
    return infra.get("managed_identity", {}).get("client_id")


def get_azure_openai_config() -> Dict[str, str]:
    """
    Get Azure OpenAI configuration for text generation from infra.json.
    
    Returns:
        Dict containing endpoint, api_version, deployment
    """
    infra = get_infra_config()
    azure_config = infra.get("azure_openai", {})
    return {
        "endpoint": azure_config.get("endpoint", ""),
        "api_version": azure_config.get("api_version", "2024-12-01-preview"),
        "deployment": azure_config.get("deployment", "")
    }


def get_azure_openai_embedding_config_from_infra() -> Dict[str, str]:
    """
    Get Azure OpenAI embedding configuration from infra.json.
    
    Returns:
        Dict containing endpoint, api_version, deployment
    """
    infra = get_infra_config()
    embedding_config = infra.get("azure_openai_embedding", {})
    # Fall back to main azure_openai config if embedding-specific not set
    azure_config = infra.get("azure_openai", {})
    return {
        "endpoint": embedding_config.get("endpoint") or azure_config.get("endpoint", ""),
        "api_version": embedding_config.get("api_version") or azure_config.get("api_version", "2024-12-01-preview"),
        "deployment": embedding_config.get("deployment", "text-embedding-3-large")
    }


def is_azure_openai_configured() -> bool:
    """
    Check if Azure OpenAI is configured in infra.json.
    
    Returns:
        bool: True if Azure OpenAI is properly configured
    """
    azure_config = get_azure_openai_config()
    
    # Check for basic Azure OpenAI configuration
    has_basic_config = bool(
        azure_config.get("endpoint") and 
        azure_config.get("api_version")
    )
    
    # Check for Azure endpoint pattern
    endpoint = azure_config.get("endpoint", "")
    has_azure_pattern = ".openai.azure.com" in endpoint
    
    # Check for MSI client ID in infra.json
    has_msi_config = bool(get_managed_identity_client_id())
    
    return has_basic_config and has_azure_pattern and has_msi_config


def get_azure_openai_text_config() -> Dict[str, Any]:
    """
    Get Azure OpenAI configuration for text generation from infra.json.
    Uses MSI authentication.
    
    Returns:
        Dict containing azure_endpoint, api_version, and managed_identity_client_id
    """
    azure_config = get_azure_openai_config()
    return {
        "azure_endpoint": azure_config.get("endpoint"),
        "api_version": azure_config.get("api_version", "2024-12-01-preview"),
        "managed_identity_client_id": get_managed_identity_client_id()
    }


def get_azure_deployment_name(model_name: str = None) -> str:
    """
    Get the Azure OpenAI deployment name.
    
    If model_name is provided, returns it directly.
    Otherwise, returns the deployment from infra.json.
    
    Args:
        model_name: Optional model name override
    
    Returns:
        The deployment name to use for Azure OpenAI API calls
    """
    if model_name:
        return model_name
    azure_config = get_azure_openai_config()
    return azure_config.get("deployment", "")


def get_azure_openai_embedding_config() -> Dict[str, Any]:
    """
    Get Azure OpenAI configuration for embeddings from infra.json.
    Uses MSI authentication.
    
    Returns:
        Dict containing azure_endpoint, api_version, and managed_identity_client_id
    """
    embedding_config = get_azure_openai_embedding_config_from_infra()
    return {
        "azure_endpoint": embedding_config.get("endpoint"),
        "api_version": embedding_config.get("api_version", "2024-12-01-preview"),
        "managed_identity_client_id": get_managed_identity_client_id()
    }


# Wiki authentication settings
raw_auth_mode = os.environ.get('DEEPWIKI_AUTH_MODE', 'False')
WIKI_AUTH_MODE = raw_auth_mode.lower() in ['true', '1', 't']
WIKI_AUTH_CODE = os.environ.get('DEEPWIKI_AUTH_CODE', '')

# Get configuration directory from environment variable, or use default if not set
CONFIG_DIR = os.environ.get('DEEPWIKI_CONFIG_DIR', None)

# Client class mapping (Azure only)
CLIENT_CLASSES = {
    "AzureAIClient": AzureAIClient,
}


def replace_env_placeholders(
    config: Union[Dict[str, Any], List[Any], str, Any]
) -> Union[Dict[str, Any], List[Any], str, Any]:
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


def load_json_config(filename):
    """Load JSON configuration file from config directory."""
    try:
        if CONFIG_DIR:
            config_path = Path(CONFIG_DIR) / filename
        else:
            config_path = Path(__file__).parent / "config" / filename

        logger.info(f"Loading configuration from {config_path}")

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


def load_generator_config():
    """Load generator model configuration for Azure OpenAI.
    Injects model and temperature from infra.json."""
    generator_config = load_json_config("generator.json")
    
    # Inject model and temperature from infra.json
    azure_config = get_azure_openai_config()
    if "generator" in generator_config:
        if "model_kwargs" not in generator_config["generator"]:
            generator_config["generator"]["model_kwargs"] = {}
        # Set model and temperature from infra.json
        generator_config["generator"]["model_kwargs"]["model"] = azure_config.get("deployment", "")
        if "temperature" in azure_config:
            generator_config["generator"]["model_kwargs"]["temperature"] = azure_config["temperature"]

    # Add client class for Azure provider (legacy support)
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            if provider_id == "azure":
                provider_config["model_client"] = AzureAIClient
            elif provider_config.get("client_class") in CLIENT_CLASSES:
                provider_config["model_client"] = CLIENT_CLASSES[provider_config["client_class"]]

    return generator_config


def load_embedder_config():
    """Load embedder configuration for Azure OpenAI.
    Injects model, dimensions, and initialize_kwargs from infra.json."""
    embedder_config = load_json_config("embedder.json")
    
    # Inject model and dimensions from infra.json
    embedding_config = get_azure_openai_embedding_config_from_infra()
    
    # Get initialize_kwargs for Azure OpenAI client
    initialize_kwargs = get_azure_openai_embedding_config()
    
    # Process embedder configurations
    for key in ["embedder", "embedder_azure"]:
        if key in embedder_config:
            if "model_kwargs" not in embedder_config[key]:
                embedder_config[key]["model_kwargs"] = {}
            # Set model and dimensions from infra.json
            embedder_config[key]["model_kwargs"]["model"] = embedding_config.get("deployment", "text-embedding-3-large")
            if "dimensions" in embedding_config:
                embedder_config[key]["model_kwargs"]["dimensions"] = embedding_config["dimensions"]
            
            # Add initialize_kwargs for Azure OpenAI client
            embedder_config[key]["initialize_kwargs"] = initialize_kwargs
            
            # Process client classes
            if "client_class" in embedder_config[key]:
                class_name = embedder_config[key]["client_class"]
                if class_name in CLIENT_CLASSES:
                    embedder_config[key]["model_client"] = CLIENT_CLASSES[class_name]

    return embedder_config


def get_embedder_config():
    """
    Get the current embedder configuration for Azure OpenAI.

    Returns:
        dict: The embedder configuration with model_client resolved
    """
    return configs.get("embedder", {})


def get_embedder_type() -> str:
    """
    Get the current embedder type.

    Returns:
        str: Always returns 'azure' as only Azure OpenAI is supported
    """
    return 'azure'


def is_ollama_embedder() -> bool:
    """
    Check if the current embedder is Ollama.
    
    Returns:
        bool: Always returns False as Ollama is not supported
    """
    return False


def load_repo_config():
    """Load repository and file filters configuration."""
    return load_json_config("repo.json")


def load_lang_config():
    """Load language configuration."""
    default_config = {
        "supported_languages": {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh": "Mandarin Chinese (中文)",
            "zh-tw": "Traditional Chinese (繁體中文)",
            "es": "Spanish (Español)",
            "kr": "Korean (한국어)",
            "vi": "Vietnamese (Tiếng Việt)",
            "pt-br": "Brazilian Portuguese (Português Brasileiro)",
            "fr": "Français (French)",
            "ru": "Русский (Russian)"
        },
        "default": "en"
    }

    loaded_config = load_json_config("lang.json")

    if not loaded_config:
        return default_config

    if "supported_languages" not in loaded_config or "default" not in loaded_config:
        logger.warning(
            "Language configuration file 'lang.json' is malformed. "
            "Using default language configuration."
        )
        return default_config

    return loaded_config


# Default excluded directories and files
DEFAULT_EXCLUDED_DIRS: List[str] = [
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/",
    "./.ruff_cache/", "./.coverage/",
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]

# Initialize empty configuration
configs = {}

# Load all configuration files
generator_config = load_generator_config()
embedder_config = load_embedder_config()
repo_config = load_repo_config()
lang_config = load_lang_config()

# Validate Azure OpenAI configuration
if not is_azure_openai_configured():
    logger.warning(
        "Azure OpenAI is not properly configured. Please set the following:\n"
        "- AZURE_OPENAI_API_KEY\n"
        "- AZURE_OPENAI_ENDPOINT\n"
        "- AZURE_OPENAI_VERSION\n"
        "- AZURE_OPENAI_EMBEDDING_ENDPOINT (or use AZURE_OPENAI_ENDPOINT)\n"
        "- AZURE_OPENAI_EMBEDDING_API_KEY (or use AZURE_OPENAI_API_KEY)\n"
    )

# Set default provider to Azure
configs["default_provider"] = "azure"
logger.info("Using Azure OpenAI as default provider")

# Store generator config
if generator_config:
    configs["generator"] = generator_config.get("generator", {})

# Update embedder configuration for Azure
if embedder_config:
    if "embedder_azure" in embedder_config:
        logger.info("Using embedder_azure configuration from config file")
        configs["embedder"] = embedder_config["embedder_azure"]
    elif "embedder" in embedder_config:
        configs["embedder"] = embedder_config["embedder"]
    else:
        # Create default Azure OpenAI embedder configuration
        azure_config = get_azure_openai_embedding_config()
        embedding_infra = get_azure_openai_embedding_config_from_infra()
        configs["embedder"] = {
            "client_class": "AzureAIClient",
            "model_client": AzureAIClient,
            "batch_size": 10,
            "model_kwargs": {
                "model": embedding_infra.get("deployment", "text-embedding-3-large"),
                "dimensions": 3072,
                "encoding_format": "float"
            },
            "initialize_kwargs": azure_config
        }
    logger.info("Using Azure OpenAI for embeddings")
    
    # Copy retriever and text_splitter configurations
    for key in ["retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]

# Update repository configuration
if repo_config:
    for key in ["file_filters", "repository"]:
        if key in repo_config:
            configs[key] = repo_config[key]

# Update language configuration
if lang_config:
    configs["lang_config"] = lang_config


def get_model_config(provider=None, model=None):
    """
    Get configuration for Azure OpenAI model.

    Parameters:
        provider (str): Model provider (ignored, always uses 'azure')
        model (str): Model name, or None to use default from infra.json

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    # Get Azure config from infra.json
    azure_config = get_azure_openai_config()
    
    # Get model from infra.json if not provided
    if not model:
        model = azure_config.get("deployment", "o4-mini")
    
    # Get temperature from infra.json
    temperature = azure_config.get("temperature", 1.0)

    # Prepare Azure configuration
    result = {
        "model_client": AzureAIClient,
        "initialize_kwargs": get_azure_openai_text_config(),
        "model_kwargs": {
            "model": model,
            "temperature": temperature
        }
    }

    return result
