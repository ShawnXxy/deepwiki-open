"""
Configuration module for DeepWiki.
This module handles loading and managing configuration for Azure OpenAI services.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Import Azure AI client
from api.azureai_client import AzureAIClient

# Azure OpenAI environment variables
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_VERSION = os.environ.get('AZURE_OPENAI_VERSION')
AZURE_OPENAI_DEPLOYMENT = os.environ.get('AZURE_OPENAI_DEPLOYMENT')
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT')
AZURE_OPENAI_EMBEDDING_API_KEY = os.environ.get('AZURE_OPENAI_EMBEDDING_API_KEY')
AZURE_OPENAI_EMBEDDING_VERSION = os.environ.get('AZURE_OPENAI_EMBEDDING_VERSION')
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')

# Set Azure OpenAI environment variables
if AZURE_OPENAI_API_KEY:
    os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
if AZURE_OPENAI_ENDPOINT:
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
if AZURE_OPENAI_VERSION:
    os.environ["AZURE_OPENAI_VERSION"] = AZURE_OPENAI_VERSION
if AZURE_OPENAI_EMBEDDING_ENDPOINT:
    os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] = AZURE_OPENAI_EMBEDDING_ENDPOINT
if AZURE_OPENAI_EMBEDDING_API_KEY:
    os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"] = AZURE_OPENAI_EMBEDDING_API_KEY
if AZURE_OPENAI_EMBEDDING_VERSION:
    os.environ["AZURE_OPENAI_EMBEDDING_VERSION"] = AZURE_OPENAI_EMBEDDING_VERSION


def is_azure_openai_configured() -> bool:
    """
    Check if Azure OpenAI is configured by checking for required environment variables
    and Azure endpoint pattern (.openai.azure.com).
    
    Returns:
        bool: True if Azure OpenAI is properly configured
    """
    # Check for basic Azure OpenAI configuration
    has_basic_config = bool(
        AZURE_OPENAI_API_KEY and 
        AZURE_OPENAI_ENDPOINT and 
        AZURE_OPENAI_VERSION
    )
    
    # Check for embedding configuration (can use same endpoint or separate)
    has_embedding_config = bool(
        (AZURE_OPENAI_EMBEDDING_ENDPOINT or AZURE_OPENAI_ENDPOINT) and
        (AZURE_OPENAI_EMBEDDING_API_KEY or AZURE_OPENAI_API_KEY) and
        (AZURE_OPENAI_EMBEDDING_VERSION or AZURE_OPENAI_VERSION)
    )
    
    # Check for Azure endpoint pattern
    endpoint_to_check = AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_EMBEDDING_ENDPOINT
    has_azure_pattern = bool(endpoint_to_check and ".openai.azure.com" in endpoint_to_check)
    
    return has_basic_config and has_embedding_config and has_azure_pattern


def get_azure_openai_text_config() -> Dict[str, str]:
    """
    Get Azure OpenAI configuration for text generation.
    
    Returns:
        Dict containing api_key, azure_endpoint, api_version
    """
    return {
        "api_key": AZURE_OPENAI_API_KEY,
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
        "api_version": AZURE_OPENAI_VERSION or "2024-12-01-preview"
    }


def get_azure_deployment_name(model_name: str) -> str:
    """
    Get the Azure OpenAI deployment name for a given model name.
    
    The deployment name is taken directly from the model_name parameter,
    which should come from the config file or user selection.
    
    Args:
        model_name: The model name from config (e.g., 'gpt-4.1', 'o4-mini')
    
    Returns:
        The deployment name to use for Azure OpenAI API calls
    """
    # Use the model name from config/request as the deployment name
    return model_name


def get_azure_openai_embedding_config() -> Dict[str, str]:
    """
    Get Azure OpenAI configuration for embeddings.
    Falls back to text generation config if embedding-specific config is not available.
    
    Returns:
        Dict containing api_key, azure_endpoint, api_version
    """
    return {
        "api_key": AZURE_OPENAI_EMBEDDING_API_KEY or AZURE_OPENAI_API_KEY,
        "azure_endpoint": AZURE_OPENAI_EMBEDDING_ENDPOINT or AZURE_OPENAI_ENDPOINT,
        "api_version": AZURE_OPENAI_EMBEDDING_VERSION or AZURE_OPENAI_VERSION or "2024-12-01-preview"
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
    """Load generator model configuration for Azure OpenAI."""
    generator_config = load_json_config("generator.json")

    # Add client class for Azure provider
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            if provider_id == "azure":
                provider_config["model_client"] = AzureAIClient
            elif provider_config.get("client_class") in CLIENT_CLASSES:
                provider_config["model_client"] = CLIENT_CLASSES[provider_config["client_class"]]

    return generator_config


def load_embedder_config():
    """Load embedder configuration for Azure OpenAI."""
    # Load Azure-specific embedder config
    azure_config_path = Path(__file__).parent / "config" / "embedder.azure.json"
    if azure_config_path.exists():
        logger.info("Loading Azure-specific embedder configuration")
        embedder_config = load_json_config("embedder.azure.json")
    else:
        logger.info("Loading default embedder configuration")
        embedder_config = load_json_config("embedder.json")

    # Process client classes for Azure embedder
    for key in ["embedder", "embedder_azure"]:
        if key in embedder_config and "client_class" in embedder_config[key]:
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

# Update provider configuration
if generator_config:
    configs["providers"] = generator_config.get("providers", {})

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
        configs["embedder"] = {
            "client_class": "AzureAIClient",
            "model_client": AzureAIClient,
            "batch_size": 10,
            "model_kwargs": {
                "model": (AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-3-large"),
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
        model (str): Model name, or None to use default model

    Returns:
        dict: Configuration containing model_client, model and other parameters
    """
    # Always use Azure provider
    provider = "azure"
    
    if "providers" not in configs:
        raise ValueError("Provider configuration not loaded")

    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError("Azure provider configuration not found")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError("Model client not specified for Azure provider")

    # If model not provided, use default model
    if not model:
        model = "gpt-4.1"

    # Get model parameters (if present)
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
        logger.info(f"Found model '{model}' in Azure config with params: {model_params}")
    else:
        logger.warning(
            f"Model '{model}' not found in Azure models. "
            f"Available models: {list(provider_config.get('models', {}).keys())}"
        )
        default_model = provider_config.get("default_model")
        if default_model and default_model in provider_config.get("models", {}):
            model_params = provider_config["models"][default_model]
            logger.info(f"Using default model '{default_model}' params: {model_params}")

    # Prepare Azure configuration
    result = {
        "model_client": model_client,
        "initialize_kwargs": get_azure_openai_text_config(),
        "model_kwargs": {"model": model, **model_params}
    }

    return result
