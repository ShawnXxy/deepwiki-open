"""
Types module for DeepWiki.
Provides type classes for configuration management.
"""

from .config_types import (
    # Infra config types
    ManagedIdentityConfig,
    AzureOpenAILLMConfig,
    AzureOpenAIEmbeddingConfig,
    AzureOpenAIGroupConfig,
    AzureBlobStorageConfig,
    AzureApplicationInsightsConfig,
    InfraConfig,
    
    # Embedder config types
    ModelKwargs,
    EmbedderModelConfig,
    RetrieverConfig,
    TextSplitterConfig,
    EmbedderConfig,
    
    # Generator config types
    GeneratorConfig,
    
    # Repository config types
    FileFiltersConfig,
    IncludedConfig,
    
    # Language config types
    LanguageConfig,
)

from .converter import (
    # Converter functions
    from_dict,
    to_dict,
)

from .git_types import (
    # Git types
    RepoType,
    GitCredentials,
    GitReference,
    GitRepository,
    GitSource,
    WikiCacheIdentifier,
    
    # Helper functions
    parse_github_url,
    create_git_source_from_params,
)

from .processor_types import (
    # Processing types
    FileFilter,
    ProcessedFile,
    FileProcessingStats,
    DocumentChunk,
    RepositoryMetadata,
)

__all__ = [
    # Infra config
    'ManagedIdentityConfig',
    'AzureOpenAILLMConfig',
    'AzureOpenAIEmbeddingConfig',
    'AzureOpenAIGroupConfig',
    'AzureBlobStorageConfig',
    'AzureApplicationInsightsConfig',
    'InfraConfig',
    
    # Embedder config
    'ModelKwargs',
    'EmbedderModelConfig',
    'RetrieverConfig',
    'TextSplitterConfig',
    'EmbedderConfig',
    
    # Generator config
    'GeneratorConfig',
    
    # Repository config
    'FileFiltersConfig',
    'IncludedConfig',
    
    # Language config
    'LanguageConfig',
    
    # Converter utilities
    'from_dict',
    'to_dict',
    
    # Git types
    'RepoType',
    'GitCredentials',
    'GitReference',
    'GitRepository',
    'GitSource',
    'WikiCacheIdentifier',
    'parse_github_url',
    'create_git_source_from_params',
    
    # Processing types
    'FileFilter',
    'ProcessedFile',
    'FileProcessingStats',
    'DocumentChunk',
    'RepositoryMetadata',
]
