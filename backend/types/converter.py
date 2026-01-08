"""
Configuration converter utilities for DeepWiki.
Provides generic conversion functions between JSON/dict and Pydantic models.
"""

from typing import TypeVar, Type, Dict, Any
from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)


def from_dict(config_class: Type[T], data: Dict[str, Any]) -> T:
    """
    Generic converter to create a Pydantic model instance from a dictionary.
    
    Args:
        config_class: The Pydantic model class to instantiate
        data: The dictionary data (typically from JSON)
    
    Returns:
        An instance of the config class
    
    Example:
        infra = from_dict(InfraConfig, json_data)
        embedder = from_dict(EmbedderFullConfig, json_data)
    """
    return config_class(**data)


def to_dict(config: BaseModel, exclude_none: bool = False) -> Dict[str, Any]:
    """
    Convert a Pydantic model instance to a dictionary.
    
    Args:
        config: The Pydantic model instance
        exclude_none: If True, excludes fields with None values (default: False)
    
    Returns:
        Dictionary representation of the config
    
    Examples:
        # Include all fields
        config_dict = to_dict(infra_config)
        
        # Exclude None values for cleaner JSON output
        config_dict = to_dict(infra_config, exclude_none=True)
    """
    return config.model_dump(exclude_none=exclude_none)
