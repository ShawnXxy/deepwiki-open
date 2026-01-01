#!/usr/bin/env python3
"""
Configuration checker for CodeWiki/DeepWiki.
Validates that required Azure configurations are properly set.
SECURITY: Never prints actual key/secret values, only shows if they are set.
"""

import os
import sys
from pathlib import Path

def mask_value(value: str | None) -> str:
    """Return status indicator without exposing actual value."""
    if value is None or value == "":
        return "✗ Not set"
    return "✓ Set"

def check_env_config():
    """Check environment variable configuration."""
    print("\n📋 Environment Variables:")
    print(f"  • AZURE_CLIENT_ID (MSI):      {mask_value(os.environ.get('AZURE_CLIENT_ID'))}")
    print(f"  • AZURE_OPENAI_API_KEY:       {mask_value(os.environ.get('AZURE_OPENAI_API_KEY'))}")
    print(f"  • AZURE_OPENAI_ENDPOINT:      {mask_value(os.environ.get('AZURE_OPENAI_ENDPOINT'))}")

def check_infra_config():
    """Check infra.json configuration."""
    print("\n📋 Config Files:")
    
    # Find config directory
    config_paths = [
        Path("backend/config/infra.json"),
        Path("api/config/infra.json"),
        Path("/app/backend/config/infra.json"),
    ]
    
    infra_path = None
    for path in config_paths:
        if path.exists():
            infra_path = path
            break
    
    if not infra_path:
        print("  • infra.json: ✗ Not found")
        return
    
    print(f"  • infra.json: ✓ Found at {infra_path}")
    
    try:
        import json
        with open(infra_path) as f:
            config = json.load(f)
        
        # Check Azure OpenAI config
        aoai = config.get("azure_openai", {})
        print("\n📋 Azure OpenAI (from infra.json):")
        print(f"  • endpoint:        {mask_value(aoai.get('endpoint'))}")
        print(f"  • deployment:      {mask_value(aoai.get('deployment'))}")
        print(f"  • api_version:     {mask_value(aoai.get('api_version'))}")
        
        # Check MSI config
        msi = config.get("managed_identity", {})
        print("\n📋 Managed Identity (from infra.json):")
        print(f"  • client_id:       {mask_value(msi.get('client_id'))}")
        
        # Check Blob Storage config
        blob = config.get("azure_blob_storage", {})
        print("\n📋 Azure Blob Storage (from infra.json):")
        print(f"  • enabled:         {'✓ Yes' if blob.get('enabled') else '✗ No'}")
        print(f"  • account_name:    {mask_value(blob.get('account_name'))}")
        print(f"  • container_name:  {mask_value(blob.get('container_name'))}")
        
        # Check App Insights config
        insights = config.get("azure_application_insights", {})
        print("\n📋 Application Insights (from infra.json):")
        print(f"  • enabled:         {'✓ Yes' if insights.get('enabled') else '✗ No'}")
        print(f"  • connection_string: {mask_value(insights.get('connection_string'))}")
        
    except Exception as e:
        print(f"  • Error reading config: {e}")

def check_backend_config():
    """Check backend configuration module."""
    print("\n📋 Backend Configuration Status:")
    
    try:
        from backend.config import (
            is_azure_openai_configured,
            get_embedder_type,
        )
        
        print(f"  • Azure OpenAI configured: {'✓ Yes' if is_azure_openai_configured() else '✗ No'}")
        
        try:
            embedder_type = get_embedder_type()
            print(f"  • Embedder type: {embedder_type}")
        except Exception as e:
            print(f"  • Embedder type: ✗ Error - {e}")
            
    except ImportError as e:
        print(f"  • Backend module: ✗ Not available ({e})")

def main():
    print("=" * 50)
    print("  CodeWiki Configuration Checker")
    print("  SECURITY: No secrets are printed")
    print("=" * 50)
    
    check_env_config()
    check_infra_config()
    check_backend_config()
    
    print("\n" + "=" * 50)
    print("  Check complete")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
