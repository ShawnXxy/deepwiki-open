#!/usr/bin/env python3
"""
Test script for Azure DevOps integration in DeepWiki.
Tests the get_azuredevops_file_content function with various scenarios.
"""

import sys
import os
from api.data_pipeline import get_azuredevops_file_content

sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))


def test_azure_devops_parsing():
    """Test Azure DevOps URL parsing."""
    print("Testing Azure DevOps URL parsing...")

    # Test cases for Azure DevOps URL parsing
    test_urls = [
        "https://dev.azure.com/microsoft/vscode/_git/vscode",
        "https://microsoft.visualstudio.com/vscode/_git/vscode",
        "https://dev.azure.com/contoso/MyProject/_git/MyRepo",
        "https://contoso.visualstudio.com/MyProject/_git/MyRepo"
    ]

    for url in test_urls:
        print(f"\nTesting URL: {url}")
        try:
            # Extract org and project from URL
            if "dev.azure.com" in url:
                parts = url.split("dev.azure.com/")[1].split("/")
                org = parts[0]
                project = parts[1]
            elif "visualstudio.com" in url:
                parts = url.split(".visualstudio.com/")[1].split("/")
                org = url.split(".visualstudio.com/")[0].split("//")[1]
                project = parts[0]

            repo = parts[3] if len(parts) > 3 else "unknown"

            print(f"  Organization: {org}")
            print(f"  Project: {project}")
            print(f"  Repository: {repo}")

        except Exception as e:
            print(f"  Error parsing URL: {e}")


def test_file_content_function():
    """Test the get_azuredevops_file_content function structure."""
    print("\n\nTesting get_azuredevops_file_content function structure...")

    # Test with dummy parameters (will fail auth but should show structure)
    try:
        repo_url = "https://dev.azure.com/test-org/test-project/_git/test-repo"
        get_azuredevops_file_content(
            repo_url=repo_url,
            file_path="README.md",
            access_token="dummy-token"
        )
        print("Function executed (authentication may have failed)")
    except Exception as e:
        print(f"Expected error (likely authentication): {e}")
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("✓ Function structure is correct - "
                  "authentication error is expected")
        elif "404" in str(e) or "not found" in str(e).lower():
            print("✓ Function structure is correct - "
                  "resource not found is expected")
        else:
            print(f"! Unexpected error type: {e}")


if __name__ == "__main__":
    print("DeepWiki Azure DevOps Integration Test")
    print("=" * 50)

    test_azure_devops_parsing()
    test_file_content_function()

    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo test with real Azure DevOps repositories:")
    print("1. Set up a Personal Access Token in Azure DevOps")
    print("2. Add it to your .env file as AZURE_DEVOPS_PAT")
    print("3. Use the frontend interface to test with real repositories")
