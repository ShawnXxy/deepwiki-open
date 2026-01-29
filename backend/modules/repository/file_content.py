"""
File content retrieval from remote repositories.

Provides functions to fetch file content from GitHub, GitLab, Bitbucket, and Azure DevOps.
"""

import base64
import json
import logging
from urllib.parse import urlparse, quote
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


def get_github_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitHub repository using the GitHub API.
    Supports both public GitHub (github.com) and GitHub Enterprise (custom domains).
    
    Args:
        repo_url (str): The URL of the GitHub repository 
        file_path (str): The path to the file within the repository
        access_token (str, optional): GitHub personal access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not a valid GitHub URL
    """
    try:
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # Determine the API base URL
        if parsed_url.netloc == "github.com":
            api_base = "https://api.github.com"
        else:
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"
        
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"

        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        logger.info(f"Fetching file content from GitHub API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response from GitHub API")

        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API error: {content_data['message']}")

        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                content_base64 = content_data["content"].replace("\n", "")
                content = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unexpected encoding: {content_data['encoding']}")
        else:
            raise ValueError("File content not found in GitHub API response")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository (cloud or self-hosted).

    Args:
        repo_url (str): The GitLab repo URL
        file_path (str): File path within the repository
        access_token (str, optional): GitLab personal access token

    Returns:
        str: File content

    Raises:
        ValueError: If anything fails
    """
    try:
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format")

        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')
        encoded_file_path = quote(file_path, safe='')

        # Try to get the default branch
        default_branch = None
        try:
            project_info_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}"
            project_headers = {}
            if access_token:
                project_headers["PRIVATE-TOKEN"] = access_token
            
            project_response = requests.get(project_info_url, headers=project_headers)
            if project_response.status_code == 200:
                project_data = project_response.json()
                default_branch = project_data.get('default_branch', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning("Could not fetch project info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching project info: {e}, using 'main' as default branch")
            default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"Fetching file content from GitLab API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API error: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Bitbucket repository using the Bitbucket API.

    Args:
        repo_url (str): The URL of the Bitbucket repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): Bitbucket personal access token

    Returns:
        str: The content of the file as a string
    """
    try:
        if not (repo_url.startswith("https://bitbucket.org/") or repo_url.startswith("http://bitbucket.org/")):
            raise ValueError("Not a valid Bitbucket repository URL")

        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            raise ValueError("Invalid Bitbucket URL format")

        owner = parts[-2]
        repo = parts[-1].replace(".git", "")

        # Try to get the default branch
        default_branch = None
        try:
            repo_info_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
            repo_headers = {}
            if access_token:
                repo_headers["Authorization"] = f"Bearer {access_token}"
            
            repo_response = requests.get(repo_info_url, headers=repo_headers)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                default_branch = repo_data.get('mainbranch', {}).get('name', 'main')
                logger.info(f"Found default branch: {default_branch}")
            else:
                logger.warning("Could not fetch repository info, using 'main' as default branch")
                default_branch = 'main'
        except Exception as e:
            logger.warning(f"Error fetching repository info: {e}, using 'main' as default branch")
            default_branch = 'main'

        api_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/src/{default_branch}/{file_path}"

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        logger.info(f"Fetching file content from Bitbucket API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                content = response.text
            elif response.status_code == 404:
                raise ValueError("File not found on Bitbucket.")
            elif response.status_code == 401:
                raise ValueError("Unauthorized access to Bitbucket.")
            elif response.status_code == 403:
                raise ValueError("Forbidden access to Bitbucket.")
            elif response.status_code == 500:
                raise ValueError("Internal server error on Bitbucket.")
            else:
                response.raise_for_status()
                content = response.text
            return content
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def get_azuredevops_file_content(repo_url: str, file_path: str, access_token: str = None) -> str:
    """
    Retrieves the content of a file from an Azure DevOps repository using the Azure DevOps REST API.

    Args:
        repo_url (str): The URL of the Azure DevOps repository
        file_path (str): The path to the file within the repository
        access_token (str): Azure DevOps Personal Access Token (PAT) - required

    Returns:
        str: The content of the file as a string
    """
    if not access_token:
        raise ValueError("Personal Access Token (PAT) is required for Azure DevOps repositories")
    
    try:
        if not (repo_url.startswith("https://dev.azure.com/") or 
                (repo_url.startswith("https://") and ".visualstudio.com" in repo_url)):
            raise ValueError("Not a valid Azure DevOps repository URL")

        # Extract organization, project, and repo name
        if "dev.azure.com" in repo_url:
            parts = repo_url.rstrip('/').split('/')
            if len(parts) < 6 or parts[-2] != "_git":
                raise ValueError("Invalid Azure DevOps URL format")
            organization = parts[3]
            project = parts[4]
            repo = parts[-1]
        else:
            parts = repo_url.rstrip('/').split('/')
            if ".visualstudio.com" not in parts[2]:
                raise ValueError("Invalid Azure DevOps URL format")
            organization = parts[2].split('.')[0]
            project = parts[4] if len(parts) > 4 else parts[-1]
            repo = parts[-1]

        # Get default branch first
        repo_api_url = f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo}?api-version=6.0"
        
        headers = {
            "Authorization": f"Basic {access_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Fetching repository info from Azure DevOps: {repo_api_url}")
        repo_response = requests.get(repo_api_url, headers=headers)
        
        default_branch = "main"
        if repo_response.status_code == 200:
            repo_data = repo_response.json()
            default_branch = repo_data.get('defaultBranch', 'refs/heads/main').replace('refs/heads/', '')
            logger.info(f"Found default branch: {default_branch}")
        else:
            logger.warning("Could not fetch repository info, using 'main' as default branch")

        file_api_url = f"https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repo}/items"
        params = {
            "path": f"/{file_path}",
            "version": default_branch,
            "api-version": "6.0"
        }

        logger.info(f"Fetching file content from Azure DevOps API: {file_api_url}")
        response = requests.get(file_api_url, headers=headers, params=params)
        
        if response.status_code == 200:
            content = response.text
        elif response.status_code == 404:
            raise ValueError("File not found in Azure DevOps repository.")
        elif response.status_code == 401:
            raise ValueError("Unauthorized access to Azure DevOps.")
        elif response.status_code == 403:
            raise ValueError("Forbidden access to Azure DevOps.")
        else:
            response.raise_for_status()
            content = response.text
            
        return content
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching file content from Azure DevOps: {e}")
    except Exception as e:
        raise ValueError(f"Failed to get file content from Azure DevOps: {str(e)}")


def get_file_content(repo_url: str, file_path: str, type: str = "github", access_token: str = None) -> str:
    """
    Retrieves the content of a file from a Git repository (GitHub, GitLab, Bitbucket, or Azure DevOps).

    Args:
        repo_url (str): The URL of the repository
        file_path (str): The path to the file within the repository
        type (str): Repository type (github, gitlab, bitbucket, azuredevops)
        access_token (str, optional): Access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not valid
    """
    if type == "github":
        return get_github_file_content(repo_url, file_path, access_token)
    elif type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token)
    elif type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token)
    elif type == "azuredevops":
        return get_azuredevops_file_content(repo_url, file_path, access_token)
    else:
        raise ValueError("Unsupported repository type. Only GitHub, GitLab, Bitbucket, and Azure DevOps are supported.")
