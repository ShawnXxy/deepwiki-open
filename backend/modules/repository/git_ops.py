"""
Git operations for repository management.

Provides functions for cloning, pulling, and managing Git repositories.
"""

import os
import shutil
import subprocess
import logging
import time
from urllib.parse import urlparse, urlunparse, quote

logger = logging.getLogger(__name__)


def get_head_commit_hash(local_path: str) -> str:
    """Get the HEAD commit hash from a cloned repository.

    Args:
        local_path: Path to the cloned repository

    Returns:
        str: Full SHA commit hash, or empty string on failure
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=local_path
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get HEAD commit hash: {e}")
        return ""


def detect_default_branch(local_path: str) -> str:
    """
    Detect the default branch of a cloned repository.

    Args:
        local_path: Path to the cloned repository

    Returns:
        str: Name of the default branch (e.g., 'main', 'master')
    """
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=local_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Output format: refs/remotes/origin/main
        output = result.stdout.decode("utf-8").strip()
        branch = output.split("/")[-1]
        logger.debug(f"Detected default branch: {branch}")
        return branch
    except subprocess.CalledProcessError:
        # Fallback to main/master
        logger.warning("Could not detect default branch, trying main/master")
        for fallback in ['main', 'master']:
            try:
                subprocess.run(
                    ["git", "rev-parse", "--verify", f"origin/{fallback}"],
                    cwd=local_path,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info(f"Using fallback branch: {fallback}")
                return fallback
            except subprocess.CalledProcessError:
                continue
        # Last resort
        return 'main'


def download_repo(
    repo_url: str = None,
    local_path: str = None,
    type: str = "github",
    access_token: str = None,
    branch: str = None,
    git_source=None,
    force_update: bool = False,
    token_type: str = 'pat'
) -> str:
    """
    Downloads a Git repository (GitHub, GitLab, Bitbucket, or Azure DevOps) to a specified
    local path. If the repository already exists and force_update=True, pulls latest changes.

    Supports both legacy parameter-based API and new GitSource type-based API.

    Args:
        repo_url (str, optional): The URL of the Git repository to clone.
        local_path (str, optional): The local directory where the repository will be cloned.
        type (str): The type of repository (github, gitlab, bitbucket, azuredevops).
        access_token (str, optional): Access token for private repositories.
        branch (str, optional): Specific branch to clone. If None, uses default branch.
        git_source (GitSource, optional): GitSource object containing all git parameters.
        force_update (bool): If True and repo exists, pull latest changes instead of skipping.
        token_type (str): 'pat' for Personal Access Tokens, 'bearer' for JWT/MSI tokens.

    Returns:
        str: The output message from the git command.
    """
    # If git_source provided, extract parameters from it
    if git_source is not None:
        from backend.types import GitSource
        if not isinstance(git_source, GitSource):
            raise TypeError(f"git_source must be GitSource type, got {type(git_source)}")

        repo_url = git_source.repository.url
        type = git_source.repository.repo_type
        access_token = git_source.credentials.access_token if git_source.credentials else None
        branch = git_source.reference.branch if git_source.reference else None
        # If local_path not explicitly provided, use from GitSource
        if local_path is None:
            local_path = git_source.repository.local_path

    # Validate required parameters
    if not repo_url:
        raise ValueError("repo_url must be provided either directly or via git_source")
    if not local_path:
        raise ValueError("local_path must be provided either directly or via git_source")

    try:
        # Check if Git is installed
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Enable long paths for enterprise repos (Windows 260-char limit)
        subprocess.run(
            ["git", "config", "--global", "core.longpaths", "true"],
            capture_output=True, text=True,
        )

        # Check if repository already exists
        if os.path.exists(local_path) and os.listdir(local_path):
            # Validate the clone is complete (has .git directory)
            if not os.path.isdir(os.path.join(local_path, ".git")):
                logger.warning(f"Directory {local_path} exists but has no .git — incomplete clone, removing and re-cloning")
                shutil.rmtree(local_path)
            elif force_update:
                # Pull latest changes instead of skipping
                logger.info(f"Repository exists at {local_path}, pulling latest changes...")
                return _pull_repo_internal(local_path, access_token, repo_url, type, token_type=token_type)
            else:
                logger.warning(f"Repository already exists at {local_path}. Using existing.")
                return f"Using existing repository at {local_path}"

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Prepare the clone URL with access token if provided
        clone_url = repo_url
        extra_git_config = []  # Extra -c flags for git (e.g., Bearer auth header)
        token_status = '[PROVIDED]' if access_token else '[NONE]'
        logger.debug(
            f"download_repo called with type={type}, "
            f"access_token={token_status}"
        )
        if access_token:
            # Token type is passed explicitly by the caller.
            # 'bearer' = JWT from MSI/AAD (must use Authorization header)
            # 'pat' = Personal Access Token (embeds in URL)
            is_bearer_token = (token_type == 'bearer')

            parsed = urlparse(repo_url)
            encoded_token = quote(access_token, safe='')
            logger.debug(f"Token type: {token_type}, length: {len(access_token)}")

            if type == "azuredevops" and is_bearer_token:
                # Bearer tokens (from MSI) must use the Authorization header.
                # Embedding a ~1200-char JWT in the URL fails because git
                # treats it as a username and prompts for a password.
                extra_git_config = [
                    "-c", f"http.extraHeader=Authorization: Bearer {access_token}"
                ]
                logger.info("Using Bearer token via http.extraHeader for ADO clone")
            elif type == "azuredevops":
                # PAT for ADO: embed in URL
                clone_url = urlunparse((
                    parsed.scheme,
                    f"{encoded_token}@{parsed.netloc}",
                    parsed.path, '', '', ''
                ))
            else:
                logger.warning(f"Unknown repo type: {type}, token may not be embedded correctly")

            logger.info(f"Using access token for authentication (type={type})")
        else:
            logger.warning(f"No access token provided for repo type={type}")

        # Clone the repository with branch handling
        logger.info(f"Cloning repository from {repo_url} to {local_path}")

        # Build git clone command with branch parameter if specified
        clone_cmd = ["git"] + extra_git_config + ["clone", "--depth=1", "--single-branch"]

        # Add branch parameter if specified, with fallback logic
        if branch and branch.strip():
            clone_cmd.extend(["-b", branch.strip()])
            logger.info(f"Attempting to clone branch: {branch.strip()}")

        clone_cmd.extend([clone_url, local_path])

        # Retry logic for transient network errors (DNS, timeouts)
        max_retries = 3
        retry_delay = 2  # seconds, doubles each retry

        for attempt in range(1, max_retries + 1):
            # Clean up empty directory from previous failed attempt
            if os.path.exists(local_path) and not os.listdir(local_path):
                shutil.rmtree(local_path)
                os.makedirs(local_path, exist_ok=True)

            try:
                result = subprocess.run(
                    clone_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info("Repository cloned successfully")
                return result.stdout.decode("utf-8")

            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode('utf-8')
                is_network_error = any(
                    phrase in error_msg.lower()
                    for phrase in [
                        'could not resolve host',
                        'failed to connect',
                        'connection timed out',
                        'connection refused',
                        'network is unreachable',
                        'ssl',
                    ]
                )

                if is_network_error and attempt < max_retries:
                    logger.warning(
                        f"Clone attempt {attempt}/{max_retries} failed "
                        f"(network error), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    # Clean up failed clone directory for retry
                    if os.path.exists(local_path):
                        shutil.rmtree(local_path)
                        os.makedirs(local_path, exist_ok=True)
                    continue

                # Not a network error or final attempt — try branch fallbacks
                if branch and branch.strip() and branch.strip() not in ['main', 'master']:
                    logger.warning(f"Branch {branch.strip()} not found, trying fallback branches")

                    # Clean up failed clone attempt
                    if os.path.exists(local_path):
                        shutil.rmtree(local_path)

                    # Try with 'main' branch
                    for fallback_branch in ['main', 'master']:
                        try:
                            logger.info(f"Trying fallback branch: {fallback_branch}")
                            clone_cmd_fallback = [
                                "git", "clone", "-b", fallback_branch,
                                clone_url, local_path
                            ]
                            result = subprocess.run(
                                clone_cmd_fallback,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                            logger.info(
                                f"Repository cloned successfully using "
                                f"fallback branch: {fallback_branch}"
                            )
                            return result.stdout.decode("utf-8")
                        except subprocess.CalledProcessError:
                            # Clean up and try next fallback
                            if os.path.exists(local_path):
                                shutil.rmtree(local_path)
                            continue

                    # If all fallback branches fail, try without branch
                    try:
                        logger.info(
                            "Trying clone without branch specification "
                            "(default branch)"
                        )
                        clone_cmd_default = [
                            "git", "clone", clone_url, local_path
                        ]
                        result = subprocess.run(
                            clone_cmd_default,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        logger.info(
                            "Repository cloned successfully using "
                            "default branch"
                        )
                        return result.stdout.decode("utf-8")
                    except subprocess.CalledProcessError:
                        pass

                # If we get here, all attempts failed
                error_msg = e.stderr.decode('utf-8')
                # Sanitize error message to remove any tokens
                if access_token:
                    error_msg = error_msg.replace(
                        access_token, "***TOKEN***"
                    )
                    encoded_token = quote(access_token, safe='')
                    error_msg = error_msg.replace(
                        encoded_token, "***TOKEN***"
                    )
                raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")


# Alias for backward compatibility
download_github_repo = download_repo


def _pull_repo_internal(
    local_path: str,
    access_token: str = None,
    repo_url: str = None,
    repo_type: str = "github",
    token_type: str = 'pat'
) -> str:
    """
    Internal helper: Pull the latest changes from a Git repository.
    Called by download_repo when force_update=True and repo exists.

    Args:
        token_type: 'pat' for Personal Access Tokens, 'bearer' for JWT/MSI tokens.
    """
    git_dir = os.path.join(local_path, ".git")
    if not os.path.exists(git_dir):
        raise ValueError(f"Not a Git repository: {local_path}")

    logger.info(f"Pulling latest changes for repository at {local_path}")

    try:
        # If we have an access token and repo URL, update the remote URL for auth
        # For bearer tokens (JWT from MSI), use http.extraHeader instead of URL
        use_bearer_header = False
        if access_token and repo_url:
            is_bearer_token = (token_type == 'bearer')

            if repo_type == "azuredevops" and is_bearer_token:
                # Bearer tokens: use git -c http.extraHeader for all git commands
                use_bearer_header = True
                logger.info("Using Bearer token via http.extraHeader for ADO pull")
            elif repo_type == "azuredevops":
                parsed = urlparse(repo_url)
                encoded_token = quote(access_token, safe='')
                auth_url = urlunparse((
                    parsed.scheme, f"{encoded_token}@{parsed.netloc}",
                    parsed.path, '', '', ''
                ))

                # Temporarily set the remote URL with auth
                subprocess.run(
                    ["git", "remote", "set-url", "origin", auth_url],
                    cwd=local_path,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info("Updated remote URL with authentication token")
            else:
                logger.warning(f"Unknown repo type: {repo_type}, skipping auth")

        # Get current branch name
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=local_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        current_branch = result.stdout.decode("utf-8").strip()
        if not current_branch:
            current_branch = "HEAD"
        logger.info(f"Current branch: {current_branch}")

        # Build git command prefix with bearer auth if needed
        git_cmd = ["git"]
        if use_bearer_header:
            git_cmd += ["-c", f"http.extraHeader=Authorization: Bearer {access_token}"]

        # Check if this is a shallow clone
        shallow_file = os.path.join(git_dir, "shallow")
        is_shallow = os.path.exists(shallow_file)

        if is_shallow:
            logger.info("Repository is a shallow clone, fetching with unshallow...")
            subprocess.run(
                git_cmd + ["fetch", "--unshallow", "origin"],
                cwd=local_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("Repository unshallowed successfully")

        # Try to pull
        try:
            result = subprocess.run(
                git_cmd + ["pull", "--ff-only"],
                cwd=local_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output = result.stdout.decode("utf-8")
            logger.info(f"Git pull successful: {output}")
            return f"Pull successful: {output}"
        except subprocess.CalledProcessError:
            # If fast-forward fails, reset to remote
            logger.warning("Fast-forward pull failed, resetting to remote branch")
            subprocess.run(
                git_cmd + ["fetch", "origin"],
                cwd=local_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if current_branch != "HEAD":
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{current_branch}"],
                    cwd=local_path,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            logger.info("Repository reset to remote successfully")
            return "Repository reset to remote HEAD"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        # Sanitize error message to remove tokens
        if access_token:
            error_msg = error_msg.replace(access_token, "***TOKEN***")
            encoded_token = quote(access_token, safe='')
            error_msg = error_msg.replace(encoded_token, "***TOKEN***")
        raise ValueError(f"Git pull failed: {error_msg}")
    finally:
        # Reset remote URL to original (without token) for security
        # Skip if using bearer header — remote URL was never modified
        if access_token and repo_url and not use_bearer_header:
            try:
                subprocess.run(
                    ["git", "remote", "set-url", "origin", repo_url],
                    cwd=local_path,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.debug("Reset remote URL to original (without token)")
            except Exception:
                pass
