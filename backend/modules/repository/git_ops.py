"""
Git operations for repository management.

Provides functions for cloning, pulling, and managing Git repositories.
"""

import os
import re
import shutil
import subprocess
import logging
import time
from urllib.parse import urlparse, urlunparse, quote

logger = logging.getLogger(__name__)


def _sanitize_token_from_text(text: str, access_token: str = None) -> str:
    """Remove tokens and credentials from text to prevent leaks in logs.

    Redacts:
    - The literal access_token (and its URL-encoded form)
    - Any Bearer token pattern in Authorization headers
    - Any PAT-in-URL pattern (token@host)
    """
    if not text:
        return text
    if access_token:
        text = text.replace(access_token, '***TOKEN***')
        text = text.replace(quote(access_token, safe=''), '***TOKEN***')
    # Catch any Bearer token we might have missed
    text = re.sub(
        r'Authorization: Bearer [A-Za-z0-9_\-\.]+',
        'Authorization: Bearer ***TOKEN***',
        text,
    )
    # Catch PAT-in-URL (token@dev.azure.com)
    text = re.sub(
        r'://[^@/]{8,}@',
        '://***TOKEN***@',
        text,
    )
    return text


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


def get_changed_files(local_path: str, prev_commit_hash: str) -> dict:
    """Return paths changed between ``prev_commit_hash`` and the working tree.

    The map values are git porcelain status letters:

    - ``"A"`` — added
    - ``"M"`` — modified
    - ``"D"`` — deleted
    - ``"R"`` — renamed (rename target reported as new path; old path is
      additionally returned as ``"D"`` so chunks are dropped)
    - ``"C"`` — copied
    - ``"T"`` — type change
    - ``"U"`` — unmerged

    Combines ``git diff --name-status`` against the previous commit with
    ``git status --porcelain`` so uncommitted edits in the working tree are
    captured too. Returns ``{}`` when ``prev_commit_hash`` is empty/unknown
    (caller should treat that as "cold start, embed everything").

    All paths are returned as forward-slash relative paths, matching what
    ``read_all_documents`` records in ``Document.meta_data['file_path']``.
    """
    if not prev_commit_hash:
        return {}

    changed: dict = {}

    def _norm(p: str) -> str:
        return p.replace("\\", "/").strip()

    # 1) Diff against previous commit hash (committed history).
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "-z", prev_commit_hash, "HEAD"],
            capture_output=True, text=True, cwd=local_path, check=False
        )
        if result.returncode != 0:
            # Most common cause: prev_commit_hash is no longer reachable
            # (force-push, history rewrite). Caller will fall back to full reprocess.
            logger.warning(
                f"git diff against {prev_commit_hash[:8]} failed: "
                f"{result.stderr.strip()[:200]}"
            )
            return {}

        # -z output: NUL-separated tokens. For non-rename entries it's
        # "STATUS\0PATH\0"; for rename/copy it's "R100\0OLD\0NEW\0".
        tokens = [t for t in result.stdout.split('\0') if t]
        i = 0
        while i < len(tokens):
            status = tokens[i]
            letter = status[0] if status else ''
            if letter in ('R', 'C'):
                if i + 2 < len(tokens):
                    old_path = _norm(tokens[i + 1])
                    new_path = _norm(tokens[i + 2])
                    changed[old_path] = 'D'
                    changed[new_path] = letter
                    i += 3
                    continue
                i += 1
                continue
            if letter in ('A', 'M', 'D', 'T', 'U'):
                if i + 1 < len(tokens):
                    changed[_norm(tokens[i + 1])] = letter
                    i += 2
                    continue
            # Unknown token shape — skip one to avoid infinite loop
            i += 1
    except Exception as e:
        logger.warning(f"git diff --name-status failed: {e}")
        return {}

    # 2) Uncommitted edits in working tree (covers freshly cloned repos
    #    where HEAD == prev but build artefacts were touched, plus any
    #    edits applied after clone).
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            capture_output=True, text=True, cwd=local_path, check=False
        )
        if result.returncode == 0:
            # -z porcelain format: XY <space> PATH \0 [original_path \0 if rename]
            # XY is 2 chars; index 2 is space; path starts at 3.
            tokens = [t for t in result.stdout.split('\0') if t]
            j = 0
            while j < len(tokens):
                tok = tokens[j]
                if len(tok) < 4:
                    j += 1
                    continue
                xy = tok[:2]
                path = _norm(tok[3:])
                # Map worktree (Y) or index (X) status to our letter codes.
                # Prefer worktree state; fall back to index.
                letter = (xy[1] if xy[1] != ' ' else xy[0]).upper()
                if letter == '?':
                    letter = 'A'  # untracked => treat as added
                if letter in ('R', 'C') and j + 1 < len(tokens):
                    # Original path follows in next token; mark it as deleted
                    old = _norm(tokens[j + 1])
                    changed.setdefault(old, 'D')
                    j += 2
                else:
                    j += 1
                if letter not in ('A', 'M', 'D', 'R', 'C', 'T', 'U'):
                    continue
                # Don't downgrade an existing 'D' (delete is terminal for the path)
                if changed.get(path) != 'D':
                    changed[path] = letter
    except Exception as e:
        logger.debug(f"git status --porcelain failed (non-fatal): {e}")

    return changed


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
            # 'gcm' = Git Credential Manager handles auth (no token needed)
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
        elif token_type == 'gcm':
            # Git Credential Manager handles authentication natively.
            # No token injection needed — GCM supports AAD/browser SSO
            # for Azure DevOps out of the box on developer machines.
            logger.info(
                "Delegating authentication to Git Credential Manager"
            )
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
                # Sanitize ALL error text immediately to prevent token leaks
                error_msg = _sanitize_token_from_text(error_msg, access_token)
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
                error_msg = _sanitize_token_from_text(error_msg, access_token)
                logger.error(f"Git clone failed: {error_msg}")

                # Provide actionable guidance for auth failures
                if 'authentication failed' in error_msg.lower():
                    logger.error(
                        "Authentication failed. Possible causes:\n"
                        "  1. Your identity may not have access to this repo\n"
                        "  2. The token may have expired — run 'az login' again\n"
                        "  3. Set REPO_ACCESS_TOKEN with a PAT in backend/.env"
                    )
                raise ValueError(f"Error during cloning: {error_msg}")
    except ValueError:
        # Re-raise ValueError (from inner raise) without wrapping
        raise
    except Exception as e:
        sanitized = _sanitize_token_from_text(str(e), access_token)
        logger.error(f"Unexpected error during clone: {sanitized}")
        raise ValueError(f"An unexpected error occurred: {sanitized}")


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
        token_type: 'pat' for Personal Access Tokens, 'bearer' for JWT/MSI tokens,
                    'gcm' for Git Credential Manager (no token injection).
    """
    git_dir = os.path.join(local_path, ".git")
    if not os.path.exists(git_dir):
        raise ValueError(f"Not a Git repository: {local_path}")

    logger.info(f"Pulling latest changes for repository at {local_path}")

    try:
        # If we have an access token and repo URL, update the remote URL for auth
        # For bearer tokens (JWT from MSI), use http.extraHeader instead of URL
        # For gcm, let Git Credential Manager handle auth natively
        use_bearer_header = False
        if access_token and repo_url and token_type != 'gcm':
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
        error_msg = _sanitize_token_from_text(error_msg, access_token)
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
