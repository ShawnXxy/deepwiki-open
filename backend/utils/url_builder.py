"""
URL builder for commit-pinned source file permalinks.

Generates repository file URLs with optional line-number anchors
for Azure DevOps, GitHub, GitLab, and Bitbucket.

Used by:
    - promptstore/wiki_page.py → format_file_paths_list()
    - modules/chat/service.py → format_context_text()
"""


def build_source_url(
    repo_url: str,
    file_path: str,
    commit_hash: str = "",
    repo_type: str = "github",
    start_line: int = None,
    end_line: int = None,
) -> str:
    """
    Build a commit-pinned permalink to a source file with optional line range.

    Args:
        repo_url: Base repository URL.
        file_path: Relative path within the repo (e.g. "src/main.py").
        commit_hash: Full or short commit SHA. Falls back to HEAD/branch.
        repo_type: One of "github", "gitlab", "bitbucket", "azuredevops".
        start_line: 1-based start line (inclusive).
        end_line: 1-based end line (inclusive).

    Returns:
        Full URL string.
    """
    if not repo_url or not repo_url.startswith(("http://", "https://")):
        return file_path

    # Normalize path separators
    fp = file_path.replace("\\", "/")
    if not fp.startswith("/"):
        fp = "/" + fp

    base = repo_url.rstrip("/")
    if base.endswith(".git"):
        base = base[:-4]

    ref = commit_hash or "HEAD"

    if repo_type == "azuredevops":
        url = f"{base}?path={fp}"
        if commit_hash:
            url += f"&version=GC{commit_hash}"
        if start_line is not None:
            url += f"&line={start_line}"
            if end_line is not None and end_line != start_line:
                url += f"&lineEnd={end_line}"
                url += "&lineStartColumn=1&lineEndColumn=999"
            else:
                url += "&lineStartColumn=1&lineEndColumn=999"
        return url

    if repo_type in ("github", "gitlab"):
        url = f"{base}/blob/{ref}{fp}"
        if start_line is not None:
            url += f"#L{start_line}"
            if end_line is not None and end_line != start_line:
                url += f"-L{end_line}"
        return url

    if repo_type == "bitbucket":
        url = f"{base}/src/{ref}{fp}"
        if start_line is not None:
            url += f"#lines-{start_line}"
            if end_line is not None and end_line != start_line:
                url += f":{end_line}"
        return url

    # Unknown repo type — return path only
    return file_path
