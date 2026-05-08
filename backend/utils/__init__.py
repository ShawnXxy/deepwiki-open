"""
Backend utilities module.

- url_builder: commit-pinned source file URL builder
- filter: centralized file filtering (.gitignore parsing) and
  path-safe sanitization of git branch names
- guard_checker: inspect Azure OpenAI content filter (RAI) policies
- guard_session: snapshot/relax/restore content-filter policies
  around a processor pipeline run
"""

from backend.utils.url_builder import build_source_url
from backend.utils.filter import (
    load_gitignore,
    is_gitignored,
    sanitize_branch_for_path,
)
from backend.utils.guard_checker import (
    check_content_filters,
    format_filter_report,
    is_content_filter_error,
    is_guard_check_enabled,
    update_content_filter,
)
from backend.utils.guard_session import (
    GuardSession,
    get_active,
)

__all__ = [
    "build_source_url",
    "load_gitignore",
    "is_gitignored",
    "sanitize_branch_for_path",
    "check_content_filters",
    "format_filter_report",
    "is_content_filter_error",
    "is_guard_check_enabled",
    "update_content_filter",
    "GuardSession",
    "get_active",
]
