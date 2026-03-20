"""
Backend infrastructure module.

Cross-cutting infrastructure services: logging, monitoring.
"""

from backend.infra.logger import (
    setup_logging,
    log_frontend_message,
    get_frontend_logger,
    flush_application_insights,
    is_application_insights_enabled,
    SmartLogFilter,
)

__all__ = [
    "setup_logging",
    "log_frontend_message",
    "get_frontend_logger",
    "flush_application_insights",
    "is_application_insights_enabled",
    "SmartLogFilter",
]
