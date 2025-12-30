import logging
import os
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime


class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()


def get_log_filename(prefix: str = "backend") -> str:
    """Generate log filename with date pattern: prefix-yymmdd.log"""
    date_str = datetime.now().strftime("%y%m%d")
    return f"{prefix}-{date_str}.log"


def setup_logging(format: str = None, log_prefix: str = "backend"):
    """
    Configure logging for the application with daily log rotation.

    Args:
        format: Custom log format string
        log_prefix: Prefix for log file name (default: "backend")

    Environment variables:
        LOG_LEVEL: Log level (default: INFO)
        LOG_BACKUP_COUNT: Number of backup files to keep (default: 30)

    Log files are named as {prefix}-yymmdd.log and rotate daily at midnight.
    """
    # Determine log directory at project root
    base_dir = Path(__file__).parent.parent  # Go up from backend to project root
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log file path with date
    log_filename = get_log_filename(log_prefix)
    log_file_path = log_dir / log_filename

    # Get log level from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get backup count (default: 30 days)
    try:
        backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 30))
    except ValueError:
        backup_count = 30

    # Configure format
    log_format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"

    # Create handlers
    # TimedRotatingFileHandler rotates at midnight, keeping backup_count days of logs
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8"
    )
    # Custom namer to maintain our naming pattern
    file_handler.namer = lambda name: name.replace(".log.", "-") + ".log" if ".log." in name else name
    
    console_handler = logging.StreamHandler()

    # Set format for both handlers
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add filter to suppress "Detected file change" messages
    file_handler.addFilter(IgnoreLogChangeDetectedFilter())
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())

    # Apply logging configuration
    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler], force=True)

    # Suppress verbose third-party loggers
    noisy_loggers = [
        "azure.core.pipeline.policies.http_logging_policy",  # Azure HTTP request/response details
        "azure.identity",  # Azure credential acquisition
        "azure.identity._credentials",
        "azure.identity._credentials.environment",
        "azure.identity._credentials.managed_identity",
        "azure.identity._credentials.chained",
        "adalflow.tracing.mlflow_integration",  # MLflow not available warnings
        "faiss.loader",  # FAISS loading attempts
        "faiss",  # GPU Faiss warnings
        "watchfiles.main",  # File change detection
        "httpx",  # HTTP client request logs
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Set Azure loggers to ERROR to hide most noise
    logging.getLogger("azure").setLevel(logging.ERROR)

    # Log configuration info
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging configured: level={log_level_str}, "
        f"file={log_file_path}, backup_count={backup_count} days"
    )
    
    return log_file_path


# Frontend logger singleton
_frontend_logger = None
_frontend_handler = None


def get_frontend_logger():
    """Get or create the frontend logger with daily rotation."""
    global _frontend_logger, _frontend_handler
    
    if _frontend_logger is None:
        # Create dedicated frontend logger
        _frontend_logger = logging.getLogger("frontend")
        _frontend_logger.setLevel(logging.DEBUG)
        _frontend_logger.propagate = False  # Don't propagate to root logger
        
        # Setup file handler for frontend logs at project root
        base_dir = Path(__file__).parent.parent  # Go up from backend to project root
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_filename = get_log_filename("frontend")
        log_file_path = log_dir / log_filename
        
        # Get backup count
        try:
            backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 30))
        except ValueError:
            backup_count = 30
        
        _frontend_handler = TimedRotatingFileHandler(
            log_file_path,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
        _frontend_handler.namer = lambda name: name.replace(".log.", "-") + ".log" if ".log." in name else name
        
        # Simpler format for frontend logs
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        _frontend_handler.setFormatter(formatter)
        
        _frontend_logger.addHandler(_frontend_handler)
    
    return _frontend_logger


def log_frontend_message(level: str, message: str, context: dict = None):
    """
    Log a message from the frontend.
    
    Args:
        level: Log level (debug, info, warn, error)
        message: Log message
        context: Optional context dictionary
    """
    logger = get_frontend_logger()
    
    # Format message with context if provided
    if context:
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} | {context_str}"
    else:
        full_message = message
    
    level = level.lower()
    if level == "debug":
        logger.debug(full_message)
    elif level == "info":
        logger.info(full_message)
    elif level == "warn" or level == "warning":
        logger.warning(full_message)
    elif level == "error":
        logger.error(full_message)
    else:
        logger.info(full_message)
