import logging
import os
import hashlib
import time
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Tuple


class IgnoreLogChangeDetectedFilter(logging.Filter):
    """Filter out file change detection messages from watchfiles."""
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()


class DeduplicationFilter(logging.Filter):
    """
    Filter that deduplicates identical log messages within a time window.
    
    Instead of suppressing by log level, this filter:
    - Allows ALL log levels through
    - Deduplicates identical messages within a configurable time window
    - Shows a count when the same message is repeated
    """
    
    def __init__(self, window_seconds: float = 5.0, name: str = ""):
        super().__init__(name)
        self.window_seconds = window_seconds
        # Maps message hash -> (last_time, count, last_record)
        self._seen: Dict[str, Tuple[float, int, logging.LogRecord]] = {}
        self._cleanup_interval = 60.0  # Cleanup old entries every 60 seconds
        self._last_cleanup = time.time()
    
    def _get_message_key(self, record: logging.LogRecord) -> str:
        """Generate a unique key for a log message (ignoring timestamp)."""
        # Include level, logger name, and message content
        key_parts = f"{record.levelno}:{record.name}:{record.getMessage()}"
        return hashlib.md5(key_parts.encode()).hexdigest()
    
    def _cleanup_old_entries(self, now: float):
        """Remove entries older than the window to prevent memory growth."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        cutoff = now - self.window_seconds * 2  # Keep entries for 2x window
        self._seen = {
            k: v for k, v in self._seen.items() 
            if v[0] > cutoff
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        now = time.time()
        self._cleanup_old_entries(now)
        
        key = self._get_message_key(record)
        
        if key in self._seen:
            last_time, count, last_record = self._seen[key]
            
            if now - last_time < self.window_seconds:
                # Within window - increment count but don't log
                self._seen[key] = (last_time, count + 1, record)
                return False
            else:
                # Window expired - if we had duplicates, log count first
                if count > 1:
                    # Modify the message to show how many were suppressed
                    record.msg = f"{record.msg} (repeated {count}x in last {self.window_seconds}s)"
                # Start new window
                self._seen[key] = (now, 1, record)
                return True
        else:
            # First occurrence
            self._seen[key] = (now, 1, record)
            return True


class RateLimitFilter(logging.Filter):
    """
    Rate limit extremely frequent log messages.
    
    For messages that occur more than max_per_second times per second,
    only log periodically and show the count.
    """
    
    def __init__(self, max_per_second: int = 10, name: str = ""):
        super().__init__(name)
        self.max_per_second = max_per_second
        # Maps message hash -> (window_start, count_in_window, logged_in_window)
        self._counts: Dict[str, Tuple[float, int, bool]] = {}
    
    def _get_message_key(self, record: logging.LogRecord) -> str:
        key_parts = f"{record.levelno}:{record.name}:{record.getMessage()}"
        return hashlib.md5(key_parts.encode()).hexdigest()
    
    def filter(self, record: logging.LogRecord) -> bool:
        now = time.time()
        key = self._get_message_key(record)
        
        if key in self._counts:
            window_start, count, logged = self._counts[key]
            
            if now - window_start < 1.0:
                # Still in same second
                count += 1
                if count > self.max_per_second and not logged:
                    # Rate limited - log once with warning
                    record.msg = f"[RATE LIMITED] {record.msg} (>{self.max_per_second}/s)"
                    self._counts[key] = (window_start, count, True)
                    return True
                elif count > self.max_per_second:
                    # Already logged rate limit warning this second
                    return False
                else:
                    self._counts[key] = (window_start, count, logged)
                    return True
            else:
                # New second window
                self._counts[key] = (now, 1, False)
                return True
        else:
            self._counts[key] = (now, 1, False)
            return True


class ComponentLabelFilter(logging.Filter):
    """Add component label [BE] or [FE] to log messages."""
    
    def __init__(self, label: str = "BE", name: str = ""):
        super().__init__(name)
        self.label = label
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Prepend label to message if not already present
        if not record.msg.startswith(f"[{self.label}]"):
            record.msg = f"[{self.label}] - {record.msg}"
        return True


def get_log_filename(prefix: str = "backend") -> str:
    """Generate log filename with date pattern: prefix-yymmdd.log"""
    date_str = datetime.now().strftime("%y%m%d")
    return f"{prefix}-{date_str}.log"


def setup_logging(format: str = None, log_prefix: str = "backend"):
    """
    Configure logging for the application with daily log rotation.
    
    Design Philosophy:
    - Log ALL details at all levels (no level-based suppression)
    - Use deduplication to avoid repeating identical messages
    - Use rate limiting for extremely frequent messages
    - Label all messages with [BE] for backend identification

    Args:
        format: Custom log format string
        log_prefix: Prefix for log file name (default: "backend")

    Environment variables:
        LOG_LEVEL: Minimum log level to capture (default: DEBUG for all details)
        LOG_BACKUP_COUNT: Number of backup files to keep (default: 30)
        LOG_DEDUP_WINDOW: Deduplication window in seconds (default: 5)

    Log files are named as {prefix}-yymmdd.log and rotate daily at midnight.
    """
    # Determine log directory at project root
    base_dir = Path(__file__).parent.parent  # Go up from backend to project root
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log file path with date
    log_filename = get_log_filename(log_prefix)
    log_file_path = log_dir / log_filename

    # Default to DEBUG to capture all details - filtering is done by deduplication
    log_level_str = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    log_level = getattr(logging, log_level_str, logging.DEBUG)

    # Get configuration from environment
    try:
        backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 30))
    except ValueError:
        backup_count = 30
    
    try:
        dedup_window = float(os.environ.get("LOG_DEDUP_WINDOW", 5.0))
    except ValueError:
        dedup_window = 5.0

    # Configure format - simplified to show level and labeled message
    log_format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"

    # Create handlers
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.namer = lambda name: name.replace(".log.", "-") + ".log" if ".log." in name else name
    
    console_handler = logging.StreamHandler()

    # Set format for both handlers
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add filters in order:
    # 1. Component label [BE]
    # 2. Ignore file change detection messages  
    # 3. Deduplication (avoid repeating identical messages)
    # 4. Rate limiting (prevent log flooding)
    
    label_filter = ComponentLabelFilter(label="BE")
    ignore_filter = IgnoreLogChangeDetectedFilter()
    dedup_filter = DeduplicationFilter(window_seconds=dedup_window)
    rate_filter = RateLimitFilter(max_per_second=10)
    
    for handler in [file_handler, console_handler]:
        handler.addFilter(label_filter)
        handler.addFilter(ignore_filter)
        handler.addFilter(dedup_filter)
        handler.addFilter(rate_filter)

    # Apply logging configuration - capture ALL levels
    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler], force=True)

    # For third-party loggers: still log them but with deduplication handling
    # Set to INFO to reduce noise but not silence completely
    third_party_loggers = [
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.identity",
        "azure.identity._credentials",
        "azure.identity._credentials.environment", 
        "azure.identity._credentials.managed_identity",
        "azure.identity._credentials.chained",
        "adalflow.tracing.mlflow_integration",
        "faiss.loader",
        "faiss",
        "watchfiles.main",
        "httpx",
    ]
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # Azure loggers can be very verbose - set to WARNING
    logging.getLogger("azure").setLevel(logging.WARNING)

    # Log configuration info
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={log_level_str}, "
        f"file={log_file_path}, dedup_window={dedup_window}s"
    )
    
    return log_file_path


# Frontend logger singleton
_frontend_logger = None
_frontend_handler = None
_frontend_dedup_filter = None


def get_frontend_logger():
    """
    Get or create the frontend logger with daily rotation.
    
    Uses same deduplication philosophy as backend:
    - Log all details with level labels
    - Deduplicate identical messages
    - Label with [FE] for frontend identification
    """
    global _frontend_logger, _frontend_handler, _frontend_dedup_filter
    
    if _frontend_logger is None:
        # Create dedicated frontend logger
        _frontend_logger = logging.getLogger("frontend")
        _frontend_logger.setLevel(logging.DEBUG)  # Capture all levels
        _frontend_logger.propagate = False  # Don't propagate to root logger
        
        # Setup file handler for frontend logs at project root
        base_dir = Path(__file__).parent.parent
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_filename = get_log_filename("frontend")
        log_file_path = log_dir / log_filename
        
        try:
            backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 30))
        except ValueError:
            backup_count = 30
        
        try:
            dedup_window = float(os.environ.get("LOG_DEDUP_WINDOW", 5.0))
        except ValueError:
            dedup_window = 5.0
        
        _frontend_handler = TimedRotatingFileHandler(
            log_file_path,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
        _frontend_handler.namer = lambda name: name.replace(".log.", "-") + ".log" if ".log." in name else name
        
        # Format with [FE] label already in message from log_frontend_message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        _frontend_handler.setFormatter(formatter)
        
        # Add deduplication filter for frontend logs
        # Add deduplication filter
        _frontend_dedup_filter = DeduplicationFilter(window_seconds=dedup_window)
        _frontend_handler.addFilter(_frontend_dedup_filter)
        
        # Add [FE] label to all frontend messages
        _frontend_handler.addFilter(ComponentLabelFilter("FE"))
        
        _frontend_logger.addHandler(_frontend_handler)
    
    return _frontend_logger


def log_frontend_message(level: str, message: str, context: dict = None):
    """
    Log a message from the frontend.
    
    All messages are logged with [FE] label and full context.
    Deduplication prevents identical messages from flooding the log.
    
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
