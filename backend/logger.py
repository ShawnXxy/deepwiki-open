"""
DeepWiki Logging Module - Simplified and Performance-Optimized

Design Philosophy:
- All logs use level labels (DEBUG, INFO, WARN, ERROR) for categorization
- Aggressive deduplication with level-aware windows
- Sampling for high-frequency informational logs
- Minimal overhead per log message
"""

import logging
import os
import time
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Optional, Tuple

# Application Insights state
_azure_monitor_configured = False
_logger_provider = None


class SmartLogFilter(logging.Filter):
    """
    Unified filter combining deduplication, sampling, and rate limiting.
    
    Strategy by log level:
    - ERROR/CRITICAL: Always log, short dedup window (2s)
    - WARNING: Always log, medium dedup window (10s)  
    - INFO: Sample 1-in-N after burst, long dedup window (30s)
    - DEBUG: Sample 1-in-N, very long dedup window (60s)
    
    This prevents log flooding while ensuring important messages get through.
    """
    
    # Level-specific config: (dedup_window_secs, sample_rate_after_burst, burst_limit)
    LEVEL_CONFIG = {
        logging.CRITICAL: (2.0, 1, 100),    # Always log, 2s dedup
        logging.ERROR: (2.0, 1, 100),       # Always log, 2s dedup
        logging.WARNING: (10.0, 1, 50),     # Always log, 10s dedup
        logging.INFO: (30.0, 10, 20),       # Sample 1/10 after 20 burst, 30s dedup
        logging.DEBUG: (60.0, 50, 10),      # Sample 1/50 after 10 burst, 60s dedup
    }
    
    def __init__(self, label: str = "BE", name: str = ""):
        super().__init__(name)
        self.label = label
        # message_key -> (first_seen_time, count, last_logged_time)
        self._seen: Dict[str, Tuple[float, int, float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 120.0  # Cleanup every 2 minutes
    
    def _get_config(self, level: int) -> Tuple[float, int, int]:
        """Get configuration for log level."""
        return self.LEVEL_CONFIG.get(level, self.LEVEL_CONFIG[logging.DEBUG])
    
    def _make_key(self, record: logging.LogRecord) -> str:
        """Create a simple key for deduplication (no hashing for speed)."""
        # Use level + logger name + first 100 chars of message
        msg = record.getMessage()[:100]
        return f"{record.levelno}:{record.name}:{msg}"
    
    def _cleanup(self, now: float):
        """Remove stale entries to prevent memory growth."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        # Remove entries older than max window (60s) + buffer
        cutoff = now - 120.0
        self._seen = {k: v for k, v in self._seen.items() if v[2] > cutoff}
    
    def filter(self, record: logging.LogRecord) -> bool:
        now = time.time()
        self._cleanup(now)
        
        # Add component label to message
        if not record.msg.startswith(f"[{self.label}]"):
            record.msg = f"[{self.label}] {record.msg}"
        
        # Skip file change detection messages (watchfiles noise)
        if "Detected file change in" in record.getMessage():
            return False
        
        key = self._make_key(record)
        dedup_window, sample_rate, burst_limit = self._get_config(record.levelno)
        
        if key not in self._seen:
            # First occurrence - always log
            self._seen[key] = (now, 1, now)
            return True
        
        first_seen, count, last_logged = self._seen[key]
        time_since_logged = now - last_logged
        
        # Check if dedup window expired - reset and log
        if time_since_logged >= dedup_window:
            if count > 1:
                # Show how many were suppressed
                record.msg = f"{record.msg} (repeated {count}x)"
            self._seen[key] = (now, 1, now)
            return True
        
        # Within window - apply sampling strategy
        count += 1
        
        if count <= burst_limit:
            # Within burst limit - log it
            self._seen[key] = (first_seen, count, now)
            return True
        
        # Beyond burst limit - apply sampling
        if sample_rate == 1 or (count - burst_limit) % sample_rate == 0:
            record.msg = f"{record.msg} (#{count})"
            self._seen[key] = (first_seen, count, now)
            return True
        
        # Suppressed by sampling
        self._seen[key] = (first_seen, count, last_logged)
        return False


def _get_config_path() -> Path:
    """Get path to config directory."""
    return Path(__file__).parent.parent / "config"


def _get_app_insights_config() -> Optional[dict]:
    """Get Application Insights configuration from infra.json."""
    try:
        config_path = _get_config_path() / "infra.json"
        if not config_path.exists():
            return None
        
        import json
        with open(config_path) as f:
            infra = json.load(f)
        
        return infra.get("azure_application_insights")
    except Exception:
        return None


def _get_managed_identity_client_id() -> Optional[str]:
    """Get managed identity client ID from infra.json."""
    try:
        config_path = _get_config_path() / "infra.json"
        if not config_path.exists():
            return None
        
        import json
        with open(config_path) as f:
            infra = json.load(f)
        
        return infra.get("managed_identity", {}).get("client_id")
    except Exception:
        return None


def setup_application_insights(
    connection_string: str = None,
    service_name: str = "deepwiki"
) -> bool:
    """
    Configure Azure Application Insights for centralized logging.
    
    Args:
        connection_string: Application Insights connection string (optional)
        service_name: Service name for telemetry identification
    
    Returns:
        bool: True if successfully configured
    """
    global _azure_monitor_configured, _logger_provider
    
    if _azure_monitor_configured:
        return True
    
    # Get connection string from config or environment
    if not connection_string:
        app_insights_config = _get_app_insights_config()
        if app_insights_config:
            if not app_insights_config.get("enabled", False):
                return False
            connection_string = app_insights_config.get("connection_string")
        
        if not connection_string:
            connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if not connection_string:
        return False
    
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
        
        resource = Resource.create({
            "service.name": service_name,
            "service.namespace": "deepwiki"
        })
        
        _logger_provider = LoggerProvider(resource=resource)
        set_logger_provider(_logger_provider)
        
        # Use connection string auth (instrumentation key).
        # AAD token auth via ManagedIdentityCredential causes noisy
        # errors when the identity endpoint is slow or misconfigured.
        exporter = AzureMonitorLogExporter(
            connection_string=connection_string,
        )
        
        _logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        
        otel_handler = LoggingHandler(logger_provider=_logger_provider)
        otel_handler.setLevel(logging.INFO)  # Only send INFO+ to App Insights
        logging.getLogger().addHandler(otel_handler)
        
        _azure_monitor_configured = True
        return True
        
    except ImportError:
        return False
    except Exception:
        return False


def flush_application_insights():
    """Force flush pending logs to Application Insights."""
    global _logger_provider
    if _logger_provider:
        try:
            _logger_provider.force_flush()
        except Exception:
            pass


def is_application_insights_enabled() -> bool:
    """Check if Application Insights is enabled."""
    return _azure_monitor_configured


def get_log_filename(prefix: str = "backend") -> str:
    """Generate log filename: prefix-yymmdd.log"""
    return f"{prefix}-{datetime.now().strftime('%y%m%d')}.log"


# Track if logging has been initialized to avoid duplicate setup messages
_logging_initialized = False


def setup_logging(
    log_prefix: str = "backend",
    enable_app_insights: bool = True,
    log_dir: Optional[str] = None,
) -> Path:
    """
    Configure logging with smart filtering and daily rotation.
    
    Features:
    - Level-aware deduplication (ERROR: 2s, WARN: 10s, INFO: 30s, DEBUG: 60s)
    - Sampling for high-frequency logs (prevents flooding)
    - Component labeling [BE] for backend identification
    - Daily log rotation with configurable retention
    - Optional Application Insights integration
    
    Args:
        log_prefix: Prefix for log files (default: "backend")
        enable_app_insights: Enable Azure Application Insights (default: True)
        log_dir: Directory for log files (default: project_root/logs).
                 Useful for AML jobs that write to a specific output dir.
    
    Returns:
        Path to the log file
    
    Environment Variables:
        LOG_LEVEL: Minimum level to log (default: DEBUG)
        LOG_BACKUP_COUNT: Days of logs to retain (default: 30)
    """
    global _logging_initialized
    
    # Setup log directory
    if log_dir:
        log_dir_path = Path(log_dir)
    else:
        base_dir = Path(__file__).parent.parent
        log_dir_path = base_dir / "logs"
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir_path / get_log_filename(log_prefix)
    
    # If already initialized, just return the log file path (skip re-configuration)
    if _logging_initialized:
        return log_file
    
    # Get log level from environment
    level_str = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_str, logging.DEBUG)
    
    # Get backup count
    try:
        backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 30))
    except ValueError:
        backup_count = 30
    
    # Log format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # File handler with daily rotation
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.namer = lambda n: n.replace(".log.", "-") + ".log" if ".log." in n else n
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add smart filter to both handlers
    smart_filter = SmartLogFilter(label="BE")
    file_handler.addFilter(smart_filter)
    console_handler.addFilter(smart_filter)
    
    # Configure root logger
    logging.basicConfig(level=level, handlers=[file_handler, console_handler], force=True)
    
    # Quiet down verbose third-party loggers
    noisy_loggers = [
        "azure", "azure.core", "azure.identity",
        "openai", "openai._base_client",  # Suppress request_id spam
        "httpx", "httpcore",
        "urllib3", "urllib3.connectionpool",  # Very verbose connection logs
        "watchfiles", "watchfiles.main",
        "adalflow", "faiss",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # Mark as initialized before logging to prevent recursion
    _logging_initialized = True
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: level={level_str}, file={log_file}")
    
    # Enable Application Insights if requested
    if enable_app_insights:
        if setup_application_insights(service_name=log_prefix):
            logger.info("Application Insights enabled")
    
    return log_file


# Frontend logging support
_frontend_logger = None


def get_frontend_logger() -> logging.Logger:
    """Get or create the frontend logger."""
    global _frontend_logger
    
    if _frontend_logger is None:
        _frontend_logger = logging.getLogger("frontend")
        _frontend_logger.setLevel(logging.DEBUG)
        _frontend_logger.propagate = False
        
        # Setup log file
        base_dir = Path(__file__).parent.parent
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / get_log_filename("frontend")
        
        try:
            backup_count = int(os.environ.get("LOG_BACKUP_COUNT", 30))
        except ValueError:
            backup_count = 30
        
        handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
        handler.namer = lambda n: n.replace(".log.", "-") + ".log" if ".log." in n else n
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        
        # Add smart filter with FE label
        handler.addFilter(SmartLogFilter(label="FE"))
        
        _frontend_logger.addHandler(handler)
    
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
    
    # Format message with context
    if context:
        ctx_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_msg = f"{message} | {ctx_str}"
    else:
        full_msg = message
    
    level = level.lower()
    if level == "debug":
        logger.debug(full_msg)
    elif level == "info":
        logger.info(full_msg)
    elif level in ("warn", "warning"):
        logger.warning(full_msg)
    elif level == "error":
        logger.error(full_msg)
    else:
        logger.info(full_msg)
