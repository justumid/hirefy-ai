import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGING_AVAILABLE = True
except ImportError:
    JSON_LOGGING_AVAILABLE = False

# === Configurable via ENV ===
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
USE_JSON_LOGGING = os.getenv("USE_JSON_LOGGING", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
SERVICE_NAME = os.getenv("SERVICE_NAME", "hirelyai")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")

LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_formatter(use_json: bool, service: str) -> logging.Formatter:
    if use_json and JSON_LOGGING_AVAILABLE:
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s %(service)s %(environment)s',
            rename_fields={"levelname": "level"}
        )
        formatter._required_fields.update(['service', 'environment'])
        return formatter
    else:
        return logging.Formatter(
            fmt=f"%(asctime)s | %(levelname)s | %(name)s | [svc={service}] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = LOG_LEVEL,
    use_json: bool = USE_JSON_LOGGING,
    service: str = SERVICE_NAME
) -> logging.Logger:
    """
    Sets up a logger with rotating file + stream handler
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid double logging in root

    # Clear old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = get_formatter(use_json, service)

    # === Stream Handler (STDOUT) ===
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # === File Handler (Rotating) ===
    if log_file:
        file_path = LOG_DIR / log_file
        file_handler = RotatingFileHandler(str(file_path), maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # === Sentry Integration (Optional) ===
    if SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.ERROR,
                event_level=logging.ERROR
            )
            sentry_sdk.init(
                dsn=SENTRY_DSN,
                environment=ENVIRONMENT,
                integrations=[sentry_logging],
                traces_sample_rate=0.05,
                send_default_pii=True
            )
            logger.info("[Logging] Sentry integration initialized.")
        except ImportError:
            logger.warning("[Logging] Sentry SDK not installed.")

    return logger

# === Preconfigured loggers ===
app_logger = setup_logger("app", log_file="app.log")
interview_logger = setup_logger("interview_bot", log_file="interview.log")
scoring_logger = setup_logger("interview_scoring", log_file="scoring.log")
transcription_logger = setup_logger("interview_transcription", log_file="transcription.log")

# Usage:
# logger = logging.getLogger("app")
# logger.info("Service started", extra={"service": SERVICE_NAME, "environment": ENVIRONMENT})
