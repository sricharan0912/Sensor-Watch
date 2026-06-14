from __future__ import annotations

import logging
import sys
from datetime import UTC
from typing import Any


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback
        from datetime import datetime

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge extra fields passed via logger.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName", "taskName",
            ):
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "asyncpg"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
