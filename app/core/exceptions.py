from __future__ import annotations

from fastapi import HTTPException, status


class SensorWatchError(Exception):
    """Base domain error."""


class ModelNotLoadedError(SensorWatchError):
    """Raised when the anomaly detector model is not initialised."""


class DatabaseError(SensorWatchError):
    """Raised on unrecoverable DB errors."""


class BatchTooLargeError(SensorWatchError):
    def __init__(self, size: int, max_size: int) -> None:
        super().__init__(f"Batch size {size} exceeds maximum {max_size}")
        self.size = size
        self.max_size = max_size


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def raise_not_found(detail: str = "Resource not found") -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def raise_bad_request(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def raise_service_unavailable(detail: str = "Service temporarily unavailable") -> None:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
