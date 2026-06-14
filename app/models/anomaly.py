from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AnomalyEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    engine_id: str
    reconstruction_error: float
    threshold_warning: float
    threshold_critical: float
    severity: AlertSeverity
    window_start_time: datetime | None = None
    window_end_time: datetime | None = None
    model_version: str
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
