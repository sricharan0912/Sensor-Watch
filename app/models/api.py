from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.anomaly import AnomalyEvent

# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    accepted: int
    queued_windows: int
    anomalies_detected: int


# ── Anomaly query ─────────────────────────────────────────────────────────────

class AnomalyListResponse(BaseModel):
    anomalies: list[AnomalyEvent]
    total: int
    limit: int
    offset: int


# ── SSE alert event ───────────────────────────────────────────────────────────

class AlertSSEEvent(BaseModel):
    event_type: Literal["alert", "heartbeat"] = "alert"
    anomaly: AnomalyEvent | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Health ────────────────────────────────────────────────────────────────────

class DependencyStatus(BaseModel):
    timescaledb: str
    redis: str
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    dependencies: DependencyStatus
    model_metadata: dict[str, Any] | None = None


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    total_readings_ingested: int
    total_windows_processed: int
    total_anomalies_detected: int
    anomalies_by_severity: dict[str, int]
    avg_inference_latency_ms: float
    uptime_seconds: float
