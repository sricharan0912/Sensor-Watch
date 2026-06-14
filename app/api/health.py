from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.api import DependencyStatus, HealthResponse, MetricsResponse

router = APIRouter()
logger = logging.getLogger(__name__)

_VERSION = "1.0.0"


@router.get("/liveness", include_in_schema=False)
async def liveness() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get(
    "",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings

    # TimescaleDB
    db_status = "ok"
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as exc:
        db_status = f"error: {exc}"
        logger.warning("DB health check failed", extra={"error": str(exc)})

    # Redis
    redis_status = "ok"
    try:
        await request.app.state.redis.ping()
    except Exception as exc:
        redis_status = f"error: {exc}"
        logger.warning("Redis health check failed", extra={"error": str(exc)})

    detector = request.app.state.detector
    model_loaded = detector is not None
    model_meta = None
    if model_loaded:
        m = detector.metadata
        model_meta = {
            "subset": m.subset,
            "model_version": m.model_version,
            "threshold_warning": m.threshold_warning,
            "threshold_critical": m.threshold_critical,
            "val_loss": m.val_loss,
            "trained_at": m.trained_at,
        }

    overall = "ok" if db_status == "ok" and redis_status == "ok" and model_loaded else "degraded"

    return HealthResponse(
        status=overall,
        version=_VERSION,
        environment=settings.environment,
        dependencies=DependencyStatus(
            timescaledb=db_status,
            redis=redis_status,
            model_loaded=model_loaded,
        ),
        model_metadata=model_meta,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Pipeline metrics",
)
async def metrics(request: Request) -> MetricsResponse:
    m = request.app.state.metrics
    return MetricsResponse(
        total_readings_ingested=m.total_readings,
        total_windows_processed=m.total_windows,
        total_anomalies_detected=m.total_anomalies,
        anomalies_by_severity=m.anomalies_by_severity,
        avg_inference_latency_ms=round(m.avg_inference_ms, 3),
        uptime_seconds=round(m.uptime_seconds, 1),
    )
