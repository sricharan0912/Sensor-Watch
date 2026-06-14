from __future__ import annotations

from fastapi import APIRouter

from app.api import alerts, anomalies, health, sensor

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(sensor.router, prefix="/sensor", tags=["sensor"])
api_router.include_router(anomalies.router, prefix="/anomalies", tags=["anomalies"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
