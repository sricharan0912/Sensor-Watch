from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query, Request

from app.api.auth import require_api_key
from app.core.exceptions import raise_bad_request
from app.db.repositories import anomaly_repo
from app.models.anomaly import AlertSeverity, AnomalyEvent
from app.models.api import AnomalyListResponse

router = APIRouter()


@router.get(
    "",
    response_model=AnomalyListResponse,
    dependencies=[Depends(require_api_key)],
    summary="Query anomaly events",
)
async def list_anomalies(
    request: Request,
    engine_id: str | None = Query(None),
    severity: AlertSeverity | None = Query(None),
    start: datetime = Query(
        default_factory=lambda: datetime.now(UTC) - timedelta(hours=1)
    ),
    end: datetime = Query(default_factory=lambda: datetime.now(UTC)),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> AnomalyListResponse:
    if start >= end:
        raise_bad_request("'start' must be before 'end'")

    rows, total = await anomaly_repo.query_anomalies(
        pool=request.app.state.db_pool,
        engine_id=engine_id,
        severity=severity,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )

    anomalies = [
        AnomalyEvent(
            id=row["id"],
            time=row["time"],
            engine_id=row["engine_id"],
            reconstruction_error=row["reconstruction_error"],
            threshold_warning=row["threshold_warning"],
            threshold_critical=row["threshold_critical"],
            severity=AlertSeverity(row["severity"]),
            window_start_time=row["window_start_time"],
            window_end_time=row["window_end_time"],
            model_version=row["model_version"],
            acknowledged=row["acknowledged"],
            acknowledged_at=row["acknowledged_at"],
        )
        for row in rows
    ]

    return AnomalyListResponse(
        anomalies=anomalies,
        total=total,
        limit=limit,
        offset=offset,
    )
