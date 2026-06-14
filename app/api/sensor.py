from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.api.auth import require_api_key
from app.config import Settings, get_settings
from app.core.exceptions import raise_bad_request
from app.models.api import IngestResponse
from app.models.sensor import SensorBatch, SensorReading
from app.services import ingest_service

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
    summary="Ingest sensor readings",
    description="Accept a single SensorReading or a SensorBatch (up to 500 readings).",
)
async def ingest_readings(
    request: Request,
    body: SensorReading | SensorBatch,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    readings = body.readings if isinstance(body, SensorBatch) else [body]

    if len(readings) > settings.max_batch_size:
        raise_bad_request(
            f"Batch size {len(readings)} exceeds maximum {settings.max_batch_size}"
        )

    return await ingest_service.ingest(
        readings=readings,
        pool=request.app.state.db_pool,
        detector=request.app.state.detector,
        window_buffers=request.app.state.window_buffers,
        alert_publisher=request.app.state.alert_publisher,
        metrics=request.app.state.metrics,
    )
