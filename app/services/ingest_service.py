from __future__ import annotations

import logging
import time

import asyncpg  # type: ignore[import-untyped]

from app.db.repositories import anomaly_repo, sensor_repo
from app.detection.inference import AnomalyDetector
from app.detection.window_buffer import SlidingWindowBuffer
from app.models.anomaly import AlertSeverity, AnomalyEvent
from app.models.api import IngestResponse
from app.models.sensor import SensorReading
from app.services.alert_publisher import AlertPublisher
from app.services.metrics_service import MetricsService

logger = logging.getLogger(__name__)


async def ingest(
    readings: list[SensorReading],
    pool: asyncpg.Pool,
    detector: AnomalyDetector,
    window_buffers: dict[str, SlidingWindowBuffer],
    alert_publisher: AlertPublisher,
    metrics: MetricsService,
) -> IngestResponse:
    """Persist readings, run sliding-window inference, publish alerts.

    Returns an IngestResponse with counts of accepted readings and anomalies.
    """
    # ── 1. Persist raw readings ───────────────────────────────────────────────
    await sensor_repo.insert_batch(pool, readings)
    metrics.record_readings(len(readings))

    queued_windows = 0
    anomalies_detected = 0

    # ── 2. Feed each reading into the per-engine window buffer ────────────────
    for reading in readings:
        engine_id = reading.engine_id

        if engine_id not in window_buffers:
            window_buffers[engine_id] = SlidingWindowBuffer(
                window_size=detector.metadata.window_size,
                n_features=detector.metadata.n_features,
            )

        buf = window_buffers[engine_id]

        # Normalise sensor values using the fitted scaler
        raw_values = reading.sensor_values(detector.metadata.sensor_cols)
        normalised = detector.normalize(raw_values)

        t0 = time.perf_counter()
        window = buf.push(normalised)
        if window is None:
            continue

        queued_windows += 1

        # ── 3. Run inference ──────────────────────────────────────────────────
        score = detector.predict(window)
        inference_ms = (time.perf_counter() - t0) * 1000
        metrics.record_window(inference_ms)

        logger.debug(
            "Inference complete",
            extra={
                "engine_id": engine_id,
                "error": round(score.reconstruction_error, 6),
                "severity": score.severity.value,
                "inference_ms": round(inference_ms, 2),
            },
        )

        # ── 4. Persist and publish non-normal events ──────────────────────────
        if score.severity != AlertSeverity.NORMAL:
            event = AnomalyEvent(
                engine_id=engine_id,
                time=reading.time,
                reconstruction_error=score.reconstruction_error,
                threshold_warning=detector.metadata.threshold_warning,
                threshold_critical=detector.metadata.threshold_critical,
                severity=score.severity,
                model_version=detector.metadata.model_version,
            )
            await anomaly_repo.insert_anomaly(pool, event)
            await alert_publisher.publish(event)
            metrics.record_anomaly(score.severity)
            anomalies_detected += 1

            logger.info(
                "Anomaly detected",
                extra={
                    "engine_id": engine_id,
                    "severity": score.severity.value,
                    "error": round(score.reconstruction_error, 6),
                },
            )

    return IngestResponse(
        accepted=len(readings),
        queued_windows=queued_windows,
        anomalies_detected=anomalies_detected,
    )
