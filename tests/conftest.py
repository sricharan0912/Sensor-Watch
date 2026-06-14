from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.detection.inference import AnomalyDetector, AnomalyScore, ModelMetadata
from app.models.anomaly import AlertSeverity
from app.services.metrics_service import MetricsService

# ── Settings fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        api_key="test-key",
        timescaledb_url="postgresql://sensorwatch:sensorwatch@localhost:5432/sensorwatch_test",
        redis_url="redis://localhost:6379/15",
        artifacts_dir=Path("tests/fixtures/artifacts"),
        environment="development",
    )


# ── Model metadata fixture ────────────────────────────────────────────────────

@pytest.fixture
def model_metadata() -> ModelMetadata:
    return ModelMetadata(
        subset="FD001",
        window_size=30,
        latent_dim=32,
        hidden_size=64,
        n_features=14,
        sensor_cols=["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
        threshold_warning=0.005,
        threshold_critical=0.010,
        val_loss=0.002,
        n_train_windows=10000,
        trained_at="2026-05-16T10:00:00Z",
        model_version="FD001-20260516T100000Z",
    )


# ── Mock detector ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_detector(model_metadata: ModelMetadata) -> AnomalyDetector:
    detector = MagicMock(spec=AnomalyDetector)
    detector.metadata = model_metadata
    detector.predict.return_value = AnomalyScore(
        reconstruction_error=0.001, severity=AlertSeverity.NORMAL
    )
    detector.normalize.side_effect = lambda values: [v * 0.5 for v in values]
    return detector


# ── Mock DB pool ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db_pool() -> AsyncMock:
    pool = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


# ── Mock Redis ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.ping.return_value = True
    redis.publish.return_value = 1
    return redis


# ── Mock alert publisher ──────────────────────────────────────────────────────

@pytest.fixture
def mock_alert_publisher() -> AsyncMock:
    publisher = AsyncMock()
    publisher.publish.return_value = None
    return publisher


# ── Test client ───────────────────────────────────────────────────────────────

@pytest.fixture
def client(
    test_settings: Settings,
    mock_detector: AnomalyDetector,
    mock_db_pool: AsyncMock,
    mock_redis: AsyncMock,
    mock_alert_publisher: AsyncMock,
) -> TestClient:
    from app.main import create_app

    app = create_app(settings=test_settings)

    # Override the cached get_settings dependency so auth + config use test_settings
    app.dependency_overrides[get_settings] = lambda: test_settings

    # Patch lifespan so it doesn't try to connect to real DB / Redis
    # create_pool and aioredis.from_url are coroutines — use AsyncMock
    with patch("app.main.create_pool", new=AsyncMock(return_value=mock_db_pool)), \
         patch("app.main.run_migrations", new=AsyncMock()), \
         patch("app.main.aioredis.from_url", new=AsyncMock(return_value=mock_redis)), \
         patch("app.main.AnomalyDetector.load", return_value=mock_detector):
        with TestClient(app, raise_server_exceptions=True) as c:
            # Inject state after lifespan runs (lifespan may overwrite, so re-inject)
            c.app.state.detector = mock_detector
            c.app.state.db_pool = mock_db_pool
            c.app.state.redis = mock_redis
            c.app.state.alert_publisher = mock_alert_publisher
            c.app.state.window_buffers = {}
            c.app.state.metrics = MetricsService()
            c.app.state.settings = test_settings
            yield c
