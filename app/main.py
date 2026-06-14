"""SensorWatch — FastAPI application factory."""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.websocket import live_websocket_endpoint
from app.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db.pool import create_pool, run_migrations
from app.detection.inference import AnomalyDetector
from app.detection.window_buffer import SlidingWindowBuffer
from app.services.alert_publisher import AlertPublisher
from app.services.metrics_service import MetricsService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        setup_logging(settings.log_level)

        app.state.settings = settings

        # ── Database ──────────────────────────────────────────────────────────
        app.state.db_pool = await create_pool(settings.timescaledb_url)
        await run_migrations(app.state.db_pool)
        logger.info("TimescaleDB pool ready")

        # ── ML model ──────────────────────────────────────────────────────────
        try:
            app.state.detector = AnomalyDetector.load(settings.artifacts_dir)
        except FileNotFoundError as exc:
            logger.warning(str(exc), extra={"hint": "Run 'make train' to train the model"})
            app.state.detector = None

        # ── Redis ─────────────────────────────────────────────────────────────
        app.state.redis = await aioredis.from_url(
            settings.redis_url, encoding="utf-8", decode_responses=True
        )
        app.state.alert_publisher = AlertPublisher(
            app.state.redis, settings.redis_alert_channel
        )
        logger.info("Redis connection ready")

        # ── In-process state ──────────────────────────────────────────────────
        app.state.window_buffers: dict[str, SlidingWindowBuffer] = {}
        app.state.metrics = MetricsService()

        logger.info(
            "SensorWatch started",
            extra={"environment": settings.environment, "version": "1.0.0"},
        )

        yield

        await app.state.db_pool.close()
        await app.state.redis.aclose()
        logger.info("SensorWatch shut down")

    app = FastAPI(
        title="SensorWatch",
        description="Time Series Anomaly Detection Pipeline — LSTM Autoencoder on NASA CMAPSS",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.add_api_websocket_route("/ws/live", live_websocket_endpoint)

    return app


app = create_app()
