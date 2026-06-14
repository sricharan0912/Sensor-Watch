from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import redis.asyncio as aioredis

from app.models.anomaly import AnomalyEvent

logger = logging.getLogger(__name__)


class AlertPublisher:
    """Publishes anomaly events to a Redis pub/sub channel.

    SSE and WebSocket endpoints subscribe to the same channel for fan-out.
    """

    def __init__(self, redis: aioredis.Redis, channel: str) -> None:  # type: ignore[type-arg]
        self._redis = redis
        self._channel = channel

    async def publish(self, event: AnomalyEvent) -> None:
        payload = event.model_dump_json()
        await self._redis.publish(self._channel, payload)
        logger.debug(
            "Alert published",
            extra={"engine_id": event.engine_id, "severity": event.severity.value},
        )

    async def subscribe(self) -> AsyncIterator[AnomalyEvent]:
        """Yield AnomalyEvent objects as they arrive on the Redis channel."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        yield AnomalyEvent.model_validate_json(message["data"])
                    except Exception as exc:
                        logger.warning("Failed to parse alert message", extra={"error": str(exc)})
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.aclose()
