from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.auth import require_api_key
from app.models.api import AlertSSEEvent

router = APIRouter()
logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 15  # seconds


@router.get(
    "/stream",
    dependencies=[Depends(require_api_key)],
    summary="SSE stream of anomaly alerts",
    description=(
        "Server-sent event stream. "
        "Emits a heartbeat every 15s and an 'alert' event for each anomaly."
    ),
)
async def alert_stream(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _sse_generator(request: Request):  # type: ignore[return]
    publisher = request.app.state.alert_publisher

    async def _heartbeat() -> str:
        event = AlertSSEEvent(event_type="heartbeat")
        return f"event: heartbeat\ndata: {event.model_dump_json()}\n\n"

    async def _alert(payload: str) -> str:
        return f"event: alert\ndata: {payload}\n\n"

    # Subscribe to Redis pub/sub in background
    alert_queue: asyncio.Queue[str] = asyncio.Queue()

    async def _reader() -> None:
        async for anomaly_event in publisher.subscribe():
            await alert_queue.put(anomaly_event.model_dump_json())

    reader_task = asyncio.create_task(_reader())

    try:
        yield await _heartbeat()
        heartbeat_deadline = asyncio.get_event_loop().time() + _HEARTBEAT_INTERVAL

        while True:
            if await request.is_disconnected():
                break

            now = asyncio.get_event_loop().time()
            timeout = max(0.1, heartbeat_deadline - now)

            try:
                payload = await asyncio.wait_for(alert_queue.get(), timeout=timeout)
                yield await _alert(payload)
            except TimeoutError:
                yield await _heartbeat()
                heartbeat_deadline = asyncio.get_event_loop().time() + _HEARTBEAT_INTERVAL

    except asyncio.CancelledError:
        pass
    finally:
        reader_task.cancel()
        logger.debug("SSE client disconnected")
