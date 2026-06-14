from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Fan-out manager: one Redis reader → N connected WebSocket clients.

    Each client gets its own asyncio.Queue, preventing a slow client from
    blocking the Redis subscriber.  Clients can filter to a specific engine_id.
    """

    def __init__(self) -> None:
        # client_id → (websocket, engine_filter, queue)
        self._clients: dict[str, tuple[WebSocket, str | None, asyncio.Queue[str]]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> asyncio.Queue[str]:
        await websocket.accept()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._clients[client_id] = (websocket, None, queue)
        logger.debug("WebSocket client connected", extra={"client_id": client_id})
        return queue

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)
        logger.debug("WebSocket client disconnected", extra={"client_id": client_id})

    async def set_filter(self, client_id: str, engine_id: str | None) -> None:
        async with self._lock:
            if client_id in self._clients:
                ws, _, q = self._clients[client_id]
                self._clients[client_id] = (ws, engine_id, q)

    async def broadcast(self, payload: str, engine_id: str) -> None:
        """Put *payload* into all client queues that match the engine filter."""
        async with self._lock:
            clients = list(self._clients.values())
        for _ws, engine_filter, queue in clients:
            if engine_filter is None or engine_filter == engine_id:
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass  # slow client — drop the message rather than blocking


_manager = WebSocketManager()


def get_manager() -> WebSocketManager:
    return _manager


async def live_websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint: streams live sensor readings and anomaly alerts."""
    import uuid

    client_id = str(uuid.uuid4())
    manager = get_manager()
    queue = await manager.connect(websocket, client_id)

    # Background task: relay Redis alerts to this client's queue
    publisher = websocket.app.state.alert_publisher

    async def _relay_alerts() -> None:
        async for event in publisher.subscribe():
            payload = json.dumps({
                "type": "anomaly_alert",
                "anomaly_id": str(event.id),
                "engine_id": event.engine_id,
                "severity": event.severity.value,
                "reconstruction_error": event.reconstruction_error,
                "time": event.time.isoformat(),
            })
            await manager.broadcast(payload, event.engine_id)

    relay_task = asyncio.create_task(_relay_alerts())

    try:
        await websocket.send_json({"type": "connected", "client_id": client_id})

        while True:
            # Listen for client commands (subscribe/unsubscribe/ping)
            client_msg_task = asyncio.create_task(websocket.receive_text())
            queue_task = asyncio.create_task(queue.get())

            done, pending = await asyncio.wait(
                [client_msg_task, queue_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            for task in done:
                result = task.result()
                if task is client_msg_task:
                    try:
                        msg = json.loads(result)
                        action = msg.get("action")
                        if action == "subscribe":
                            await manager.set_filter(client_id, msg.get("engine_id"))
                        elif action == "unsubscribe":
                            await manager.set_filter(client_id, None)
                        elif action == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                elif task is queue_task:
                    await websocket.send_text(result)

    except WebSocketDisconnect:
        pass
    finally:
        relay_task.cancel()
        await manager.disconnect(client_id)
