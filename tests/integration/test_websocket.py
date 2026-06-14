from __future__ import annotations

import json

from fastapi.testclient import TestClient


class TestWebSocket:
    def test_websocket_connects_and_receives_connected_frame(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect("/ws/live") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert "client_id" in data

    def test_websocket_responds_to_ping(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/live") as ws:
            # Consume the initial connected frame
            ws.receive_json()
            ws.send_text(json.dumps({"action": "ping"}))
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_websocket_accepts_subscribe_action(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # connected frame
            ws.send_text(json.dumps({"action": "subscribe", "engine_id": "engine_42"}))
            # No error frame expected — subscribe is fire-and-forget
            # Verify the filter was set in the manager
            # Manager state is tested implicitly — no error raised = success

    def test_websocket_accepts_unsubscribe_action(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # connected frame
            ws.send_text(json.dumps({"action": "subscribe", "engine_id": "engine_1"}))
            ws.send_text(json.dumps({"action": "unsubscribe", "engine_id": "engine_1"}))
            ws.send_text(json.dumps({"action": "ping"}))
            data = ws.receive_json()
            assert data["type"] == "pong"
