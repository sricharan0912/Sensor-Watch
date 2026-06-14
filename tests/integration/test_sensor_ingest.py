from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.models.anomaly import AlertSeverity


class TestSensorIngest:
    def _reading_payload(self, engine_id: str = "engine_1", cycle: int = 1) -> dict:  # type: ignore[type-arg]
        return {
            "engine_id": engine_id,
            "time": datetime.now(UTC).isoformat(),
            "cycle": cycle,
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 0.0,
            **{f"s{col}": 0.5 for col in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]},
        }

    def test_single_reading_accepted(self, client: TestClient) -> None:
        with patch("app.db.repositories.sensor_repo.insert_batch", new_callable=AsyncMock):
            response = client.post(
                "/api/v1/sensor/ingest",
                json=self._reading_payload(),
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] == 1

    def test_batch_ingest_accepted(self, client: TestClient) -> None:
        batch = {"readings": [self._reading_payload(cycle=i) for i in range(1, 6)]}
        with patch("app.db.repositories.sensor_repo.insert_batch", new_callable=AsyncMock):
            response = client.post(
                "/api/v1/sensor/ingest",
                json=batch,
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] == 5

    def test_ingest_requires_api_key(self, client: TestClient) -> None:
        response = client.post("/api/v1/sensor/ingest", json=self._reading_payload())
        assert response.status_code == 401

    def test_ingest_wrong_api_key(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/sensor/ingest",
            json=self._reading_payload(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_window_buffer_populated_after_ingest(self, client: TestClient) -> None:
        with patch("app.db.repositories.sensor_repo.insert_batch", new_callable=AsyncMock):
            for i in range(1, 4):
                client.post(
                    "/api/v1/sensor/ingest",
                    json=self._reading_payload(engine_id="engine_buf_test", cycle=i),
                    headers={"X-API-Key": "test-key"},
                )
        # Buffer should contain entries for this engine
        assert "engine_buf_test" in client.app.state.window_buffers
        assert client.app.state.window_buffers["engine_buf_test"].fill_level == 3

    def test_anomaly_detected_when_model_returns_critical(self, client: TestClient) -> None:
        from app.detection.inference import AnomalyScore

        # Override detector to return a CRITICAL score
        client.app.state.detector.predict.return_value = AnomalyScore(
            reconstruction_error=0.050, severity=AlertSeverity.CRITICAL
        )

        # Pre-fill the buffer with 29 readings so the 30th triggers inference
        buf_engine = "engine_critical_test"
        with patch("app.db.repositories.sensor_repo.insert_batch", new_callable=AsyncMock), \
             patch("app.db.repositories.anomaly_repo.insert_anomaly", new_callable=AsyncMock, return_value="uuid-1"):
            for i in range(1, 30):
                client.post(
                    "/api/v1/sensor/ingest",
                    json=self._reading_payload(engine_id=buf_engine, cycle=i),
                    headers={"X-API-Key": "test-key"},
                )
            response = client.post(
                "/api/v1/sensor/ingest",
                json=self._reading_payload(engine_id=buf_engine, cycle=30),
                headers={"X-API-Key": "test-key"},
            )

        assert response.status_code == 202
        body = response.json()
        assert body["anomalies_detected"] == 1
        assert body["queued_windows"] >= 1
