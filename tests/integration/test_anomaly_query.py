from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


def _make_mock_row(**overrides: object) -> MagicMock:
    defaults = {
        "id": uuid4(),
        "time": datetime.now(UTC),
        "engine_id": "engine_1",
        "reconstruction_error": 0.007,
        "threshold_warning": 0.005,
        "threshold_critical": 0.010,
        "severity": "WARNING",
        "window_start_time": None,
        "window_end_time": None,
        "model_version": "test-v1",
        "acknowledged": False,
        "acknowledged_at": None,
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, key: defaults[key]
    return row


class TestAnomalyQuery:
    def test_list_anomalies_requires_api_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/anomalies")
        assert response.status_code == 401

    def test_list_anomalies_returns_paginated_response(self, client: TestClient) -> None:
        mock_rows = [_make_mock_row(severity="WARNING"), _make_mock_row(severity="CRITICAL")]
        with patch(
            "app.db.repositories.anomaly_repo.query_anomalies",
            new_callable=AsyncMock,
            return_value=(mock_rows, 2),
        ):
            response = client.get(
                "/api/v1/anomalies",
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["limit"] == 100
        assert body["offset"] == 0
        assert len(body["anomalies"]) == 2

    def test_list_anomalies_passes_engine_filter(self, client: TestClient) -> None:
        with patch(
            "app.db.repositories.anomaly_repo.query_anomalies",
            new_callable=AsyncMock,
            return_value=([], 0),
        ) as mock_query:
            response = client.get(
                "/api/v1/anomalies?engine_id=engine_42",
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 200
        mock_query.assert_called_once()
        # Verify engine_id was forwarded to the repository (positional or keyword)
        call_args = str(mock_query.call_args)
        assert "engine_42" in call_args

    def test_list_anomalies_empty_result(self, client: TestClient) -> None:
        with patch(
            "app.db.repositories.anomaly_repo.query_anomalies",
            new_callable=AsyncMock,
            return_value=([], 0),
        ):
            response = client.get(
                "/api/v1/anomalies",
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["anomalies"] == []

    def test_invalid_date_range_returns_400(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/anomalies?start=2026-05-16T12:00:00Z&end=2026-05-16T11:00:00Z",
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 400
