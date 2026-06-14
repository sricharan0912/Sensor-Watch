from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoints:
    def test_liveness_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/v1/health/liveness")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_endpoint_structure(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "version" in body
        assert "environment" in body
        assert "dependencies" in body
        deps = body["dependencies"]
        assert "timescaledb" in deps
        assert "redis" in deps
        assert "model_loaded" in deps

    def test_metrics_endpoint_structure(self, client: TestClient) -> None:
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "total_readings_ingested" in body
        assert "total_windows_processed" in body
        assert "total_anomalies_detected" in body
        assert "anomalies_by_severity" in body
        assert "avg_inference_latency_ms" in body
        assert "uptime_seconds" in body

    def test_metrics_initial_counters_are_zero(self, client: TestClient) -> None:
        response = client.get("/api/v1/health/metrics")
        body = response.json()
        assert body["total_readings_ingested"] == 0
        assert body["total_anomalies_detected"] == 0
