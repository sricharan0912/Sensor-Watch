# SensorWatch — Time Series Anomaly Detection Pipeline

End-to-end ML pipeline that ingests streaming sensor telemetry, detects anomalies with an LSTM Autoencoder, stores results in TimescaleDB, and serves alerts via a FastAPI REST API with SSE and WebSocket streaming.

**Dataset:** NASA CMAPSS turbofan engine degradation (FD001) — 14 sensor channels, 30-step sliding windows  
**Model:** LSTM Autoencoder (~64k parameters, ~256KB) — deployable to Raspberry Pi / Jetson Nano via ONNX export  
**Stack:** PyTorch · FastAPI · TimescaleDB · Redis · Docker · Grafana

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Sensor Simulator                                                │
│  (NASA CMAPSS replay @ 10Hz via HTTP POST)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                     POST /api/v1/sensor/ingest
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  FastAPI — SensorWatch                                           │
│                                                                  │
│  IngestService                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ asyncpg bulk     │  │ SlidingWindow    │  │ LSTM          │ │
│  │ INSERT           │  │ Buffer (per      │  │ Autoencoder   │ │
│  │ (sensor_readings)│  │ engine_id)       │  │ inference     │ │
│  └──────────────────┘  └──────────────────┘  └───────┬───────┘ │
│                                                       │         │
│           reconstruction_error > threshold?           │         │
│                          YES ──────────────────────── ▼         │
│                    ┌──────────────────────────────────────────┐ │
│                    │  INSERT anomaly_events                   │ │
│                    │  Redis PUBLISH sensorwatch:alerts        │ │
│                    └──────────────────────────────────────────┘ │
│                                                                  │
│  GET /api/v1/anomalies    ─── TimescaleDB query                 │
│  GET /api/v1/alerts/stream ── Redis sub → SSE fan-out           │
│  WS  /ws/live             ── Redis sub → WebSocket fan-out      │
│  GET /api/v1/health       ── DB + Redis + model status          │
│  GET /api/v1/metrics      ── in-process counters                │
└─────────────────────────────────────────────────────────────────┘
         │                         │
  TimescaleDB                    Redis
  (hypertables,                  (pub/sub,
   retention,                    128MB LRU)
   cont. aggregates)
```

---

## Quick Start

### 1. Train the model

```bash
# Install dependencies
poetry install

# Download CMAPSS + train LSTM Autoencoder + calibrate thresholds
make train
# → artifacts/model.pt, scaler.joblib, model_metadata.json (~8–12 min on CPU)
```

### 2. Start all services

```bash
cp .env.example .env
make dev
# → TimescaleDB:5432, Redis:6379, SensorWatch API:8000
```

### 3. Health check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "dependencies": {"timescaledb": "ok", "redis": "ok", "model_loaded": true},
  "model_metadata": {"threshold_warning": 0.0045, "threshold_critical": 0.0089, ...}
}
```

### 4. Run the sensor simulator

```bash
make simulate
# Replays CMAPSS test sequences at 10Hz; injects degraded engine to trigger alerts
```

### 5. Query anomalies

```bash
curl "http://localhost:8000/api/v1/anomalies?severity=CRITICAL" \
  -H "X-API-Key: dev-key"
```

### 6. Subscribe to real-time alerts (SSE)

```bash
curl -N "http://localhost:8000/api/v1/alerts/stream" -H "X-API-Key: dev-key"
```

### 7. WebSocket live feed

```bash
# Install wscat: npm i -g wscat
wscat -c ws://localhost:8000/ws/live
# → Send: {"action": "subscribe", "engine_id": "engine_1"}
```

### 8. Grafana dashboard

```bash
make grafana
# → http://localhost:3000  (admin / admin)
```

---

## API Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/sensor/ingest` | ✓ | Single or batch (≤500) sensor readings |
| GET | `/api/v1/anomalies` | ✓ | Query anomaly events (engine_id, severity, time range) |
| GET | `/api/v1/alerts/stream` | ✓ | SSE stream — heartbeat 15s + alert events |
| WS | `/ws/live` | — | WebSocket — real-time sensor + anomaly frames |
| GET | `/api/v1/health` | — | Service health (DB, Redis, model) |
| GET | `/api/v1/health/metrics` | — | Pipeline counters and inference latency |

Interactive docs: http://localhost:8000/docs

---

## ML Model

**Architecture:** LSTM Autoencoder  
- Encoder: LSTM(14→64) → Dropout(0.2) → LSTM(64→32) → latent z (32-dim)  
- Decoder: repeat z × 30 → LSTM(32→32) → Dropout(0.2) → LSTM(32→64) → Linear(64→14)  
- Loss: MSE(input, reconstruction) — reconstruction error is the anomaly score

**Dataset:** NASA CMAPSS FD001 — 100 training engines, 14 sensor channels, normal cycles (RUL > 125)

**Thresholds:** Calibrated on normal-data reconstruction errors  
- WARNING: 90th percentile  
- CRITICAL: 99th percentile

**Edge deployment:** Export to ONNX for inference on Raspberry Pi / Jetson Nano:
```bash
poetry run python -m ml.scripts.export_onnx  # → artifacts/model.onnx (~256KB)
```

---

## Development

```bash
make test        # pytest with coverage
make lint        # ruff linter
make fmt         # ruff formatter
make typecheck   # mypy strict
make evaluate    # precision/recall on CMAPSS test set
```

---

## Project Structure

```
sensorwatch/
├── app/            FastAPI application (API, detection, DB, services)
├── ml/             ML training pipeline (data, model, training, scripts)
├── simulator/      Async CMAPSS replay producer
├── infra/          Grafana + TimescaleDB configuration
├── tests/          Unit + integration tests
├── artifacts/      Trained model weights (git-ignored)
└── docker-compose.yml
```
