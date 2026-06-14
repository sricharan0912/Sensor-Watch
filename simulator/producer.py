"""Async sensor data producer — replays NASA CMAPSS sequences via HTTP POST.

Usage:
    python -m simulator.producer [--api-url URL] [--hz N] [--unit N] [--inject-anomaly]

The producer replays one test unit's readings from the CMAPSS dataset,
then (if --inject-anomaly) replays a degraded engine unit to trigger alerts.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:8000"
_INGEST_ENDPOINT = "/api/v1/sensor/ingest"

# Sensor columns in CMAPSS order (matching the SensorReading model)
_SENSOR_COLS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


def _load_data(data_dir: Path, subset: str = "FD001"):  # type: ignore[return]  # noqa: ANN201
    import pandas as pd

    from ml.data.cmapss_loader import _DROP_SENSORS, _RAW_COLS, download_cmapss

    data_path = download_cmapss(data_dir)

    def _find(name: str) -> Path:
        candidates = list(data_path.rglob(name))
        if not candidates:
            # Try adjacent directory
            candidates = list(data_dir.rglob(name))
        if not candidates:
            raise FileNotFoundError(f"{name} not found in {data_dir}")
        return candidates[0]

    train_path = _find(f"train_{subset}.txt")
    df = pd.read_csv(train_path, sep=r"\s+", header=None, names=_RAW_COLS)
    return df.drop(columns=list(_DROP_SENSORS), errors="ignore")


class SensorProducer:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        hz: float = 10.0,
        engine_unit: int = 1,
        inject_anomaly: bool = True,
        data_dir: Path = Path("data/"),
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._interval = 1.0 / hz
        self._engine_unit = engine_unit
        self._inject_anomaly = inject_anomaly
        self._data_dir = data_dir

    async def run(self) -> None:
        logger.info(
            "SensorProducer starting",
            extra={
                "api_url": self._api_url,
                "hz": 1.0 / self._interval,
                "engine_unit": self._engine_unit,
            },
        )


        df = _load_data(self._data_dir)

        # Select the target engine unit
        unit_df = df[df["unit"] == self._engine_unit].copy()

        if self._inject_anomaly:
            # Also replay a heavily degraded unit (last 30 cycles of engine 1)
            max_cycle = df.groupby("unit")["cycle"].max()
            unit_df["rul"] = max_cycle[unit_df["unit"].values].values - unit_df["cycle"].values
            anomaly_unit = df[df["unit"] == 1].copy()
            anomaly_unit_last = anomaly_unit.tail(60)
            # Rename to a different engine_id so the API sees it as a separate sensor
            anomaly_rows = anomaly_unit_last.copy()
            anomaly_engine_id = f"engine_degraded_{self._engine_unit}"
        else:
            anomaly_engine_id = None
            anomaly_rows = None

        async with httpx.AsyncClient(timeout=10.0) as client:
            cycle_idx = 0
            total_sent = 0
            total_errors = 0

            all_rows = list(unit_df.itertuples())
            anomaly_row_list = list(anomaly_rows.itertuples()) if anomaly_rows is not None else []
            combined = list(zip(all_rows, anomaly_row_list)) if anomaly_rows is not None else [(r, None) for r in all_rows]

            logger.info(f"Replaying {len(all_rows)} cycles from engine unit {self._engine_unit}")

            for normal_row, anomaly_row in combined:
                now = datetime.now(UTC).isoformat()

                reading = {
                    "engine_id": f"engine_{self._engine_unit}",
                    "time": now,
                    "cycle": int(normal_row.cycle),
                    "op_setting_1": float(normal_row.op_setting_1),
                    "op_setting_2": float(normal_row.op_setting_2),
                    "op_setting_3": float(normal_row.op_setting_3),
                }
                for col in _SENSOR_COLS:
                    reading[col] = float(getattr(normal_row, col, 0.0))

                readings = [reading]

                if anomaly_row is not None:
                    anomaly_reading = {
                        "engine_id": anomaly_engine_id,
                        "time": now,
                        "cycle": int(anomaly_row.cycle),
                        "op_setting_1": float(anomaly_row.op_setting_1),
                        "op_setting_2": float(anomaly_row.op_setting_2),
                        "op_setting_3": float(anomaly_row.op_setting_3),
                    }
                    for col in _SENSOR_COLS:
                        anomaly_reading[col] = float(getattr(anomaly_row, col, 0.0))
                    readings.append(anomaly_reading)

                # POST batch
                try:
                    await self._post(client, {"readings": readings})
                    total_sent += len(readings)
                    cycle_idx += 1
                    if cycle_idx % 50 == 0:
                        logger.info(f"Sent {total_sent} readings ({cycle_idx} cycles)")
                except Exception as exc:
                    total_errors += 1
                    logger.warning(f"Failed to send reading: {exc}")

                await asyncio.sleep(self._interval)

            logger.info(
                "Simulation complete",
                extra={"total_sent": total_sent, "errors": total_errors},
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=5))
    async def _post(self, client: httpx.AsyncClient, body: dict) -> None:  # type: ignore[type-arg]
        response = await client.post(
            f"{self._api_url}{_INGEST_ENDPOINT}",
            json=body,
            headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
        )
        response.raise_for_status()


def parse_args() -> argparse.Namespace:
    import os
    p = argparse.ArgumentParser(description="CMAPSS sensor data simulator")
    p.add_argument("--api-url", default=os.environ.get("API_URL", _DEFAULT_API_URL))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", "dev-key"))
    p.add_argument("--hz", default=10.0, type=float)
    p.add_argument("--unit", default=1, type=int, dest="engine_unit")
    p.add_argument("--inject-anomaly", action="store_true", default=True)
    p.add_argument("--data-dir", default="data/", type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    producer = SensorProducer(
        api_url=args.api_url,
        api_key=args.api_key,
        hz=args.hz,
        engine_unit=args.engine_unit,
        inject_anomaly=args.inject_anomaly,
        data_dir=args.data_dir,
    )
    asyncio.run(producer.run())
