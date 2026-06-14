from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single sensor reading from one engine unit at one point in time."""

    engine_id: str = Field(..., description="Sensor unit / engine identifier")
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cycle: int = Field(..., ge=1)

    # Operational settings
    op_setting_1: float = 0.0
    op_setting_2: float = 0.0
    op_setting_3: float = 0.0

    # 14 informative CMAPSS sensor channels
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    s7: float = 0.0
    s8: float = 0.0
    s9: float = 0.0
    s11: float = 0.0
    s12: float = 0.0
    s13: float = 0.0
    s14: float = 0.0
    s15: float = 0.0
    s17: float = 0.0
    s20: float = 0.0
    s21: float = 0.0

    def sensor_values(self, sensor_cols: list[str]) -> list[float]:
        """Return ordered sensor values matching the training sensor_cols list."""
        return [getattr(self, col) for col in sensor_cols]


class SensorBatch(BaseModel):
    readings: list[SensorReading] = Field(..., max_length=500)
