from __future__ import annotations

from datetime import datetime

import asyncpg  # type: ignore[import-untyped]

from app.models.sensor import SensorReading

_SENSOR_COLS = [
    "time", "engine_id", "cycle",
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
    "s13", "s14", "s15", "s17", "s20", "s21",
    "op_setting_1", "op_setting_2", "op_setting_3",
]


def _reading_to_record(r: SensorReading) -> tuple[object, ...]:
    return (
        r.time, r.engine_id, r.cycle,
        r.s2, r.s3, r.s4, r.s7, r.s8, r.s9, r.s11, r.s12,
        r.s13, r.s14, r.s15, r.s17, r.s20, r.s21,
        r.op_setting_1, r.op_setting_2, r.op_setting_3,
    )


async def insert_batch(pool: asyncpg.Pool, readings: list[SensorReading]) -> int:
    records = [_reading_to_record(r) for r in readings]
    async with pool.acquire() as conn:
        await conn.copy_records_to_table(
            "sensor_readings",
            records=records,
            columns=_SENSOR_COLS,
        )
    return len(records)


async def query_readings(
    pool: asyncpg.Pool,
    engine_id: str,
    start: datetime,
    end: datetime,
    limit: int = 1000,
) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM sensor_readings
            WHERE engine_id = $1 AND time >= $2 AND time <= $3
            ORDER BY time DESC
            LIMIT $4
            """,
            engine_id, start, end, limit,
        )
