from __future__ import annotations

from datetime import datetime

import asyncpg  # type: ignore[import-untyped]

from app.models.anomaly import AlertSeverity, AnomalyEvent


async def insert_anomaly(pool: asyncpg.Pool, event: AnomalyEvent) -> str:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO anomaly_events (
                time, engine_id, reconstruction_error,
                threshold_warning, threshold_critical, severity,
                window_start_time, window_end_time, model_version
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            event.time,
            event.engine_id,
            event.reconstruction_error,
            event.threshold_warning,
            event.threshold_critical,
            event.severity.value,
            event.window_start_time,
            event.window_end_time,
            event.model_version,
        )
    return str(row["id"])  # type: ignore[index]


async def query_anomalies(
    pool: asyncpg.Pool,
    engine_id: str | None,
    severity: AlertSeverity | None,
    start: datetime,
    end: datetime,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[asyncpg.Record], int]:
    conditions = ["time >= $1", "time <= $2"]
    params: list[object] = [start, end]
    idx = 3

    if engine_id is not None:
        conditions.append(f"engine_id = ${idx}")
        params.append(engine_id)
        idx += 1

    if severity is not None:
        conditions.append(f"severity = ${idx}")
        params.append(severity.value)
        idx += 1

    where_clause = " AND ".join(conditions)

    async with pool.acquire() as conn:
        count_row = await conn.fetchrow(
            f"SELECT COUNT(*) AS total FROM anomaly_events WHERE {where_clause}",
            *params,
        )
        total: int = count_row["total"]  # type: ignore[index]

        rows = await conn.fetch(
            f"""
            SELECT * FROM anomaly_events
            WHERE {where_clause}
            ORDER BY time DESC
            LIMIT ${idx} OFFSET ${idx + 1}
            """,
            *params, limit, offset,
        )

    return list(rows), total


async def acknowledge(pool: asyncpg.Pool, anomaly_id: str) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE anomaly_events
            SET acknowledged = TRUE, acknowledged_at = NOW()
            WHERE id = $1::uuid AND acknowledged = FALSE
            """,
            anomaly_id,
        )
    return result == "UPDATE 1"
