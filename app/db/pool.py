from __future__ import annotations

import logging
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def create_pool(dsn: str) -> asyncpg.Pool:
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=5,
        max_size=20,
        command_timeout=30,
        server_settings={"application_name": "sensorwatch"},
    )
    return pool


async def run_migrations(pool: asyncpg.Pool) -> None:
    migration_file = _MIGRATIONS_DIR / "001_initial_schema.sql"
    sql = migration_file.read_text()

    async with pool.acquire() as conn:
        # Split on semicolons to run statements individually;
        # TimescaleDB functions don't work well inside a single transaction.
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for statement in statements:
            try:
                await conn.execute(statement)
            except asyncpg.PostgresError as exc:
                # Already-exists errors are safe to ignore (idempotent migrations)
                if "already exists" in str(exc).lower():
                    logger.debug("Migration statement skipped (already applied)", extra={"stmt_preview": statement[:60]})
                else:
                    raise

    logger.info("Database migrations applied")
