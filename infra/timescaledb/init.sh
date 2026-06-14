#!/bin/bash
# This script runs inside the TimescaleDB container on first start
# (placed in /docker-entrypoint-initdb.d/).
# The migration SQL is re-applied idempotently by the app's run_migrations()
# at startup, so this script is a no-op beyond the container's own init.

echo "TimescaleDB container initialised for database: $POSTGRES_DB"
