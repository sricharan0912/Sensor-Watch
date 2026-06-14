-- ── TimescaleDB extension ────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── sensor_readings hypertable ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_readings (
    time            TIMESTAMPTZ     NOT NULL,
    engine_id       TEXT            NOT NULL,
    cycle           INTEGER         NOT NULL,

    -- 14 informative CMAPSS sensor channels
    s2              REAL,
    s3              REAL,
    s4              REAL,
    s7              REAL,
    s8              REAL,
    s9              REAL,
    s11             REAL,
    s12             REAL,
    s13             REAL,
    s14             REAL,
    s15             REAL,
    s17             REAL,
    s20             REAL,
    s21             REAL,

    -- Operational settings (affect sensor baselines)
    op_setting_1    REAL,
    op_setting_2    REAL,
    op_setting_3    REAL
);

SELECT create_hypertable(
    'sensor_readings',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_engine_time
    ON sensor_readings (engine_id, time DESC);

SELECT add_retention_policy(
    'sensor_readings',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Per-minute continuous aggregate for Grafana dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time)   AS bucket,
    engine_id,
    AVG(s4)                         AS s4_avg,
    AVG(s11)                        AS s11_avg,
    AVG(s12)                        AS s12_avg,
    AVG(s14)                        AS s14_avg,
    COUNT(*)                        AS sample_count
FROM sensor_readings
GROUP BY bucket, engine_id
WITH NO DATA;

-- ── anomaly_events hypertable ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_events (
    id                   UUID            NOT NULL DEFAULT gen_random_uuid(),
    time                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    engine_id            TEXT            NOT NULL,
    reconstruction_error REAL            NOT NULL,
    threshold_warning    REAL            NOT NULL,
    threshold_critical   REAL            NOT NULL,
    severity             TEXT            NOT NULL CHECK (severity IN ('NORMAL', 'WARNING', 'CRITICAL')),
    window_start_time    TIMESTAMPTZ,
    window_end_time      TIMESTAMPTZ,
    model_version        TEXT            NOT NULL,
    acknowledged         BOOLEAN         NOT NULL DEFAULT FALSE,
    acknowledged_at      TIMESTAMPTZ,
    PRIMARY KEY (time, id)
);

SELECT create_hypertable(
    'anomaly_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_engine_time
    ON anomaly_events (engine_id, time DESC);

-- Partial index — only non-NORMAL events are queried by severity
CREATE INDEX IF NOT EXISTS idx_anomaly_events_severity
    ON anomaly_events (severity, time DESC)
    WHERE severity != 'NORMAL';

SELECT add_retention_policy(
    'anomaly_events',
    INTERVAL '365 days',
    if_not_exists => TRUE
);
