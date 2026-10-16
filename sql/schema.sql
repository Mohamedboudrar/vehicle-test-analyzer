-- Vehicle Test Data Analyzer: SQLite schema (analytical star-like model)
-- vehicles 1--n tests 1--n measurements ; tests 1--n anomaly_events ; tests 1--1 test_kpis

DROP VIEW  IF EXISTS v_test_summary;
DROP TABLE IF EXISTS injected_anomalies;
DROP TABLE IF EXISTS braking_events;
DROP TABLE IF EXISTS anomaly_events;
DROP TABLE IF EXISTS data_quality;
DROP TABLE IF EXISTS test_kpis;
DROP TABLE IF EXISTS measurements;
DROP TABLE IF EXISTS tests;
DROP TABLE IF EXISTS vehicles;

CREATE TABLE vehicles (
    vehicle_id  TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    fuel_type   TEXT NOT NULL CHECK (fuel_type IN ('petrol', 'diesel')),
    engine_cc   INTEGER,
    mass_kg     INTEGER,
    model_year  INTEGER
);

CREATE TABLE tests (
    test_id         TEXT PRIMARY KEY,
    vehicle_id      TEXT NOT NULL REFERENCES vehicles (vehicle_id),
    test_type       TEXT NOT NULL,
    start_time      TEXT NOT NULL,
    duration_s      INTEGER,
    ambient_temp_c  REAL
);

CREATE TABLE measurements (
    test_id              TEXT    NOT NULL REFERENCES tests (test_id),
    timestamp            TEXT    NOT NULL,
    t_s                  REAL    NOT NULL,
    speed_kmh            REAL,
    rpm                  REAL,
    throttle_pct         REAL,
    engine_temp_c        REAL,
    oil_pressure_bar     REAL,
    fuel_rate_lph        REAL,
    battery_v            REAL,
    vibration_g          REAL,
    accel_ms2            REAL,
    engine_temp_slope    REAL,
    rpm_roll_std         REAL,
    vibration_roll_mean  REAL,
    fuel_l100km          REAL,
    res_oil_pressure_bar REAL,
    res_fuel_rate_lph    REAL,
    res_vibration_g      REAL,
    res_battery_v        REAL,
    res_rpm              REAL,
    is_valid             INTEGER NOT NULL,
    anomaly_score        REAL,
    anomaly_margin       REAL,
    is_anomaly           INTEGER NOT NULL DEFAULT 0,
    severity             TEXT    NOT NULL DEFAULT 'none',
    top_sensor           TEXT,
    PRIMARY KEY (test_id, t_s)
);
CREATE INDEX idx_meas_anomaly ON measurements (is_anomaly, test_id);

CREATE TABLE test_kpis (
    test_id               TEXT PRIMARY KEY REFERENCES tests (test_id),
    n_rows                INTEGER,
    distance_km           REAL,
    avg_speed_kmh         REAL,
    max_speed_kmh         REAL,
    avg_rpm               REAL,
    max_rpm               REAL,
    max_engine_temp_c     REAL,
    avg_fuel_l100km       REAL,
    idle_pct              REAL,
    max_accel_ms2         REAL,
    hard_brake_count      INTEGER,
    avg_vibration_g       REAL,
    min_oil_pressure_bar  REAL,
    min_battery_v         REAL,
    n_anomaly_events      INTEGER NOT NULL DEFAULT 0,
    anomaly_points        INTEGER NOT NULL DEFAULT 0,
    anomaly_pct           REAL    NOT NULL DEFAULT 0,
    n_braking_events      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE data_quality (
    test_id                     TEXT PRIMARY KEY REFERENCES tests (test_id),
    n_rows_raw                  INTEGER,
    n_duplicates_removed        INTEGER,
    n_cells_missing_raw         INTEGER,
    n_cells_out_of_range        INTEGER,
    n_cells_missing_after_fill  INTEGER,
    n_rows_valid                INTEGER,
    valid_pct                   REAL
);

CREATE TABLE anomaly_events (
    event_id    INTEGER PRIMARY KEY,
    test_id     TEXT NOT NULL REFERENCES tests (test_id),
    start_t_s   REAL,
    end_t_s     REAL,
    duration_s  INTEGER,
    n_points    INTEGER,
    max_score   REAL,
    mean_score  REAL,
    severity    TEXT,
    top_sensor  TEXT,
    start_time  TEXT
);
CREATE INDEX idx_events_test ON anomaly_events (test_id);

-- Hard-braking events (accel_ms2 below HARD_BRAKE_THRESHOLD_MS2), merged from consecutive seconds.
CREATE TABLE braking_events (
    event_id          INTEGER PRIMARY KEY,
    test_id           TEXT NOT NULL REFERENCES tests (test_id),
    start_t_s         REAL,
    end_t_s           REAL,
    duration_s        INTEGER,
    n_points          INTEGER,
    min_accel_ms2     REAL,
    speed_before_kmh  REAL,
    speed_after_kmh   REAL,
    speed_drop_kmh    REAL,
    severity          TEXT,
    start_time        TEXT
);
CREATE INDEX idx_braking_test ON braking_events (test_id);

-- Ground truth (synthetic data only): used to evaluate the detector.
CREATE TABLE injected_anomalies (
    test_id       TEXT NOT NULL,
    anomaly_type  TEXT NOT NULL,
    category      TEXT NOT NULL,
    sensor        TEXT,
    start_s       INTEGER,
    end_s         INTEGER
);

CREATE VIEW v_test_summary AS
SELECT t.test_id, t.test_type, t.start_time, v.vehicle_id, v.model, v.fuel_type,
       k.distance_km, k.avg_speed_kmh, k.avg_fuel_l100km, k.max_engine_temp_c,
       k.avg_vibration_g, k.hard_brake_count, k.n_braking_events, k.n_anomaly_events, k.anomaly_pct
FROM tests t
JOIN vehicles v  ON v.vehicle_id = t.vehicle_id
JOIN test_kpis k ON k.test_id    = t.test_id;
