"""Central configuration: paths, sensor definitions and model defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VTA_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = Path(os.environ.get("VTA_DB_PATH", DATA_DIR / "vehicle_tests.db"))
SQL_DIR = ROOT / "sql"

# Physically plausible range per sensor. Values outside are treated as sensor errors
# (e.g. the -999 sentinel) and replaced by NULL during cleaning.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "speed_kmh": (0.0, 250.0),
    "rpm": (0.0, 7500.0),
    "throttle_pct": (0.0, 100.0),
    "engine_temp_c": (-30.0, 130.0),
    "oil_pressure_bar": (0.0, 8.0),
    "fuel_rate_lph": (0.0, 60.0),
    "battery_v": (8.0, 16.0),
    "vibration_g": (0.0, 5.0),
}
SENSORS: list[str] = list(SENSOR_RANGES)

# Engineered features (identical definitions in the Pandas and PySpark pipelines).
ROLL_WINDOW = 10  # seconds (1 Hz data)
FILL_LIMIT = 5  # max consecutive missing samples repaired by forward fill
ENGINEERED: list[str] = ["accel_ms2", "engine_temp_slope", "rpm_roll_std", "vibration_roll_mean"]
MODEL_FEATURES: list[str] = SENSORS + ENGINEERED

# Engineered feature -> physical sensor it describes (used for root-cause attribution).
# Context residuals ("res_<sensor>") map back to <sensor> by stripping the prefix.
FEATURE_TO_SENSOR: dict[str, str] = {
    "accel_ms2": "speed_kmh",
    "engine_temp_slope": "engine_temp_c",
    "rpm_roll_std": "rpm",
    "vibration_roll_mean": "vibration_g",
}

# Context-aware modelling: each sensor is predicted from the operating conditions, and the
# residual (measured - expected) is what the Isolation Forest looks at. A fuel rate of 8 L/h
# is normal at full throttle but abnormal at idle.
CONTEXT_MODELS: dict[str, list[str]] = {
    "oil_pressure_bar": ["rpm", "engine_temp_c"],
    "fuel_rate_lph": ["rpm", "throttle_pct", "speed_kmh"],
    "vibration_g": ["rpm", "speed_kmh"],
    "battery_v": ["rpm", "throttle_pct"],
    "rpm": ["speed_kmh", "throttle_pct", "accel_ms2"],
}
RESIDUAL_FEATURES: list[str] = [f"res_{s}" for s in CONTEXT_MODELS]
# Features fed to the Isolation Forest: context residuals + thermal / dynamic signals.
DETECTOR_FEATURES: list[str] = RESIDUAL_FEATURES + ["engine_temp_c", "engine_temp_slope", "accel_ms2"]

KPI_COLUMNS: list[str] = [
    "n_rows",
    "distance_km",
    "avg_speed_kmh",
    "max_speed_kmh",
    "avg_rpm",
    "max_rpm",
    "max_engine_temp_c",
    "avg_fuel_l100km",
    "idle_pct",
    "max_accel_ms2",
    "hard_brake_count",
    "avg_vibration_g",
    "min_oil_pressure_bar",
    "min_battery_v",
]
HARD_BRAKE_THRESHOLD_MS2 = -3.0

# Isolation Forest defaults
IF_DEFAULTS = {
    "contamination": 0.02,
    "n_estimators": 200,
    "max_samples": 1024,
    "random_state": 42,
}
