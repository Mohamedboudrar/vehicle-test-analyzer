"""Pandas pipeline: cleaning, feature engineering, KPIs and data-quality report.

The PySpark pipeline (``pipeline_spark``) implements exactly the same logic so that
the two engines can be cross-checked (see ``tests/test_spark_equivalence.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FILL_LIMIT,
    HARD_BRAKE_THRESHOLD_MS2,
    RAW_DIR,
    ROLL_WINDOW,
    SENSOR_RANGES,
    SENSORS,
)


def load_raw(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    out = {
        "measurements": pd.read_csv(raw_dir / "measurements.csv"),
        "tests": pd.read_csv(raw_dir / "tests.csv"),
        "vehicles": pd.read_csv(raw_dir / "vehicles.csv"),
    }
    labels = raw_dir / "anomaly_labels.csv"
    out["anomaly_labels"] = pd.read_csv(labels) if labels.exists() else pd.DataFrame()
    return out


def clean_and_featurize(meas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw measurements and compute features.

    Steps: parse timestamps -> drop duplicates -> sort -> null out-of-range values
    (sensor errors such as -999) -> forward-fill short gaps -> engineered features.

    Returns ``(features, quality)`` where ``quality`` has one row per test.
    """
    df = meas.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    n_raw = df.groupby("test_id").size()
    missing_raw = df[SENSORS].isna().sum(axis=1).groupby(df["test_id"]).sum()

    df = df.drop_duplicates(["test_id", "timestamp"]).sort_values(["test_id", "timestamp"]).reset_index(drop=True)
    n_dups = n_raw - df.groupby("test_id").size()

    # Out-of-range values -> NaN
    oor = pd.Series(0, index=df.index)
    for s, (lo, hi) in SENSOR_RANGES.items():
        bad = (df[s] < lo) | (df[s] > hi)
        oor = oor + bad.astype(int)
        df.loc[bad, s] = np.nan
    n_oor = oor.groupby(df["test_id"]).sum()

    g = df.groupby("test_id", sort=False)
    df[SENSORS] = g[SENSORS].ffill(limit=FILL_LIMIT)

    # Engineered features
    df["t_s"] = (df["timestamp"] - g["timestamp"].transform("min")).dt.total_seconds()
    dt = g["t_s"].diff()
    df["accel_ms2"] = ((g["speed_kmh"].diff() / 3.6) / dt).fillna(0.0)
    lag_t = g["t_s"].shift(ROLL_WINDOW)
    df["engine_temp_slope"] = ((df["engine_temp_c"] - g["engine_temp_c"].shift(ROLL_WINDOW)) / (df["t_s"] - lag_t)).fillna(0.0)
    df["rpm_roll_std"] = (
        g["rpm"].rolling(ROLL_WINDOW, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
    )
    df["vibration_roll_mean"] = (
        g["vibration_g"].rolling(ROLL_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["fuel_l100km"] = np.where(df["speed_kmh"] >= 5, df["fuel_rate_lph"] / df["speed_kmh"] * 100.0, np.nan)
    df["is_valid"] = df[SENSORS].notna().all(axis=1)

    n_missing_after = df[SENSORS].isna().sum(axis=1).groupby(df["test_id"]).sum()
    n_valid = df.groupby("test_id")["is_valid"].sum()
    quality = pd.DataFrame(
        {
            "n_rows_raw": n_raw,
            "n_duplicates_removed": n_dups,
            "n_cells_missing_raw": missing_raw,
            "n_cells_out_of_range": n_oor,
            "n_cells_missing_after_fill": n_missing_after,
            "n_rows_valid": n_valid,
        }
    ).reset_index()
    quality["valid_pct"] = (quality["n_rows_valid"] / (quality["n_rows_raw"] - quality["n_duplicates_removed"]) * 100).round(2)
    return df, quality


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-test KPIs computed on valid rows (1 Hz sampling)."""
    d = df[df["is_valid"]]
    g = d.groupby("test_id")
    total_fuel_l = g["fuel_rate_lph"].sum() / 3600.0
    distance = g["speed_kmh"].sum() / 3600.0
    kpi = pd.DataFrame(
        {
            "n_rows": g.size(),
            "distance_km": distance,
            "avg_speed_kmh": g["speed_kmh"].mean(),
            "max_speed_kmh": g["speed_kmh"].max(),
            "avg_rpm": g["rpm"].mean(),
            "max_rpm": g["rpm"].max(),
            "max_engine_temp_c": g["engine_temp_c"].max(),
            "avg_fuel_l100km": np.where(distance > 0, total_fuel_l / distance * 100.0, np.nan),
            "idle_pct": g["speed_kmh"].apply(lambda s: (s < 1).mean() * 100.0),
            "max_accel_ms2": g["accel_ms2"].max(),
            "hard_brake_count": g["accel_ms2"].apply(lambda s: int((s < HARD_BRAKE_THRESHOLD_MS2).sum())),
            "avg_vibration_g": g["vibration_g"].mean(),
            "min_oil_pressure_bar": g["oil_pressure_bar"].min(),
            "min_battery_v": g["battery_v"].min(),
        }
    )
    return kpi.reset_index()


def run_pandas(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    raw = load_raw(raw_dir)
    features, quality = clean_and_featurize(raw["measurements"])
    kpis = compute_kpis(features)
    return {**raw, "features": features, "quality": quality, "kpis": kpis}
