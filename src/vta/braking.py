"""Braking-event detection: consecutive seconds of strong deceleration become one event.

Hard braking is read directly off the longitudinal acceleration signal (``accel_ms2``,
m/s^2, already computed by the cleaning pipeline) -- no model is needed, just a threshold
and the same event-merging logic used for anomalies (consecutive flagged seconds are
grouped into one event, isolated blips are dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HARD_BRAKE_THRESHOLD_MS2

SEVERITY_BANDS_MS2 = [-7.0, -5.0]  # ascending; more negative accel_ms2 = harder braking
SEVERITY_LABELS = ["severe", "moderate", "light"]
EVENT_COLUMNS = [
    "event_id",
    "test_id",
    "start_t_s",
    "end_t_s",
    "duration_s",
    "n_points",
    "min_accel_ms2",
    "speed_before_kmh",
    "speed_after_kmh",
    "speed_drop_kmh",
    "severity",
    "start_time",
]


def extract_braking_events(
    df: pd.DataFrame,
    threshold: float = HARD_BRAKE_THRESHOLD_MS2,
    min_points: int = 1,
    max_gap_s: int = 2,
) -> pd.DataFrame:
    """Group consecutive seconds with ``accel_ms2 < threshold`` into braking events."""
    valid = df[df["is_valid"]] if "is_valid" in df.columns else df
    flagged = valid[valid["accel_ms2"] < threshold].sort_values(["test_id", "t_s"]).copy()
    if flagged.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    gap = flagged.groupby("test_id")["t_s"].diff()
    flagged["_ev"] = ((gap.isna()) | (gap > max_gap_s)).cumsum()
    ev = flagged.groupby("_ev").agg(
        test_id=("test_id", "first"),
        start_t_s=("t_s", "min"),
        end_t_s=("t_s", "max"),
        n_points=("t_s", "size"),
        min_accel_ms2=("accel_ms2", "min"),
        start_time=("timestamp", "min"),
    )
    ev = ev[ev["n_points"] >= min_points].copy()
    if ev.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev["duration_s"] = (ev["end_t_s"] - ev["start_t_s"] + 1).astype(int)
    ev = ev.reset_index(drop=True)

    speed = df[["test_id", "t_s", "speed_kmh"]]
    ev["speed_before_kmh"] = ev[["test_id", "start_t_s"]].rename(columns={"start_t_s": "t_s"}).merge(
        speed, on=["test_id", "t_s"], how="left"
    )["speed_kmh"].to_numpy()
    ev["speed_after_kmh"] = ev[["test_id", "end_t_s"]].rename(columns={"end_t_s": "t_s"}).merge(
        speed, on=["test_id", "t_s"], how="left"
    )["speed_kmh"].to_numpy()
    ev["speed_drop_kmh"] = ev["speed_before_kmh"] - ev["speed_after_kmh"]

    bins = [-np.inf, *SEVERITY_BANDS_MS2, threshold]
    ev["severity"] = pd.cut(ev["min_accel_ms2"], bins=bins, labels=SEVERITY_LABELS).astype(str)
    ev = ev.sort_values(["test_id", "start_t_s"]).reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))
    ev["start_time"] = ev["start_time"].astype(str)
    return ev[EVENT_COLUMNS]
