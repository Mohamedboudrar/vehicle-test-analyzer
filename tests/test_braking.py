import numpy as np
import pandas as pd

from vta.braking import extract_braking_events


def _df(accel, speed, test_id="T1"):
    n = len(accel)
    return pd.DataFrame(
        {
            "test_id": test_id,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="s"),
            "t_s": np.arange(n, dtype=float),
            "is_valid": True,
            "accel_ms2": accel,
            "speed_kmh": speed,
        }
    )


def test_extract_braking_events_merges_consecutive_seconds_and_scores_severity():
    accel = np.zeros(30)
    accel[10:13] = -4.0  # light braking, 3 consecutive seconds
    accel[14] = -4.0  # gap of 1 s (<= 2) -> merged into the same event
    accel[20] = -8.0  # severe, isolated -> its own event
    speed = 80.0 - np.cumsum(np.clip(-accel, 0, None))
    df = _df(accel, speed)

    ev = extract_braking_events(df, min_points=1, max_gap_s=2)
    assert len(ev) == 2
    first, second = ev.iloc[0], ev.iloc[1]
    assert (first.start_t_s, first.end_t_s, first.n_points) == (10, 14, 4)
    assert first.severity == "light"
    assert first.speed_drop_kmh > 0
    assert second.start_t_s == 20 and second.severity == "severe"


def test_extract_braking_events_empty_when_no_hard_braking():
    df = _df(np.zeros(20), np.full(20, 50.0))
    assert extract_braking_events(df).empty


def test_extract_braking_events_drops_runs_below_min_points():
    accel = np.zeros(20)
    accel[5] = -4.0
    df = _df(accel, np.full(20, 50.0))
    assert extract_braking_events(df, min_points=2).empty


def test_extract_braking_events_ignores_invalid_rows():
    accel = np.zeros(20)
    accel[5] = -4.0
    df = _df(accel, np.full(20, 50.0))
    df.loc[5, "is_valid"] = False
    assert extract_braking_events(df).empty
