import numpy as np
import pandas as pd
import pytest

from vta.config import SENSORS
from vta.pipeline_pandas import clean_and_featurize, compute_kpis, run_pandas


def _raw(n=60, speed=lambda i: 36.0):
    ts = pd.date_range("2026-01-01 10:00:00", periods=n, freq="s").strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        {
            "test_id": "T1",
            "timestamp": ts,
            "speed_kmh": [speed(i) for i in range(n)],
            "rpm": 2000.0,
            "throttle_pct": 20.0,
            "engine_temp_c": 90.0,
            "oil_pressure_bar": 3.0,
            "fuel_rate_lph": 3.6,
            "battery_v": 13.8,
            "vibration_g": 0.2,
        }
    )
    return df


def test_cleaning_handles_duplicates_sentinels_and_gaps():
    df = _raw(60)
    df.loc[10, "rpm"] = -999.0  # sentinel -> out of range -> filled from previous second
    df.loc[15:17, "battery_v"] = np.nan  # short gap (3) -> forward-filled
    df.loc[30:37, "oil_pressure_bar"] = np.nan  # long gap (8): 5 filled, 3 stay missing
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True).sample(frac=1, random_state=1)  # duplicate + shuffle

    feat, q = clean_and_featurize(df)
    assert len(feat) == 60
    assert q.loc[0, "n_duplicates_removed"] == 1
    assert q.loc[0, "n_cells_out_of_range"] == 1
    assert feat["t_s"].is_monotonic_increasing
    assert feat.loc[10, "rpm"] == 2000.0
    assert feat.loc[15:17, "battery_v"].eq(13.8).all()
    assert int((~feat["is_valid"]).sum()) == 3
    assert q.loc[0, "n_rows_valid"] == 57


def test_engineered_features():
    feat, _ = clean_and_featurize(_raw(60, speed=lambda i: 3.6 * i))  # +1 m/s every second
    assert feat.loc[0, "accel_ms2"] == 0.0
    assert np.allclose(feat.loc[1:, "accel_ms2"], 1.0)
    assert np.allclose(feat["engine_temp_slope"], 0.0)
    assert np.allclose(feat["rpm_roll_std"], 0.0)


def test_kpis_are_physically_consistent():
    feat, _ = clean_and_featurize(_raw(600, speed=lambda i: 36.0))  # 36 km/h for 10 min, 3.6 L/h
    k = compute_kpis(feat).iloc[0]
    assert k["distance_km"] == pytest.approx(6.0)
    assert k["avg_fuel_l100km"] == pytest.approx(10.0)  # 0.6 L over 6 km
    assert k["idle_pct"] == 0.0
    assert k["hard_brake_count"] == 0


def test_full_pipeline_on_generated_data(dataset_dir):
    res = run_pandas(dataset_dir)
    f = res["features"]
    assert f["is_valid"].mean() > 0.999
    assert not f.duplicated(["test_id", "t_s"]).any()
    assert f[SENSORS].dropna().shape[0] == int(f["is_valid"].sum())
    assert len(res["kpis"]) == 48
    assert (res["kpis"]["avg_fuel_l100km"].between(3, 20)).all()
