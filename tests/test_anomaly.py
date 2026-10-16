import numpy as np
import pandas as pd
import pytest

from vta.anomaly import (
    AnomalyConfig,
    ContextResiduals,
    VehicleAnomalyDetector,
    evaluate_against_labels,
    extract_events,
)
from vta.config import DETECTOR_FEATURES
from vta.pipeline_pandas import run_pandas


def test_extract_events_merges_gaps_and_filters_short_runs():
    flagged_t = [10, 11, 12, 14, 30, 50, 51, 52]  # 10-14 (gap<=3) -> one event; 30 alone -> dropped
    t = np.arange(100.0)
    df = pd.DataFrame(
        {
            "test_id": "T1",
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="s"),
            "t_s": t,
            "is_anomaly": np.isin(t, flagged_t),
            "anomaly_score": 0.6,
            "anomaly_margin": 0.05,
            "top_sensor": "rpm",
        }
    )
    ev = extract_events(df, min_points=3, max_gap_s=3)
    assert len(ev) == 2
    assert ev.loc[0, ["start_t_s", "end_t_s", "n_points", "duration_s"]].tolist() == [10, 14, 4, 5]
    assert ev.loc[1, "start_t_s"] == 50
    assert (ev["severity"] == "medium").all()


def test_extract_events_empty():
    df = pd.DataFrame({"test_id": ["A"], "t_s": [0.0], "is_anomaly": [False]})
    assert extract_events(df).empty


def _synthetic(n=4000, n_out=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, len(DETECTOR_FEATURES))), columns=DETECTOR_FEATURES)
    out_idx = rng.choice(n, n_out, replace=False)
    X.loc[out_idx, DETECTOR_FEATURES[:3]] += rng.choice([-1, 1], (n_out, 3)) * 7  # clear multivariate outliers
    X["is_valid"] = True
    X["test_type"] = "urban"
    X["test_id"] = "T1"
    X["t_s"] = np.arange(n, dtype=float)
    X["timestamp"] = pd.date_range("2026-01-01", periods=n, freq="s")
    truth = np.zeros(n, dtype=bool)
    truth[out_idx] = True
    return X, truth


def test_isolation_forest_separates_clear_outliers():
    X, truth = _synthetic()
    scored = VehicleAnomalyDetector(AnomalyConfig(contamination=0.02, n_estimators=100)).fit_score(X)
    flagged = scored["is_anomaly"].to_numpy()
    assert flagged[truth].mean() > 0.9  # recall on obvious outliers
    assert flagged[~truth].mean() < 0.02  # few false alarms
    assert set(scored.loc[flagged & truth, "top_sensor"].dropna()) <= {s.removeprefix("res_") for s in DETECTOR_FEATURES[:3]}
    assert (scored.loc[~scored["is_valid"], "anomaly_score"].isna()).all()


def test_invalid_rows_are_not_scored():
    X, _ = _synthetic(n=1000, n_out=10)
    X.loc[:49, "is_valid"] = False
    scored = VehicleAnomalyDetector(AnomalyConfig(n_estimators=50)).fit_score(X)
    assert scored.loc[:49, "anomaly_score"].isna().all()
    assert not scored.loc[:49, "is_anomaly"].any()


def test_context_residuals_remove_operating_point_effect():
    rng = np.random.default_rng(1)
    n = 3000
    rpm = rng.uniform(800, 4000, n)
    df = pd.DataFrame(
        {
            "is_valid": True,
            "rpm": rpm,
            "engine_temp_c": 90.0,
            "oil_pressure_bar": 0.8 + 0.00085 * rpm + rng.normal(0, 0.03, n),
        }
    )
    ctx = ContextResiduals(context={"oil_pressure_bar": ["rpm", "engine_temp_c"]}).fit(df)
    res = ctx.transform(df)["res_oil_pressure_bar"]
    assert res.std() < 0.2 * df["oil_pressure_bar"].std()


def test_evaluate_against_labels_perfect_and_empty():
    labels = pd.DataFrame(
        [{"test_id": "T1", "anomaly_type": "fuel_spike", "category": "behavior", "sensor": "fuel_rate_lph", "start_s": 10, "end_s": 20}]
    )
    t = np.arange(40.0)
    scored = pd.DataFrame({"test_id": "T1", "t_s": t, "is_valid": True, "is_anomaly": (t >= 10) & (t <= 20)})
    events = pd.DataFrame([{"test_id": "T1", "start_t_s": 10.0, "end_t_s": 20.0}])
    m, by_type = evaluate_against_labels(scored, events, labels)
    assert m["event_recall"] == 1.0 and m["event_precision"] == 1.0
    assert m["point_recall"] == 1.0 and m["point_precision"] == 1.0
    assert by_type.loc[0, "recall"] == 1.0
    assert evaluate_against_labels(scored, events, pd.DataFrame())[0] == {}


def test_detector_save_and_load(tmp_path):
    X, _ = _synthetic(n=1500, n_out=20)
    det = VehicleAnomalyDetector(AnomalyConfig(n_estimators=50)).fit(X)
    det.save(tmp_path / "d.joblib")
    loaded = VehicleAnomalyDetector.load(tmp_path / "d.joblib")
    pd.testing.assert_series_equal(det.score(X)["anomaly_score"], loaded.score(X)["anomaly_score"])


def test_quality_gate_on_generated_fleet(dataset_dir):
    """Regression gate: the end-to-end detector must keep finding most injected faults."""
    res = run_pandas(dataset_dir)
    feats = res["features"].merge(res["tests"][["test_id", "test_type"]], on="test_id")
    scored = VehicleAnomalyDetector(AnomalyConfig(contamination=0.02)).fit_score(feats)
    events = extract_events(scored)
    metrics, _ = evaluate_against_labels(scored, events, res["anomaly_labels"])
    assert metrics["event_recall"] >= 0.6, metrics
    assert metrics["event_precision"] >= 0.6, metrics
    # the context-residual design must beat raw sensors fed directly to the forest
    raw = VehicleAnomalyDetector(AnomalyConfig(contamination=0.02, features=["speed_kmh", "rpm", "throttle_pct", "engine_temp_c",
                                                                           "oil_pressure_bar", "fuel_rate_lph", "battery_v", "vibration_g"]))
    raw_scored = raw.fit_score(feats)
    raw_metrics, _ = evaluate_against_labels(raw_scored, extract_events(raw_scored), res["anomaly_labels"])
    assert metrics["event_recall"] > raw_metrics["event_recall"] + 0.15, (metrics, raw_metrics)
