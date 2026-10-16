"""Anomaly detection: context residuals + Isolation Forest.

Design choices
--------------
* **Context-aware features.** Raw sensor values are poor anomaly signals on their own:
  a fuel rate of 8 L/h is normal at full throttle and abnormal at idle. A small gradient
  boosting model predicts each sensor from the operating conditions and the *residual*
  (measured - expected) is given to the Isolation Forest. On the demo data, at the same
  contamination (2 %), event recall goes from 51 % to 71 % and event precision from 39 %
  to 81 % compared with feeding the raw sensors directly.
* **One Isolation Forest per ``test_type``**: cold-start warm-up would be an anomaly in a
  highway test. Trees are scale-invariant, so no feature scaling is needed.
* **Events, not points.** Consecutive flagged seconds are merged into events, which is
  what an engineer reviews.
* **Attribution.** Each flagged point gets a ``top_sensor``: the feature deviating most
  from the group's robust baseline (median / MAD), as a cheap root-cause hint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

from .config import CONTEXT_MODELS, DETECTOR_FEATURES, FEATURE_TO_SENSOR, IF_DEFAULTS

SEVERITY_BANDS = [0.0, 0.04, 0.09]  # margin above the decision threshold -> low / medium / high
SEVERITY_LABELS = ["low", "medium", "high"]
EVENT_COLUMNS = [
    "event_id",
    "test_id",
    "start_t_s",
    "end_t_s",
    "duration_s",
    "n_points",
    "max_score",
    "mean_score",
    "severity",
    "top_sensor",
    "start_time",
]


# --------------------------------------------------------------------------- context
class ContextResiduals:
    """Predict each sensor from its operating context and expose the residual as ``res_<sensor>``."""

    def __init__(self, context: dict[str, list[str]] | None = None, random_state: int = 0):
        self.context = context or CONTEXT_MODELS
        self.random_state = random_state
        self.models_: dict[str, HistGradientBoostingRegressor] = {}

    def fit(self, df: pd.DataFrame) -> "ContextResiduals":
        valid = df[df["is_valid"]]
        for target, inputs in self.context.items():
            self.models_[target] = HistGradientBoostingRegressor(
                max_depth=5, max_iter=150, min_samples_leaf=200, random_state=self.random_state
            ).fit(valid[inputs], valid[target])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ok = out["is_valid"].to_numpy()
        for target, inputs in self.context.items():
            res = np.full(len(out), np.nan)
            if ok.any():
                res[ok] = out.loc[ok, target].to_numpy() - self.models_[target].predict(out.loc[ok, inputs])
            out[f"res_{target}"] = res
        return out


# --------------------------------------------------------------------------- detector
@dataclass
class AnomalyConfig:
    contamination: float = IF_DEFAULTS["contamination"]
    n_estimators: int = IF_DEFAULTS["n_estimators"]
    max_samples: int = IF_DEFAULTS["max_samples"]
    random_state: int = IF_DEFAULTS["random_state"]
    group_col: str = "test_type"
    features: list[str] = field(default_factory=lambda: list(DETECTOR_FEATURES))
    min_event_points: int = 3
    max_gap_s: int = 3


class VehicleAnomalyDetector:
    """Context residuals + group-wise Isolation Forest, with robust attribution."""

    def __init__(self, config: AnomalyConfig | None = None):
        self.cfg = config or AnomalyConfig()
        self.context_: ContextResiduals | None = None
        self.models_: dict[str, IsolationForest] = {}
        self.baselines_: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ fit
    def _fit_one(self, X: np.ndarray) -> tuple[IsolationForest, tuple[np.ndarray, np.ndarray]]:
        model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            contamination=self.cfg.contamination,
            max_samples=min(self.cfg.max_samples, len(X)),
            random_state=self.cfg.random_state,
            n_jobs=-1,
        ).fit(X)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        scale = np.maximum(mad, 0.1 * X.std(axis=0))  # avoid exploding z-scores on near-constant features
        scale[scale < 1e-9] = 1.0
        return model, (med, scale)

    def _with_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in self.cfg.features):
            return df  # residuals already computed (e.g. cached by the dashboard)
        return self.context_.transform(df)

    def fit(self, df: pd.DataFrame) -> "VehicleAnomalyDetector":
        if not all(c in df.columns for c in self.cfg.features):
            self.context_ = ContextResiduals().fit(df)
        df = self._with_residuals(df)
        valid = df[df["is_valid"]]
        self.models_, self.baselines_ = {}, {}
        for key, g in valid.groupby(self.cfg.group_col):
            if len(g) >= 200:
                self.models_[str(key)], self.baselines_[str(key)] = self._fit_one(g[self.cfg.features].to_numpy())
        self.models_["__global__"], self.baselines_["__global__"] = self._fit_one(valid[self.cfg.features].to_numpy())
        return self

    # ------------------------------------------------------------------ score
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` (+ residual columns) with anomaly columns; invalid rows stay unscored."""
        out = self._with_residuals(df).copy()
        out["anomaly_score"] = np.nan
        out["anomaly_margin"] = np.nan
        out["is_anomaly"] = False
        out["severity"] = "none"
        out["top_sensor"] = None

        for key, idx in out[out["is_valid"]].groupby(self.cfg.group_col).groups.items():
            k = str(key) if str(key) in self.models_ else "__global__"
            model, (med, scale) = self.models_[k], self.baselines_[k]
            X = out.loc[idx, self.cfg.features].to_numpy()
            raw = -model.score_samples(X)  # higher = more anomalous
            margin = raw - (-model.offset_)  # > 0 means flagged by the model
            flagged = margin > 0
            out.loc[idx, "anomaly_score"] = raw
            out.loc[idx, "anomaly_margin"] = margin
            out.loc[idx, "is_anomaly"] = flagged

            sev = np.full(len(idx), "none", dtype=object)
            if flagged.any():
                sev[flagged] = pd.cut(margin[flagged], bins=SEVERITY_BANDS + [np.inf], labels=SEVERITY_LABELS).astype(str)
            out.loc[idx, "severity"] = sev

            z = np.abs((X - med) / scale)
            names = np.array(self.cfg.features, dtype=object)[z.argmax(axis=1)]
            top = np.array([_to_sensor(n) for n in names], dtype=object)
            top[~flagged] = None
            out.loc[idx, "top_sensor"] = top
        return out

    def fit_score(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).score(df)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> "VehicleAnomalyDetector":
        return joblib.load(path)


def _to_sensor(feature: str) -> str:
    if feature.startswith("res_"):
        return feature[4:]
    return FEATURE_TO_SENSOR.get(feature, feature)


# --------------------------------------------------------------------------- events
def extract_events(scored: pd.DataFrame, min_points: int = 3, max_gap_s: int = 3) -> pd.DataFrame:
    """Group consecutive flagged seconds into events and summarise each one."""
    flagged = scored[scored["is_anomaly"]].sort_values(["test_id", "t_s"]).copy()
    if flagged.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    gap = flagged.groupby("test_id")["t_s"].diff()
    flagged["_ev"] = ((gap.isna()) | (gap > max_gap_s)).cumsum()
    ev = flagged.groupby("_ev").agg(
        test_id=("test_id", "first"),
        start_t_s=("t_s", "min"),
        end_t_s=("t_s", "max"),
        n_points=("t_s", "size"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
        peak_margin=("anomaly_margin", "max"),
        top_sensor=("top_sensor", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        start_time=("timestamp", "min"),
    )
    ev = ev[ev["n_points"] >= min_points].copy()
    if ev.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev["duration_s"] = (ev["end_t_s"] - ev["start_t_s"] + 1).astype(int)
    ev["severity"] = pd.cut(ev["peak_margin"], bins=SEVERITY_BANDS + [np.inf], labels=SEVERITY_LABELS).astype(str)
    ev = ev.sort_values(["test_id", "start_t_s"]).reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))
    ev["start_time"] = ev["start_time"].astype(str)
    return ev[EVENT_COLUMNS]


# --------------------------------------------------------------------------- evaluation
def evaluate_against_labels(
    scored: pd.DataFrame, events: pd.DataFrame, labels: pd.DataFrame, tolerance_s: int = 2
) -> tuple[dict, pd.DataFrame]:
    """Compare detections with injected ground truth (behavioural anomalies only).

    * event recall: share of injected anomalies overlapped by at least one detected event
    * event precision: share of detected events overlapping an injected anomaly
    * point precision / recall: second-by-second agreement
    """
    if labels is None or len(labels) == 0:
        return {}, pd.DataFrame()
    lab = labels[labels["category"] == "behavior"].copy()
    if lab.empty:
        return {}, pd.DataFrame()

    ev = events.copy()
    ev_by_test = {k: g for k, g in ev.groupby("test_id")}
    empty_ev = ev.iloc[0:0]
    lab["detected"] = [
        bool(
            (
                (te["start_t_s"] <= r["end_s"] + tolerance_s) & (te["end_t_s"] >= r["start_s"] - tolerance_s)
            ).any()
        )
        for r in lab.to_dict("records")
        for te in [ev_by_test.get(r["test_id"], empty_ev)]
    ]

    lab_by_test = {k: g for k, g in lab.groupby("test_id")}
    empty_lab = lab.iloc[0:0]
    ev["true_positive"] = [
        bool(((tl["start_s"] - tolerance_s <= r["end_t_s"]) & (tl["end_s"] + tolerance_s >= r["start_t_s"])).any())
        for r in ev.to_dict("records")
        for tl in [lab_by_test.get(r["test_id"], empty_lab)]
    ]

    truth = pd.Series(False, index=scored.index)
    for r in lab.to_dict("records"):
        truth |= (scored["test_id"] == r["test_id"]) & scored["t_s"].between(r["start_s"], r["end_s"])
    pred = scored["is_anomaly"] & scored["is_valid"]
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())

    by_type = (
        lab.groupby("anomaly_type")["detected"].agg(detected="sum", total="size").assign(
            recall=lambda d: d["detected"] / d["total"]
        )
    ).reset_index()

    metrics = {
        "n_injected_behavior": int(len(lab)),
        "n_detected_events": int(len(ev)),
        "event_recall": float(lab["detected"].mean()),
        "event_precision": float(ev["true_positive"].mean()) if len(ev) else float("nan"),
        "point_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "point_precision": tp / (tp + fp) if tp + fp else float("nan"),
    }
    return metrics, by_type
