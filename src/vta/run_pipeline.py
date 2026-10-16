"""End-to-end CLI: generate (if needed) -> clean/features (Pandas or Spark) -> Isolation Forest -> SQLite.

    python -m vta.run_pipeline --engine pandas
    python -m vta.run_pipeline --engine spark --contamination 0.03 --regenerate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import anomaly as an
from . import braking as brk
from .config import DB_PATH, PROCESSED_DIR, RAW_DIR
from .data_generator import generate_dataset
from .storage import save_all


def _log(msg: str) -> None:
    print(f"[vta] {msg}", flush=True)


def run(
    engine: str = "pandas",
    raw_dir: Path | str = RAW_DIR,
    db_path: Path | str = DB_PATH,
    contamination: float = 0.02,
    n_estimators: int = 200,
    regenerate: bool = False,
    n_vehicles: int = 12,
    seed: int = 42,
    quiet: bool = False,
    processed_dir: Path | str = PROCESSED_DIR,
) -> dict:
    log = (lambda m: None) if quiet else _log
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    t0 = time.time()

    if regenerate or not (raw_dir / "measurements.csv").exists():
        log(f"generating synthetic dataset ({n_vehicles} vehicles, seed={seed})")
        generate_dataset(n_vehicles=n_vehicles, seed=seed, out_dir=raw_dir)

    log(f"cleaning + feature engineering with {engine}")
    if engine == "spark":
        from .pipeline_spark import run_spark

        res = run_spark(raw_dir, processed_dir)
    elif engine == "pandas":
        from .pipeline_pandas import run_pandas

        res = run_pandas(raw_dir)
    else:
        raise ValueError(f"unknown engine: {engine}")
    t_pipeline = time.time() - t0

    features = res["features"].merge(res["tests"][["test_id", "vehicle_id", "test_type"]], on="test_id", how="left")

    log("fitting context models + Isolation Forest")
    t1 = time.time()
    cfg = an.AnomalyConfig(contamination=contamination, n_estimators=n_estimators)
    detector = an.VehicleAnomalyDetector(cfg)
    scored = detector.fit_score(features)
    events = an.extract_events(scored, cfg.min_event_points, cfg.max_gap_s)
    t_model = time.time() - t1
    processed_dir.mkdir(parents=True, exist_ok=True)
    detector.save(processed_dir / "detector.joblib")

    # per-test anomaly KPIs
    pts = scored[scored["is_anomaly"]].groupby("test_id").size().rename("anomaly_points")
    evc = events.groupby("test_id").size().rename("n_anomaly_events") if len(events) else pd.Series(dtype=int, name="n_anomaly_events")
    kpis = res["kpis"].merge(pts, on="test_id", how="left").merge(evc, on="test_id", how="left")
    kpis[["anomaly_points", "n_anomaly_events"]] = kpis[["anomaly_points", "n_anomaly_events"]].fillna(0).astype(int)
    kpis["anomaly_pct"] = (kpis["anomaly_points"] / kpis["n_rows"] * 100).round(3)

    log("detecting hard-braking events")
    braking_events = brk.extract_braking_events(scored)
    bkc = (
        braking_events.groupby("test_id").size().rename("n_braking_events")
        if len(braking_events)
        else pd.Series(dtype=int, name="n_braking_events")
    )
    kpis = kpis.merge(bkc, on="test_id", how="left")
    kpis["n_braking_events"] = kpis["n_braking_events"].fillna(0).astype(int)

    log(f"writing SQLite database -> {db_path}")
    save_all(
        db_path, res["vehicles"], res["tests"], scored, kpis, res["quality"], events,
        res.get("anomaly_labels"), braking_events,
    )

    metrics, by_type = an.evaluate_against_labels(scored, events, res.get("anomaly_labels", pd.DataFrame()))
    summary = {
        "engine": engine,
        "rows": int(len(scored)),
        "tests": int(scored["test_id"].nunique()),
        "flagged_points": int(scored["is_anomaly"].sum()),
        "events": int(len(events)),
        "braking_events": int(len(braking_events)),
        "contamination": contamination,
        "seconds": {"pipeline": round(t_pipeline, 1), "model": round(t_model, 1), "total": round(time.time() - t0, 1)},
        "evaluation_vs_injected_labels": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()},
    }
    (processed_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log("done")
    if not quiet:
        print(json.dumps(summary, indent=2))
        if len(by_type):
            print(by_type.round(2).to_string(index=False))
    summary["by_type"] = by_type
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    p.add_argument("--raw-dir", default=str(RAW_DIR))
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--contamination", type=float, default=0.02, help="expected share of anomalous seconds")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    p.add_argument("--n-vehicles", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    run(a.engine, a.raw_dir, a.db, a.contamination, a.n_estimators, a.regenerate, a.n_vehicles, a.seed)


if __name__ == "__main__":
    main()
