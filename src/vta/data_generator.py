"""Synthetic vehicle test-bench data generator.

Produces 1 Hz telemetry for a small fleet running several test scenarios, with
realistic couplings between sensors (speed -> gear -> rpm, load -> temperature, ...)
and injected faults so the anomaly detector can be evaluated against ground truth.

Outputs (CSV, in ``RAW_DIR``):
    vehicles.csv, tests.csv, measurements.csv, anomaly_labels.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, SENSORS

TEST_TYPES = ["urban", "highway", "mixed", "acceleration", "cold_start"]

# (label, fuel_type, engine_cc, mass_kg)
MODELS = [
    ("Compact 1.0 petrol", "petrol", 999, 1090),
    ("Compact 1.5 diesel", "diesel", 1461, 1120),
    ("Hatch 1.2 petrol", "petrol", 1199, 1070),
    ("Saloon 1.5 diesel", "diesel", 1461, 1310),
    ("Hatch 1.4 petrol", "petrol", 1395, 1290),
    ("SUV 1.5 diesel", "diesel", 1461, 1350),
]

GEAR_EDGES = [15, 30, 50, 70, 95, 120]
GEAR_RATIOS = [0.0, 85.0, 52.0, 36.0, 27.0, 21.5, 17.5]  # rpm per km/h, per gear


# --------------------------------------------------------------------------- fleet
def _make_vehicles(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, dict]]:
    degraded = set(rng.choice(n, size=max(1, n // 6), replace=False).tolist())
    rows, hidden = [], {}
    for i in range(n):
        label, fuel, cc, mass = MODELS[i % len(MODELS)]
        vid = f"VEH-{i + 1:02d}"
        rows.append(
            {
                "vehicle_id": vid,
                "model": label,
                "fuel_type": fuel,
                "engine_cc": cc,
                "mass_kg": mass + int(rng.integers(-30, 30)),
                "model_year": int(rng.integers(2019, 2025)),
            }
        )
        worn = i in degraded
        hidden[vid] = {
            "fuel_factor": 1.18 if worn else float(rng.uniform(0.96, 1.04)),
            "vib_factor": 1.5 if worn else float(rng.uniform(0.92, 1.08)),
            "temp_offset": 3.0 if worn else float(rng.uniform(-1.5, 1.5)),
            "oil_factor": 0.93 if worn else 1.0,
        }
    return pd.DataFrame(rows), hidden


# --------------------------------------------------------------------------- profiles
def _speed_profile(test_type: str, n: int, rng: np.random.Generator) -> np.ndarray:
    target = np.zeros(n)
    max_acc, max_dec = 6.0, 9.0  # km/h per second

    def segments(levels, dur_range, stop_prob=0.0):
        i = 0
        while i < n:
            seg = int(rng.integers(*dur_range))
            lvl = 0.0 if rng.random() < stop_prob else float(rng.choice(levels))
            target[i : i + seg] = max(0.0, lvl + (rng.normal(0, 3) if lvl > 0 else 0))
            i += seg

    if test_type == "urban":
        segments([20, 30, 40, 50, 50], (15, 60), stop_prob=0.25)
    elif test_type == "highway":
        target[:60] = np.linspace(0, 115, 60)
        i = 60
        while i < n:
            seg = int(rng.integers(120, 300))
            target[i : i + seg] = rng.uniform(100, 130)
            i += seg
    elif test_type == "mixed":
        segments([30, 50, 70, 90, 110], (30, 120), stop_prob=0.12)
    elif test_type == "cold_start":
        segments([20, 30, 40, 50], (20, 70), stop_prob=0.2)
    elif test_type == "acceleration":
        max_acc, max_dec = 9.0, 14.0
        i = 0
        while i < n:
            idle = int(rng.integers(15, 30))
            run = int(rng.integers(30, 45))
            hold = int(rng.integers(3, 8))
            target[i + idle : i + idle + run + hold] = rng.uniform(100, 140)
            i += idle + run + hold + int(rng.integers(12, 20))
    else:  # pragma: no cover
        raise ValueError(test_type)

    v = np.zeros(n)
    for t in range(1, n):
        v[t] = v[t - 1] + np.clip(target[t] - v[t - 1], -max_dec, max_acc)
    return np.clip(v + rng.normal(0, 0.3, n), 0, None)


def _simulate_test(veh: dict, hidden: dict, test_type: str, n: int, ambient: float, rng) -> dict[str, np.ndarray]:
    speed = _speed_profile(test_type, n, rng)
    acc = np.diff(speed, prepend=speed[0]) / 3.6  # m/s^2
    diesel = veh["fuel_type"] == "diesel"

    thr = 3 + 0.12 * speed + 0.0009 * speed**2 + 18 * np.maximum(acc, 0) - 4 * np.maximum(-acc, 0)
    thr = np.clip(thr + rng.normal(0, 1.0, n), 0, 100)

    gear = np.digitize(speed, GEAR_EDGES)
    idle = (750 if diesel else 800) + rng.normal(0, 15, n)
    rpm = np.where(gear == 0, idle + 33 * speed, np.array(GEAR_RATIOS)[gear] * speed) + 6 * thr
    rpm = rpm * (0.85 if diesel else 1.0) + rng.normal(0, 25, n)
    rpm = np.clip(rpm, 650, 6500)

    temp = np.empty(n)
    temp[0] = ambient + 4 if test_type == "cold_start" else rng.uniform(80, 90)
    for t in range(1, n):
        target = 89 + hidden["temp_offset"] + 0.05 * thr[t] + 0.012 * speed[t]
        tau = 240 if temp[t - 1] < 75 else 70
        temp[t] = temp[t - 1] + (target - temp[t - 1]) / tau + rng.normal(0, 0.08)

    oil = (0.8 + 0.00085 * rpm + 0.008 * (90 - temp)) * hidden["oil_factor"] + rng.normal(0, 0.04, n)
    battery = 13.7 + 0.2 * (rpm > 1500) - 0.15 * (thr > 60) + rng.normal(0, 0.05, n)
    vib = (0.06 + 0.00005 * rpm + 0.0012 * speed) * hidden["vib_factor"] + rng.normal(0, 0.015, n)
    fuel = (0.7 + 0.1 * thr * rpm / 1000) * (veh["engine_cc"] / 1400) ** 0.6
    fuel = fuel * (0.85 if diesel else 1.0) * hidden["fuel_factor"] + rng.normal(0, 0.1, n)

    return {
        "speed_kmh": speed,
        "rpm": rpm,
        "throttle_pct": thr,
        "engine_temp_c": temp,
        "oil_pressure_bar": np.clip(oil, 0.2, None),
        "fuel_rate_lph": np.clip(fuel, 0.1, None),
        "battery_v": battery,
        "vibration_g": np.clip(vib, 0.01, None),
    }


# --------------------------------------------------------------------------- anomalies
def _overheat(d, s, e, rng):
    dur = e - s
    up = int(dur * 0.7)
    shape = np.concatenate([np.linspace(0, 1, up), np.linspace(1, 0, dur - up)])
    d["engine_temp_c"][s:e] += rng.uniform(22, 34) * shape
    d["oil_pressure_bar"][s:e] *= 1 - 0.12 * shape
    return "engine_temp_c", s + int(dur * 0.35), s + int(dur * 0.85)


def _oil_drop(d, s, e, rng):
    d["oil_pressure_bar"][s:e] *= rng.uniform(0.4, 0.6)
    return "oil_pressure_bar", s, e


def _vibration(d, s, e, rng):
    d["vibration_g"][s:e] = d["vibration_g"][s:e] * rng.uniform(3, 5) + rng.normal(0, 0.03, e - s)
    return "vibration_g", s, e


def _battery(d, s, e, rng):
    d["battery_v"][s:e] = rng.uniform(11.0, 12.2) + rng.normal(0, 0.08, e - s)
    return "battery_v", s, e


def _fuel(d, s, e, rng):
    d["fuel_rate_lph"][s:e] *= rng.uniform(2.2, 3.2)
    return "fuel_rate_lph", s, e


def _rpm_glitch(d, s, e, rng):
    d["rpm"][s:e] = np.minimum(d["rpm"][s:e] + rng.uniform(1800, 2800), 6500)
    return "rpm", s, e


def _dropout(d, s, e, rng):
    sensor = str(rng.choice(SENSORS))
    d[sensor][s:e] = np.nan
    return sensor, s, e


def _sentinel(d, s, e, rng):
    sensor = str(rng.choice(SENSORS))
    d[sensor][s:e] = -999.0
    return sensor, s, e


# name -> (function, duration range in seconds, category, sampling weight)
INJECTORS = {
    "overheating": (_overheat, (80, 140), "behavior", 0.15),
    "oil_pressure_drop": (_oil_drop, (30, 70), "behavior", 0.15),
    "vibration_burst": (_vibration, (8, 25), "behavior", 0.20),
    "battery_undervoltage": (_battery, (40, 100), "behavior", 0.12),
    "fuel_spike": (_fuel, (12, 35), "behavior", 0.15),
    "rpm_glitch": (_rpm_glitch, (4, 9), "behavior", 0.13),
    "sensor_dropout": (_dropout, (3, 18), "quality", 0.05),
    "sentinel_value": (_sentinel, (1, 4), "quality", 0.05),
}


def _inject(d: dict[str, np.ndarray], n: int, test_id: str, rng) -> list[dict]:
    names = list(INJECTORS)
    probs = np.array([INJECTORS[k][3] for k in names])
    k = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
    labels, used = [], []
    for name in rng.choice(names, size=k, p=probs / probs.sum()):
        fn, (lo, hi), category, _ = INJECTORS[str(name)]
        dur = int(rng.integers(lo, hi))
        placed = False
        for _ in range(20):
            start = int(rng.integers(90, max(91, n - dur - 30)))
            if all(start + dur + 20 < a or start > b + 20 for a, b in used):
                placed = True
                break
        if not placed:
            continue
        sensor, cs, ce = fn(d, start, start + dur, rng)
        used.append((start, start + dur))
        labels.append(
            {
                "test_id": test_id,
                "anomaly_type": str(name),
                "category": category,
                "sensor": sensor,
                "start_s": int(cs),
                "end_s": int(ce),
            }
        )
    return labels


# --------------------------------------------------------------------------- public API
def generate_dataset(
    n_vehicles: int = 12,
    tests_per_vehicle: int = 4,
    seed: int = 42,
    anomaly_prob: float = 0.75,
    out_dir: Path | str = RAW_DIR,
) -> dict[str, pd.DataFrame]:
    """Generate and write the synthetic dataset. Returns the four DataFrames."""
    rng = np.random.default_rng(seed)
    vehicles, hidden = _make_vehicles(n_vehicles, rng)

    tests, frames, labels = [], [], []
    base = pd.Timestamp("2026-09-01 08:00:00")
    counter = 0
    for veh in vehicles.to_dict("records"):
        types = rng.choice(TEST_TYPES, size=min(tests_per_vehicle, len(TEST_TYPES)), replace=False)
        for tt in types:
            counter += 1
            test_id = f"T{counter:03d}"
            n = int(rng.integers(900, 1500))
            ambient = float(np.round(rng.uniform(5, 18) if tt == "cold_start" else rng.uniform(15, 32), 1))
            start = base + pd.Timedelta(days=int(rng.integers(0, 28)), minutes=int(rng.integers(0, 600)))
            data = _simulate_test(veh, hidden[veh["vehicle_id"]], str(tt), n, ambient, rng)
            if rng.random() < anomaly_prob:
                labels += _inject(data, n, test_id, rng)

            ts = pd.date_range(start, periods=n, freq="s").strftime("%Y-%m-%d %H:%M:%S")
            frame = pd.DataFrame({"test_id": test_id, "timestamp": ts, **data})
            frames.append(frame)
            tests.append(
                {
                    "test_id": test_id,
                    "vehicle_id": veh["vehicle_id"],
                    "test_type": str(tt),
                    "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_s": n,
                    "ambient_temp_c": ambient,
                }
            )

    meas = pd.concat(frames, ignore_index=True)
    meas[SENSORS] = meas[SENSORS].round(3)

    # Raw-data realism: a few duplicated rows and an unsorted file.
    dups = meas.sample(frac=0.0005, random_state=seed)
    meas = pd.concat([meas, dups], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out = {
        "vehicles": vehicles,
        "tests": pd.DataFrame(tests),
        "measurements": meas,
        "anomaly_labels": pd.DataFrame(
            labels, columns=["test_id", "anomaly_type", "category", "sensor", "start_s", "end_s"]
        ),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":  # pragma: no cover
    res = generate_dataset()
    print({k: v.shape for k, v in res.items()})
