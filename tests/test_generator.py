import pandas as pd

from vta.config import SENSORS
from vta.data_generator import generate_dataset


def test_files_and_columns(dataset_dir):
    meas = pd.read_csv(dataset_dir / "measurements.csv")
    assert list(meas.columns) == ["test_id", "timestamp"] + SENSORS
    tests = pd.read_csv(dataset_dir / "tests.csv")
    veh = pd.read_csv(dataset_dir / "vehicles.csv")
    assert len(veh) == 12 and len(tests) == 48
    assert set(tests["vehicle_id"]) <= set(veh["vehicle_id"])
    assert set(meas["test_id"]) == set(tests["test_id"])


def test_raw_data_is_messy_on_purpose(dataset_dir):
    meas = pd.read_csv(dataset_dir / "measurements.csv")
    assert meas.duplicated(["test_id", "timestamp"]).any()  # duplicates injected
    assert not meas["timestamp"].is_monotonic_increasing  # file is shuffled


def test_labels_are_consistent(dataset_dir):
    lab = pd.read_csv(dataset_dir / "anomaly_labels.csv")
    tests = pd.read_csv(dataset_dir / "tests.csv").set_index("test_id")
    assert len(lab) > 30
    assert set(lab["category"]) <= {"behavior", "quality"}
    assert (lab["start_s"] < lab["end_s"]).all()
    assert (lab["end_s"] <= lab["test_id"].map(tests["duration_s"])).all()


def test_generation_is_deterministic(tmp_path):
    a = generate_dataset(n_vehicles=3, tests_per_vehicle=2, seed=7, out_dir=tmp_path / "a")
    b = generate_dataset(n_vehicles=3, tests_per_vehicle=2, seed=7, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a["measurements"], b["measurements"])
    pd.testing.assert_frame_equal(a["anomaly_labels"], b["anomaly_labels"])
