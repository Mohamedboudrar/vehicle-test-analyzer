"""Cross-check: the PySpark pipeline must reproduce the Pandas pipeline."""
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:  # pragma: no cover
    pytest.skip("Java runtime not available", allow_module_level=True)

from vta.config import MODEL_FEATURES
from vta.data_generator import generate_dataset
from vta.pipeline_pandas import run_pandas
from vta.pipeline_spark import run_spark


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw_small")
    generate_dataset(n_vehicles=3, tests_per_vehicle=2, seed=11, out_dir=raw)
    return run_pandas(raw), run_spark(raw, tmp_path_factory.mktemp("spark_out"))


def test_features_match(both):
    p, s = both
    fp = p["features"].sort_values(["test_id", "t_s"]).reset_index(drop=True)
    fs = s["features"]
    assert len(fp) == len(fs)
    assert (fp["is_valid"].to_numpy() == fs["is_valid"].to_numpy()).all()
    m = fp["is_valid"].to_numpy()
    for col in MODEL_FEATURES:
        np.testing.assert_allclose(fp.loc[m, col], fs.loc[m, col], rtol=1e-6, atol=1e-3, err_msg=col)


def test_kpis_and_quality_match(both):
    p, s = both
    k1 = p["kpis"].sort_values("test_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(k1, s["kpis"][k1.columns], check_dtype=False, rtol=1e-9, atol=1e-9)
    q1 = p["quality"].sort_values("test_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(q1, s["quality"][q1.columns], check_dtype=False)
