import pytest

from vta.data_generator import generate_dataset


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    """Full default synthetic dataset (seed 42), generated once per test session."""
    d = tmp_path_factory.mktemp("raw")
    generate_dataset(out_dir=d)
    return d
