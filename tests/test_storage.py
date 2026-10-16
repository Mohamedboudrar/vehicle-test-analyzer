import sqlite3

import pytest

from vta.run_pipeline import run
from vta.storage import is_read_only_query, load_named_queries, read_sql


@pytest.fixture(scope="module")
def db(dataset_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("db")
    path = out / "t.db"
    run(engine="pandas", raw_dir=dataset_dir, db_path=path, quiet=True, processed_dir=out / "processed")
    return path


def test_tables_are_populated_and_consistent(db):
    counts = {t: read_sql(f"SELECT COUNT(*) AS n FROM {t}", db).n[0] for t in
              ["vehicles", "tests", "measurements", "test_kpis", "data_quality", "anomaly_events",
               "injected_anomalies", "braking_events"]}
    assert counts["vehicles"] == 12 and counts["tests"] == 48 and counts["test_kpis"] == 48
    assert counts["measurements"] > 50_000 and counts["anomaly_events"] > 0 and counts["braking_events"] > 0
    # KPI event counts agree with the events table
    mismatch = read_sql(
        "SELECT COUNT(*) AS n FROM test_kpis k WHERE k.n_anomaly_events <> "
        "(SELECT COUNT(*) FROM anomaly_events e WHERE e.test_id = k.test_id)", db).n[0]
    assert mismatch == 0
    # KPI braking-event counts agree with the braking_events table
    mismatch = read_sql(
        "SELECT COUNT(*) AS n FROM test_kpis k WHERE k.n_braking_events <> "
        "(SELECT COUNT(*) FROM braking_events b WHERE b.test_id = k.test_id)", db).n[0]
    assert mismatch == 0


def test_foreign_keys_hold(db):
    with sqlite3.connect(db) as con:
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_all_named_queries_run(db):
    queries = load_named_queries()
    assert len(queries) >= 7
    for name, q in queries.items():
        assert is_read_only_query(q["sql"]), name
        df = read_sql(q["sql"], db)
        assert len(df) > 0, name


def test_read_only_guard():
    assert is_read_only_query("SELECT 1")
    assert is_read_only_query("  with a as (select 1) select * from a;  ")
    assert not is_read_only_query("DELETE FROM tests")
    assert not is_read_only_query("SELECT 1; DROP TABLE tests")
    assert not is_read_only_query("PRAGMA writable_schema = 1")


def test_read_connection_is_read_only(db):
    with pytest.raises(Exception):
        read_sql("INSERT INTO vehicles (vehicle_id, model, fuel_type) VALUES ('X','m','petrol')", db)
