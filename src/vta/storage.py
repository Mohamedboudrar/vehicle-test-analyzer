"""SQLite persistence and read-only query helpers."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .config import DB_PATH, SQL_DIR


def init_db(db_path: Path | str = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript((SQL_DIR / "schema.sql").read_text())


def _columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _append(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    """Append only the columns that exist in the table schema, in schema order."""
    cols = [c for c in _columns(con, table) if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif out[c].dtype == bool:
            out[c] = out[c].astype(int)
    out.to_sql(table, con, if_exists="append", index=False, chunksize=20_000)


def save_all(
    db_path: Path | str,
    vehicles: pd.DataFrame,
    tests: pd.DataFrame,
    measurements: pd.DataFrame,
    kpis: pd.DataFrame,
    quality: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    braking: pd.DataFrame | None = None,
) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        _append(con, "vehicles", vehicles)
        _append(con, "tests", tests)
        _append(con, "measurements", measurements)
        _append(con, "test_kpis", kpis)
        _append(con, "data_quality", quality)
        _append(con, "anomaly_events", events)
        if labels is not None and len(labels):
            _append(con, "injected_anomalies", labels)
        if braking is not None and len(braking):
            _append(con, "braking_events", braking)


def connect_ro(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Read-only connection (SQLite URI mode)."""
    return sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)


def read_sql(query: str, db_path: Path | str = DB_PATH, params: tuple | dict | None = None) -> pd.DataFrame:
    with connect_ro(db_path) as con:
        return pd.read_sql_query(query, con, params=params)


def load_named_queries(path: Path | str | None = None) -> dict[str, dict[str, str]]:
    """Parse ``queries.sql`` into ``{name: {"desc": ..., "sql": ...}}``."""
    text = Path(path or SQL_DIR / "queries.sql").read_text()
    out: dict[str, dict[str, str]] = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        name, _, rest = block.partition("\n")
        desc_m = re.match(r"-- desc:\s*(.*)\n", rest)
        desc = desc_m.group(1).strip() if desc_m else ""
        sql = rest[desc_m.end():] if desc_m else rest
        out[name.strip()] = {"desc": desc, "sql": sql.strip()}
    return out


def is_read_only_query(sql: str) -> bool:
    """Allow a single SELECT / WITH statement only (used by the dashboard SQL explorer)."""
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return False
    return bool(re.match(r"^(select|with)\b", stripped, flags=re.I))
