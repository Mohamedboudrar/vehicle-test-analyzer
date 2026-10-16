"""Run the analytical queries from ``sql/queries.sql`` and print the results.

    python -m vta.sql_report                      # all queries
    python -m vta.sql_report fleet_fuel_ranking   # one query
"""
from __future__ import annotations

import sys

import pandas as pd

from .storage import load_named_queries, read_sql


def main(argv: list[str] | None = None) -> None:
    names = (argv if argv is not None else sys.argv[1:]) or None
    queries = load_named_queries()
    for name, q in queries.items():
        if names and name not in names:
            continue
        print(f"\n=== {name} — {q['desc']}")
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(read_sql(q["sql"]).to_string(index=False))


if __name__ == "__main__":
    main()
