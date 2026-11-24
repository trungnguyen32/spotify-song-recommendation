"""Export the four documented PostgreSQL analyses to CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import psycopg

from spotify_data_pipeline import ROOT

RESULT_NAMES = ["genre_popularity_ranking", "top_tracks_by_genre",
                "energy_band_summary", "genre_audio_profiles"]


def export_analytics(connection: psycopg.Connection, output: Path) -> dict:
    """Read all results from the same snapshot, then write local exports."""
    results = []
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute("SELECT source_sha256 FROM spotify.dataset_snapshot")
            snapshot = cursor.fetchone()
            if snapshot is None:
                raise ValueError("Load a dataset snapshot before exporting analytics")
            cursor.execute((ROOT / "sql/analytics.sql").read_text(encoding="utf-8"))
            while True:
                if cursor.description:
                    results.append(([col.name for col in cursor.description], cursor.fetchall()))
                if not cursor.nextset():
                    break
    if len(results) != len(RESULT_NAMES):
        raise RuntimeError("SQL result count differs from the documented export contract")
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, (columns, rows) in zip(RESULT_NAMES, results):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        counts[name] = len(rows)
    manifest = {"source_sha256": snapshot[0], "result_rows": counts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/sql")
    args = parser.parse_args()
    dsn = os.environ.get("SPOTIFY_DATABASE_URL")
    if not dsn:
        parser.error("Set SPOTIFY_DATABASE_URL to the PostgreSQL database connection string.")
    try:
        with psycopg.connect(dsn, connect_timeout=10, autocommit=True) as connection:
            manifest = export_analytics(connection, args.output_dir)
    except psycopg.Error as error:
        parser.exit(1, f"SQL export failed ({type(error).__name__}); check database access and run the loader first.\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
