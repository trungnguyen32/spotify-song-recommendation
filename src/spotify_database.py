"""Load a validated raw snapshot into PostgreSQL in one transaction."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from spotify_data_pipeline import ROOT, prepare_tables

TRACK_COLUMNS = ["track_id", "artists", "album_name", "track_name", "popularity",
                 "duration_ms", "explicit", "danceability", "energy", "key", "loudness",
                 "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
                 "valence", "tempo", "time_signature"]


def load_snapshot(connection: psycopg.Connection, source: Path) -> dict:
    """Replace only spotify's catalog, including stale rows, or roll back entirely.

    Validation happens before database writes. The caller owns the connection;
    an idle connection commits on success, while an existing transaction receives
    a savepoint and remains the caller's responsibility.
    """
    payload = source.read_bytes()
    tables, report = prepare_tables(pd.read_csv(BytesIO(payload), dtype={"track_id": "string"}))
    if tables["tracks"].empty:
        raise ValueError("Refusing to replace the catalog with an empty validated snapshot.")
    report.update(source_name=source.name, source_sha256=hashlib.sha256(payload).hexdigest())
    tracks = tables["tracks"][TRACK_COLUMNS].copy()
    # COPY uses the text representation supplied by the client, not a server cast
    # from float. Integer fields must therefore be encoded as integers explicitly.
    for column in ["key", "mode", "time_signature"]:
        tracks[column] = tracks[column].astype(int)
    genres = sorted(tables["track_genres"]["track_genre"].unique().tolist())
    with connection.transaction():
        with connection.cursor() as cursor:
            # Serialize concurrent loaders for this application schema.
            cursor.execute("SELECT pg_advisory_xact_lock(7028114)")
            cursor.execute((ROOT / "sql/schema.sql").read_text(encoding="utf-8"))
            cursor.execute("LOCK TABLE spotify.tracks, spotify.genres, spotify.track_genres, "
                           "spotify.dataset_snapshot IN EXCLUSIVE MODE")
            cursor.execute("DELETE FROM spotify.track_genres")
            cursor.execute("DELETE FROM spotify.tracks")
            cursor.execute("DELETE FROM spotify.genres")
            with cursor.copy("COPY spotify.tracks (" + ", ".join(TRACK_COLUMNS) + ") FROM STDIN") as copy:
                for row in tracks.itertuples(index=False, name=None):
                    copy.write_row(row)
            with cursor.copy("COPY spotify.genres (genre_name) FROM STDIN") as copy:
                for genre in genres:
                    copy.write_row((genre,))
            with cursor.copy("COPY spotify.track_genres (track_id, genre_name) FROM STDIN") as copy:
                for row in tables["track_genres"].itertuples(index=False, name=None):
                    copy.write_row(row)
            cursor.execute("DELETE FROM spotify.dataset_snapshot")
            cursor.execute(
                "INSERT INTO spotify.dataset_snapshot (source_name, source_sha256, quality_report) "
                "VALUES (%s, %s, %s)",
                (source.name, report["source_sha256"], Jsonb(report)),
            )
            cursor.execute((ROOT / "sql/views.sql").read_text(encoding="utf-8"))
            for table, expected in [("tracks", len(tracks)), ("genres", len(genres)),
                                    ("track_genres", len(tables["track_genres"]))]:
                cursor.execute(f"SELECT COUNT(*) FROM spotify.{table}")
                if cursor.fetchone()[0] != expected:
                    raise RuntimeError(f"Loaded {table} count differs from validated input")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/dataset.csv")
    args = parser.parse_args()
    dsn = os.environ.get("SPOTIFY_DATABASE_URL")
    if not dsn:
        parser.error("Set SPOTIFY_DATABASE_URL to the target PostgreSQL database connection string.")
    try:
        with psycopg.connect(dsn, connect_timeout=10) as connection:
            report = load_snapshot(connection, args.input)
    except psycopg.Error as error:
        # Do not print connection credentials or server diagnostics containing data.
        parser.exit(1, f"PostgreSQL load failed ({type(error).__name__}); check database access and schema permissions.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
