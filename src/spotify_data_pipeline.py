"""Validate the Kaggle snapshot and produce tables with explicit row grains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOUNDED = ["danceability", "energy", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence"]
NUMERIC = BOUNDED + ["popularity", "duration_ms", "key", "loudness", "mode",
                     "tempo", "time_signature"]
TEXT = ["track_id", "artists", "album_name", "track_name", "track_genre"]
REQUIRED = TEXT + NUMERIC + ["explicit"]


def prepare_tables(raw: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Reject invalid rows, preserve memberships, and report ambiguous identities.

    A canonical popularity value is the median of distinct observed values per
    track. Repeated genre rows therefore cannot change the chosen value.
    Conflicting non-popularity track attributes stop the pipeline for review.
    """
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = raw[REQUIRED].copy().reset_index(drop=True)
    data.insert(0, "source_row", np.arange(len(data)))
    for col in TEXT:
        data[col] = data[col].astype("string").str.strip().replace("", pd.NA)
    data["album_name"] = data["album_name"].fillna("Unknown")
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["explicit"] = data["explicit"].astype("string").str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    checks = {
        "missing_identity": data[["track_id", "artists", "track_name", "track_genre"]].isna().any(axis=1),
        "invalid_numeric": ~np.isfinite(data[NUMERIC].to_numpy(dtype=float)).all(axis=1),
        "invalid_explicit": data["explicit"].isna(),
        "invalid_popularity": ~data["popularity"].between(0, 100),
        "invalid_duration": ~data["duration_ms"].gt(0),
        "invalid_tempo": ~data["tempo"].gt(0),
        "invalid_key": ~data["key"].isin(range(-1, 12)),
        "invalid_mode": ~data["mode"].isin([0, 1]),
        "invalid_meter": ~data["time_signature"].isin(range(8)),
    }
    for col in BOUNDED:
        checks[f"invalid_{col}"] = ~data[col].between(0, 1)
    flags = pd.DataFrame(checks, index=data.index).fillna(True)
    invalid = flags.any(axis=1)
    rejected = data.loc[invalid, ["source_row", "track_id"]].copy()
    rejected["reasons"] = [
        ";".join(flags.columns[row]) for row in flags.loc[invalid].to_numpy(dtype=bool)
    ]
    valid = data.loc[~invalid].drop(columns="source_row").copy()
    duplicate_count = int(valid.duplicated().sum())
    valid = valid.drop_duplicates().sort_values(
        ["track_id", "track_genre", "popularity"], kind="stable"
    ).reset_index(drop=True)
    invariant = [c for c in REQUIRED if c not in ["track_id", "track_genre", "popularity"]]
    conflicts = valid.groupby("track_id")[invariant].nunique().gt(1)
    if conflicts.any().any():
        ids = conflicts.index[conflicts.any(axis=1)].tolist()
        raise ValueError(f"Conflicting track attributes; review track IDs: {ids[:10]}")
    tracks = valid.drop_duplicates("track_id").drop(columns="track_genre").copy()
    distinct_popularity = valid[["track_id", "popularity"]].drop_duplicates()
    medians = distinct_popularity.groupby("track_id")["popularity"].median()
    tracks["popularity"] = tracks["track_id"].map(medians)
    tracks["duration_min"] = tracks["duration_ms"] / 60000
    tracks["explicit_int"] = tracks["explicit"].astype(int)
    tracks["explicit"] = tracks["explicit"].astype(bool)
    memberships = valid[["track_id", "track_genre"]].drop_duplicates().reset_index(drop=True)
    tables = {"tracks": tracks.reset_index(drop=True), "track_genres": memberships,
              "observations": valid, "rejected_rows": rejected.reset_index(drop=True)}
    report = {
        "schema_version": 1, "input_rows": len(raw), "rejected_rows": int(invalid.sum()),
        "rule_failures": {k: int(v) for k, v in flags.sum().items()},
        "exact_duplicates_removed": duplicate_count, "observations": len(valid),
        "unique_tracks": len(tracks), "track_genre_memberships": len(memberships),
        "genres": int(memberships["track_genre"].nunique()),
        "popularity_conflict_tracks": int(distinct_popularity.groupby("track_id").size().gt(1).sum()),
    }
    return tables, report


def run_pipeline(source: Path, output: Path) -> dict:
    """Overwrite deterministic derived tables; never append or change raw data."""
    payload = source.read_bytes()
    # Parse the same bytes that are fingerprinted for lineage.
    from io import BytesIO
    tables, report = prepare_tables(pd.read_csv(BytesIO(payload), dtype={"track_id": "string"}))
    report.update({"source_name": source.name, "source_sha256": hashlib.sha256(payload).hexdigest(),
                   "pandas_version": pd.__version__, "numpy_version": np.__version__})
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/canonical")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output_dir.resolve()
    if source == output or source.is_relative_to(output):
        parser.error("Output directory must not contain the raw input file.")
    print(json.dumps(run_pipeline(source, output), indent=2))


if __name__ == "__main__":
    main()
