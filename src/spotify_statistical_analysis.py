"""Analyze audio features and popularity from the canonical PostgreSQL snapshot."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from spotify_data_pipeline import ROOT, prepare_tables

FEATURES = (
    "danceability", "energy", "valence", "acousticness", "instrumentalness",
    "tempo", "loudness",
)


def load_analysis_data(connection, source: Path):
    """Read a consistent database snapshot and recover conflict IDs from its raw input."""
    columns = sql.SQL(", ").join(map(sql.Identifier, ("track_id", "popularity", *FEATURES)))
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        snapshot = connection.execute(
            "SELECT source_sha256, quality_report FROM spotify.dataset_snapshot"
        ).fetchone()
        if snapshot is None:
            raise ValueError("Load the PostgreSQL catalog before running the analysis.")
        fingerprint, quality = snapshot
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != fingerprint:
            raise ValueError("The raw source hash differs from the loaded PostgreSQL snapshot.")
        tracks = pd.DataFrame(
            connection.execute(
                sql.SQL("SELECT {} FROM spotify.tracks ORDER BY track_id").format(columns)
            ).fetchall(),
            columns=["track_id", "popularity", *FEATURES],
        )
        memberships = pd.DataFrame(
            connection.execute(
                "SELECT track_id, genre_name FROM spotify.track_genres ORDER BY track_id, genre_name"
            ).fetchall(),
            columns=["track_id", "genre_name"],
        )
    if len(tracks) != quality["unique_tracks"] or len(memberships) != quality["track_genre_memberships"]:
        raise ValueError("Database counts differ from the recorded snapshot quality report.")
    tables, _ = prepare_tables(pd.read_csv(BytesIO(payload), dtype={"track_id": "string"}))
    observations = tables["observations"]
    conflicts = observations.groupby("track_id")["popularity"].nunique().gt(1)
    conflict_ids = conflicts.index[conflicts]
    if len(conflict_ids) != quality["popularity_conflict_tracks"]:
        raise ValueError("Recovered popularity conflicts differ from the snapshot quality report.")
    return tracks, memberships, conflict_ids, fingerprint


class AssociationModel:
    """Estimate separate linear associations overall and with genre fixed effects."""

    def __init__(self, tracks, memberships, iqr):
        self.x = tracks.loc[:, FEATURES].to_numpy().T / iqr[:, None]
        self.y = tracks["popularity"].to_numpy()
        self.track = pd.Index(tracks["track_id"]).get_indexer(memberships["track_id"])
        if len(self.track) == 0 or np.any(self.track < 0):
            raise ValueError("Genre memberships must reference existing tracks.")
        self.genre, genres = pd.factorize(memberships["genre_name"], sort=True)
        self.genre_count = len(genres)
        membership_counts = np.bincount(self.track, minlength=len(tracks))
        if (membership_counts == 0).any():
            raise ValueError("Every analysis track must have a genre membership.")
        self.membership_weight = 1 / membership_counts[self.track]
        self.joined_x = self.x[:, self.track]
        self.joined_y = self.y[self.track]
        self.joined_x2 = self.joined_x ** 2
        self.joined_xy = self.joined_x * self.joined_y

    def estimate(self, weights):
        """Use track weights globally and split each track's weight across its genres."""
        normalized = weights / weights.sum()
        centered_x = self.x - np.sum(self.x * normalized, axis=1)[:, None]
        centered_y = self.y - np.sum(self.y * normalized)
        variance = np.sum(centered_x ** 2 * normalized, axis=1)
        overall = np.sum(centered_x * centered_y * normalized, axis=1) / variance

        def totals(values):
            return np.bincount(self.genre, weights=values, minlength=self.genre_count)

        member_weights = weights[self.track] * self.membership_weight
        group_weights = totals(member_weights)
        safe_group_weights = np.where(group_weights > 0, group_weights, 1)
        group_y_mean = totals(member_weights * self.joined_y) / safe_group_weights
        adjusted = []
        # Weighted within-genre sums of squares and cross-products. Recompute
        # genre means for every bootstrap sample to refit the fixed effects.
        for x, x2, xy in zip(self.joined_x, self.joined_x2, self.joined_xy):
            group_x_total = totals(member_weights * x)
            numerator = np.sum(member_weights * xy) - np.sum(group_x_total * group_y_mean)
            denominator = np.sum(member_weights * x2) - np.sum(group_x_total ** 2 / safe_group_weights)
            adjusted.append(numerator / denominator if denominator > 1e-10 else np.nan)
        return np.array([overall, adjusted])


def coefficient_table(tracks, memberships, conflict_ids, resamples, seed):
    quartiles = tracks.loc[:, FEATURES].quantile([0.25, 0.75])
    iqr = (quartiles.loc[0.75] - quartiles.loc[0.25]).to_numpy()
    if not np.all(iqr > 0):
        raise ValueError("All selected audio features must have a positive interquartile range.")
    model = AssociationModel(tracks, memberships, iqr)
    overall, adjusted = model.estimate(np.ones(len(tracks)))
    without_conflicts = ~tracks["track_id"].isin(conflict_ids).to_numpy()
    clean_overall, clean_adjusted = model.estimate(without_conflicts)

    rng = np.random.default_rng(seed)
    estimates = np.empty((resamples, 2, len(FEATURES)))
    for i in range(resamples):
        selected = rng.integers(0, len(tracks), size=len(tracks))
        estimates[i] = model.estimate(np.bincount(selected, minlength=len(tracks)))
    if not np.isfinite(estimates).all():
        raise ValueError("A bootstrap sample yielded an undefined slope.")
    lower, upper = np.quantile(estimates, [0.025, 0.975], axis=0)
    diff_lower, diff_upper = np.quantile(estimates[:, 1] - estimates[:, 0], [0.025, 0.975], axis=0)
    # Pearson correlation of average ranks is Spearman with tie handling.
    ranks = tracks.loc[:, [*FEATURES, "popularity"]].rank()
    return pd.DataFrame({
        "feature": FEATURES,
        "feature_q25": quartiles.loc[0.25].to_numpy(),
        "feature_q75": quartiles.loc[0.75].to_numpy(),
        "feature_iqr": iqr,
        "spearman_rho": ranks.corr()["popularity"].loc[list(FEATURES)].to_numpy(),
        "overall_points": overall,
        "overall_ci_low": lower[0],
        "overall_ci_high": upper[0],
        "genre_adjusted_points": adjusted,
        "genre_adjusted_ci_low": lower[1],
        "genre_adjusted_ci_high": upper[1],
        "adjusted_minus_overall": adjusted - overall,
        "difference_ci_low": diff_lower,
        "difference_ci_high": diff_upper,
        "overall_excluding_conflicts": clean_overall,
        "genre_adjusted_excluding_conflicts": clean_adjusted,
    })


def describe_genres(tracks, memberships):
    grouped = memberships.merge(tracks, on="track_id", validate="many_to_one").groupby("genre_name")
    summaries = grouped.agg(
        track_count=("track_id", "size"),
        popularity_median=("popularity", "median"),
        popularity_q25=("popularity", lambda values: values.quantile(0.25)),
        popularity_q75=("popularity", lambda values: values.quantile(0.75)),
        **{f"median_{feature}": (feature, "median") for feature in FEATURES},
    )
    return summaries.reset_index()


def describe_trends(tracks):
    """Use quantile buckets without splitting ties, and retain visible sample counts."""
    results = []
    for feature in FEATURES:
        bands = pd.qcut(tracks[feature], q=10, duplicates="drop")
        summary = tracks.groupby(bands, observed=True).agg(
            median_feature=(feature, "median"), mean_popularity=("popularity", "mean"),
            median_popularity=("popularity", "median"), track_count=("popularity", "size"),
        )
        summary.insert(0, "feature", feature)
        results.append(summary.reset_index(drop=True))
    return pd.concat(results, ignore_index=True)


def save_charts(effects, genres, trends, output):
    overall_color, adjusted_color = "#35486a", "#00826c"
    fig, ax = plt.subplots(figsize=(8.6, 4.7), layout="constrained")
    positions = np.arange(len(effects))
    for stem, delta, label, color in (
        ("overall", -0.13, "Overall", overall_color),
        ("genre_adjusted", 0.13, "Adjusted for genre", adjusted_color),
    ):
        y = positions + delta
        ax.hlines(y, effects[stem + "_ci_low"], effects[stem + "_ci_high"], color=color)
        ax.scatter(effects[stem + "_points"], y, label=label, color=color, s=30, zorder=3)
    ax.axvline(0, color="#7a828a", linewidth=1)
    ax.set_yticks(positions, effects["feature"].str.capitalize())
    ax.invert_yaxis()
    ax.set_title("Audio features and popularity before and after genre adjustment", loc="left")
    ax.set_xlabel("Popularity points per feature interquartile range (95% bootstrap intervals)")
    ax.legend(loc="best", frameon=False)
    fig.savefig(output / "audio_feature_associations.png", dpi=160)
    plt.close(fig)

    median_order = genres.sort_values(["popularity_median", "genre_name"], ascending=[False, True])
    shown = pd.concat([median_order.head(6), median_order.tail(6)]).drop_duplicates("genre_name")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    ax.hlines(shown["genre_name"], shown["popularity_q25"], shown["popularity_q75"],
              color=overall_color, linewidth=3, alpha=0.65)
    ax.scatter(shown["popularity_median"], shown["genre_name"], color=adjusted_color, zorder=3)
    ax.invert_yaxis()
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlim(-2, 100)  # Leave room for the median marker at zero.
    ax.set_title("Popularity in genres with the highest and lowest medians", loc="left")
    ax.set_xlabel("Popularity score (dot = median, line = middle 50% of tracks)")
    fig.savefig(output / "genre_popularity_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, layout="constrained")
    for ax, feature in zip(axes.flat, FEATURES):
        points = trends[trends["feature"] == feature]
        ax.plot(points["median_feature"], points["mean_popularity"], "o-", color=adjusted_color)
        label = {"tempo": "Tempo (BPM)", "loudness": "Loudness (dB)"}.get(feature, feature.capitalize())
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Mean popularity across audio feature bins")
    fig.supxlabel("Median audio feature value in each bin")
    fig.supylabel("Mean popularity score")
    fig.savefig(output / "audio_feature_trends.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/statistics")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 100 or args.seed < 0:
        parser.error("Use at least 100 bootstrap resamples and a nonnegative seed.")
    dsn = os.environ.get("SPOTIFY_DATABASE_URL")
    if not dsn:
        parser.error("Set SPOTIFY_DATABASE_URL to the loaded project database.")
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as connection:
            tracks, memberships, conflict_ids, fingerprint = load_analysis_data(connection, args.input)
    except psycopg.Error as error:
        parser.exit(1, f"PostgreSQL read failed ({type(error).__name__}). Check the database and load the catalog.\n")
    except ValueError as error:
        parser.exit(1, f"{error}\n")

    matplotlib.use("Agg")
    effects = coefficient_table(tracks, memberships, conflict_ids, args.resamples, args.seed)
    genres = describe_genres(tracks, memberships)
    trends = describe_trends(tracks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("audio_associations", effects), ("genre_profiles", genres), ("audio_trends", trends)):
        table.to_csv(args.output_dir / f"{name}.csv", index=False, float_format="%.8g")
    save_charts(effects, genres, trends, args.output_dir)
    manifest = {
        "source_sha256": fingerprint, "track_count": len(tracks), "genre_count": len(genres),
        "membership_count": len(memberships), "popularity_conflict_tracks": len(conflict_ids),
        "seed": args.seed, "bootstrap_resamples": args.resamples,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
