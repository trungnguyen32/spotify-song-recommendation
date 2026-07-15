"""Reusable EDA focused on relationships between song features and popularity.

The functions in this module support the EDA notebook and can also generate a
small, reproducible report from the command line. Results describe association,
not causation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_cleaned_eda.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

POPULARITY_TARGET = "popularity"
GENRE_COLUMN = "track_genre"
ID_COLUMN = "track_id"
CONTINUOUS_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_min",
]
CATEGORICAL_FEATURES = ["explicit_int", "key", "mode", "time_signature"]


def load_eda_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the columns required by the popularity analysis."""
    data = pd.read_csv(path)
    return prepare_popularity_analysis_data(data)


def prepare_popularity_analysis_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate analysis fields and retain one row per track when IDs exist."""
    required = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + [POPULARITY_TARGET, GENRE_COLUMN]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required EDA columns: {missing}")

    selected_columns = required + ([ID_COLUMN] if ID_COLUMN in data.columns else [])
    clean = data[selected_columns].copy()
    numeric_columns = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + [POPULARITY_TARGET]
    clean[numeric_columns] = clean[numeric_columns].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=numeric_columns + [GENRE_COLUMN])
    clean = clean[clean[POPULARITY_TARGET].between(0, 100)]
    clean[GENRE_COLUMN] = clean[GENRE_COLUMN].astype(str).str.strip()
    if ID_COLUMN in clean.columns:
        clean = clean.dropna(subset=[ID_COLUMN]).drop_duplicates(subset=[ID_COLUMN], keep="first")
    return clean.reset_index(drop=True)


def popularity_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Compare overall and within-genre feature correlations with popularity.

    Pearson captures linear association; Spearman captures monotonic association.
    The within-genre columns reduce genre-level confounding by centering values or
    ranking observations inside each genre before calculating the correlation.
    """
    continuous = data[CONTINUOUS_FEATURES + [POPULARITY_TARGET]].copy()
    overall_pearson = continuous.corr(method="pearson")[POPULARITY_TARGET].drop(POPULARITY_TARGET)
    overall_spearman = continuous.corr(method="spearman")[POPULARITY_TARGET].drop(POPULARITY_TARGET)

    centered = continuous - data.groupby(GENRE_COLUMN)[continuous.columns].transform("mean")
    within_genre_pearson = centered.corr(method="pearson")[POPULARITY_TARGET].drop(POPULARITY_TARGET)

    ranked = data.groupby(GENRE_COLUMN)[continuous.columns].rank(method="average", pct=True)
    ranked_centered = ranked - ranked.groupby(data[GENRE_COLUMN]).transform("mean")
    within_genre_spearman = ranked_centered.corr(method="pearson")[POPULARITY_TARGET].drop(POPULARITY_TARGET)

    result = pd.DataFrame(
        {
            "pearson": overall_pearson,
            "spearman": overall_spearman,
            "within_genre_pearson": within_genre_pearson,
            "within_genre_spearman": within_genre_spearman,
        }
    )
    result.index.name = "feature"
    result = result.reset_index()
    result["absolute_within_genre_spearman"] = result["within_genre_spearman"].abs()
    return result.sort_values("absolute_within_genre_spearman", ascending=False).reset_index(drop=True)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Calculate eta-squared for a categorical feature and numeric target."""
    frame = pd.DataFrame({"category": categories, "value": values}).dropna()
    if frame.empty:
        return float("nan")

    overall_mean = frame["value"].mean()
    group_stats = frame.groupby("category")["value"].agg(["count", "mean"])
    between_group_variation = (group_stats["count"] * (group_stats["mean"] - overall_mean) ** 2).sum()
    total_variation = ((frame["value"] - overall_mean) ** 2).sum()
    return float(between_group_variation / total_variation) if total_variation else 0.0


def categorical_popularity_effects(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize popularity differences for discrete composition attributes."""
    rows = []
    for feature in CATEGORICAL_FEATURES:
        rows.append(
            {
                "feature": feature,
                "category_count": int(data[feature].nunique()),
                "eta_squared": correlation_ratio(data[feature], data[POPULARITY_TARGET]),
            }
        )
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)


def binned_popularity_trend(data: pd.DataFrame, feature: str, bins: int = 10) -> pd.DataFrame:
    """Summarize genre-adjusted popularity across feature quantile bins."""
    if feature not in CONTINUOUS_FEATURES:
        raise ValueError(f"Unknown continuous feature: {feature}")

    work = data[[feature, POPULARITY_TARGET, GENRE_COLUMN]].dropna().copy()
    genre_baseline = work.groupby(GENRE_COLUMN)[POPULARITY_TARGET].transform("mean")
    work["genre_adjusted_popularity"] = (
        work[POPULARITY_TARGET] - genre_baseline + work[POPULARITY_TARGET].mean()
    )
    work["feature_bin"] = pd.qcut(work[feature], q=bins, duplicates="drop")
    trend = (
        work.groupby("feature_bin", observed=True)
        .agg(
            feature_median=(feature, "median"),
            popularity_mean=("genre_adjusted_popularity", "mean"),
            popularity_median=(POPULARITY_TARGET, "median"),
            track_count=(POPULARITY_TARGET, "size"),
        )
        .reset_index(drop=True)
    )
    trend.insert(0, "feature", feature)
    trend.insert(1, "bin", np.arange(1, len(trend) + 1))
    return trend


def plot_popularity_correlations(correlations: pd.DataFrame, output_path: Path | None = None):
    """Plot overall and within-genre Spearman correlations side by side."""
    plot_data = correlations.melt(
        id_vars="feature",
        value_vars=["spearman", "within_genre_spearman"],
        var_name="scope",
        value_name="correlation",
    )
    plot_data["scope"] = plot_data["scope"].map(
        {"spearman": "Overall", "within_genre_spearman": "Within genre"}
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=plot_data,
        y="feature",
        x="correlation",
        hue="scope",
        palette={"Overall": "#d96f55", "Within genre": "#18864b"},
        ax=axis,
    )
    axis.axvline(0, color="#5f6862", linewidth=1)
    axis.set_title("Song-feature association with popularity")
    axis.set_xlabel("Spearman correlation")
    axis.set_ylabel(None)
    axis.legend(title=None)
    sns.despine(ax=axis)
    figure.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    return figure


def plot_top_feature_trends(
    data: pd.DataFrame,
    correlations: pd.DataFrame,
    top_n: int = 6,
    output_path: Path | None = None,
):
    """Plot decile trends for the strongest within-genre associations."""
    selected_features = correlations.head(top_n)["feature"].tolist()
    trend_data = pd.concat([binned_popularity_trend(data, feature) for feature in selected_features])

    grid = sns.relplot(
        data=trend_data,
        x="feature_median",
        y="popularity_mean",
        col="feature",
        col_wrap=3,
        kind="line",
        marker="o",
        color="#18864b",
        facet_kws={"sharex": False, "sharey": True},
        height=3.2,
        aspect=1.15,
    )
    grid.set_axis_labels("Feature value (bin median)", "Genre-adjusted mean popularity")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Genre-adjusted popularity trend across feature quantiles", y=1.03)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    return grid.figure


def generate_popularity_report(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate reproducible correlation tables and figures under reports/."""
    data = prepare_popularity_analysis_data(data)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    correlations = popularity_correlations(data)
    categorical_effects = categorical_popularity_effects(data)
    correlations.to_csv(TABLE_DIR / "popularity_feature_correlations.csv", index=False)
    categorical_effects.to_csv(TABLE_DIR / "popularity_categorical_effects.csv", index=False)

    plot_popularity_correlations(correlations, FIGURE_DIR / "popularity_correlations.png")
    plot_top_feature_trends(data, correlations, output_path=FIGURE_DIR / "popularity_feature_trends.png")
    plt.close("all")
    return correlations, categorical_effects


def main() -> None:
    """Run the focused popularity EDA report from the command line."""
    sns.set_theme(style="whitegrid", context="notebook")
    data = load_eda_data()
    correlations, categorical_effects = generate_popularity_report(data)

    print(f"Analyzed {len(data):,} rows across {data[GENRE_COLUMN].nunique():,} genres.")
    print("\nContinuous features ranked by within-genre Spearman correlation:")
    print(
        correlations[["feature", "spearman", "within_genre_spearman"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nCategorical feature effect sizes (eta-squared):")
    print(categorical_effects.round(4).to_string(index=False))
    print(f"\nSaved report outputs to {REPORT_DIR}")


if __name__ == "__main__":
    main()
