from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spotify_cleaned_eda.csv"
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
BASE_POPULARITY_MODEL_NAME = "HistGB_tuned"

NUMERIC_FEATURES = [
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
    "explicit_int",
    "key",
    "mode",
    "time_signature",
]

RECOMMENDER_FEATURES = [
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
    "explicit_int",
]

POPULARITY_FEATURES = NUMERIC_FEATURES + ["track_genre"]
POPULARITY_BANDS = [
    {"label": "Low", "min": 0, "max": 34},
    {"label": "Moderate", "min": 35, "max": 59},
    {"label": "High", "min": 60, "max": 100},
]
DISPLAY_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "duration_min",
]


def load_modeling_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(DISPLAY_COLUMNS + NUMERIC_FEATURES + ["track_genre"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=["track_id", "track_name", "artists", "track_genre"])
    for col in NUMERIC_FEATURES + ["popularity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES + ["popularity"])
    df = df[np.isfinite(df[NUMERIC_FEATURES + ["popularity"]]).all(axis=1)]
    df = df[(df["popularity"] >= 0) & (df["popularity"] <= 100)]
    df = df[df["duration_min"] > 0]
    df = df[df["tempo"] > 0]
    df["track_genre"] = df["track_genre"].astype(str).str.strip()
    df["artists"] = df["artists"].astype(str).str.strip()
    df["track_name"] = df["track_name"].astype(str).str.strip()
    df["album_name"] = df["album_name"].fillna("Unknown").astype(str).str.strip()
    return df.reset_index(drop=True)


def build_recommender(df: pd.DataFrame) -> tuple[StandardScaler, NearestNeighbors, pd.DataFrame]:
    catalog_columns = list(dict.fromkeys(DISPLAY_COLUMNS + RECOMMENDER_FEATURES))
    song_catalog = df[catalog_columns].copy()
    song_catalog = song_catalog.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(song_catalog[RECOMMENDER_FEATURES])

    recommender = NearestNeighbors(
        n_neighbors=50,
        metric="cosine",
        algorithm="brute",
    )
    recommender.fit(features_scaled)
    return scaler, recommender, song_catalog


def build_popularity_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("genre", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["track_genre"]),
        ],
        remainder="drop",
    )


def get_popularity_model_candidates() -> dict[str, object]:
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0),
        "HistGB_current": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.05,
            l2_regularization=0.05,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "HistGB_tuned": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.04,
            max_leaf_nodes=63,
            l2_regularization=0.02,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=6,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def build_popularity_model(model_name: str = BASE_POPULARITY_MODEL_NAME) -> Pipeline:
    candidates = get_popularity_model_candidates()
    if model_name not in candidates:
        raise ValueError(f"Unknown popularity model: {model_name}. Available: {sorted(candidates)}")

    if model_name == "DummyMean":
        return Pipeline(steps=[("model", candidates[model_name])])

    return Pipeline(
        steps=[
            ("preprocess", build_popularity_preprocessor()),
            ("model", candidates[model_name]),
        ]
    )


def split_train_test_by_track(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df["popularity"], groups=df["track_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def score_regression_model(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_popularity_models(df: pd.DataFrame) -> pd.DataFrame:
    train, test = split_train_test_by_track(df)
    rows = []

    for model_name in get_popularity_model_candidates():
        model = build_popularity_model(model_name)
        if model_name == "DummyMean":
            model.fit(train[["popularity"]], train["popularity"])
            predictions = model.predict(test[["popularity"]])
        else:
            model.fit(train[POPULARITY_FEATURES], train["popularity"])
            predictions = model.predict(test[POPULARITY_FEATURES])

        metrics = score_regression_model(test["popularity"], predictions)
        rows.append(
            {
                "model": model_name,
                **metrics,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "unique_train_tracks": int(train["track_id"].nunique()),
                "unique_test_tracks": int(test["track_id"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)


def popularity_band(score: float) -> str:
    clipped_score = float(np.clip(score, 0, 100))
    for band in POPULARITY_BANDS:
        if band["min"] <= clipped_score <= band["max"]:
            return band["label"]
    return "Unknown"


def evaluate_popularity_model(
    df: pd.DataFrame,
    model_name: str = BASE_POPULARITY_MODEL_NAME,
) -> tuple[Pipeline, dict[str, float]]:
    train, test = split_train_test_by_track(df)
    model = build_popularity_model(model_name)
    model.fit(train[POPULARITY_FEATURES], train["popularity"])
    predictions = model.predict(test[POPULARITY_FEATURES])
    metrics = {
        "model": model_name,
        **score_regression_model(test["popularity"], predictions),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "unique_train_tracks": int(train["track_id"].nunique()),
        "unique_test_tracks": int(test["track_id"].nunique()),
    }
    return model, metrics


def train_final_popularity_model(
    df: pd.DataFrame,
    model_name: str = BASE_POPULARITY_MODEL_NAME,
) -> Pipeline:
    model = build_popularity_model(model_name)
    model.fit(df[POPULARITY_FEATURES], df["popularity"])
    return model


def predict_base_popularity(input_profile: dict, model: Pipeline) -> tuple[float, str]:
    input_df = pd.DataFrame([input_profile])
    prediction = float(np.clip(model.predict(input_df[POPULARITY_FEATURES])[0], 0, 100))
    return prediction, popularity_band(prediction)


def build_model_metadata(
    df: pd.DataFrame,
    song_catalog: pd.DataFrame,
    artist_profiles: pd.DataFrame,
    model_comparison: pd.DataFrame,
    selected_model_name: str,
) -> dict:
    selected_metrics = (
        model_comparison[model_comparison["model"] == selected_model_name]
        .iloc[0]
        .to_dict()
    )
    return {
        "data_path": str(DATA_PATH),
        "model_dir": str(MODEL_DIR),
        "row_count_after_cleaning": int(len(df)),
        "song_catalog_rows": int(len(song_catalog)),
        "artist_profile_rows": int(len(artist_profiles)),
        "selected_base_popularity_model": selected_model_name,
        "selected_base_popularity_metrics": selected_metrics,
        "model_comparison": model_comparison.to_dict(orient="records"),
        "popularity_bands": POPULARITY_BANDS,
        "numeric_features": NUMERIC_FEATURES,
        "recommender_features": RECOMMENDER_FEATURES,
        "popularity_features": POPULARITY_FEATURES,
        "notes": [
            "Base popularity model excludes artist, album, track name, and track id.",
            "Selected model is tuned gradient boosting because it performed best in the comparison split.",
            "Artist-assisted popularity uses historical artist residual lift and style similarity.",
            "Evaluation uses GroupShuffleSplit by track_id to reduce duplicate-song leakage.",
            "Popularity is a social/commercial outcome, so moderate R2 is expected with audio features only.",
        ],
    }


def split_artist_names(artists: str) -> list[str]:
    return [name.strip() for name in str(artists).split(";") if name.strip()]


def build_artist_profiles(df: pd.DataFrame, base_model: Pipeline, min_tracks: int = 5) -> pd.DataFrame:
    work = df.copy()
    work["base_popularity_prediction"] = base_model.predict(work[POPULARITY_FEATURES])
    work["artist_lift"] = work["popularity"] - work["base_popularity_prediction"]
    work["artist"] = work["artists"].map(split_artist_names)
    work = work.explode("artist")
    work = work.dropna(subset=["artist"])
    work["artist"] = work["artist"].astype(str).str.strip()
    work = work[work["artist"] != ""]

    grouped = work.groupby("artist")
    profiles = grouped.agg(
        track_count=("track_id", "nunique"),
        avg_popularity=("popularity", "mean"),
        artist_lift=("artist_lift", "mean"),
        main_genre=("track_genre", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        genre_count=("track_genre", "nunique"),
        **{feature: (feature, "mean") for feature in RECOMMENDER_FEATURES},
    ).reset_index()

    profiles = profiles[profiles["track_count"] >= min_tracks].copy()
    shrinkage = profiles["track_count"] / (profiles["track_count"] + 10)
    profiles["artist_lift_shrunk"] = profiles["artist_lift"] * shrinkage
    profiles = profiles.sort_values(
        ["artist_lift_shrunk", "avg_popularity", "track_count"],
        ascending=False,
    ).reset_index(drop=True)
    return profiles

def deduplicate_song_results(results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    results = results.copy()

    results["song_artist_key"] = (
        results["track_name"].astype(str).str.lower().str.strip()
        + " - "
        + results["artists"].astype(str).str.lower().str.strip()
    )
    results = (
        results.sort_values(["similarity_score", "popularity"], ascending=False)
        .drop_duplicates(subset = ["song_artist_key"], keep="first")
        .drop(columns=["song_artist_key"])
        .head(top_n)
    )
    return results

def recommend_songs(
    input_profile: dict,
    scaler: StandardScaler,
    recommender: NearestNeighbors,
    song_catalog: pd.DataFrame,
    top_n: int = 10,
    genre: str | None = None,
) -> pd.DataFrame:
    input_row = pd.DataFrame([{feature: input_profile[feature] for feature in RECOMMENDER_FEATURES}])
    input_scaled = scaler.transform(input_row)

    if genre:
        genre_catalog = song_catalog[song_catalog["track_genre"].str.lower() == genre.lower()].copy()
        if not genre_catalog.empty:
            genre_scaled = scaler.transform(genre_catalog[RECOMMENDER_FEATURES])
            input_norm = np.linalg.norm(input_scaled, axis=1)[0]
            genre_norm = np.linalg.norm(genre_scaled, axis=1)
            similarity = (genre_scaled @ input_scaled.T).ravel() / np.maximum(genre_norm * input_norm, 1e-9)
            genre_catalog["similarity_score"] = np.clip(similarity, -1, 1)
            return deduplicate_song_results(genre_catalog, top_n)

    distances, indices = recommender.kneighbors(input_scaled, n_neighbors=min(200, len(song_catalog)))
    candidates = song_catalog.iloc[indices[0]].copy()
    candidates["similarity_score"] = 1 - distances[0]

    return deduplicate_song_results(candidates, top_n)


def recommend_artists(
    input_profile: dict,
    artist_profiles: pd.DataFrame,
    base_popularity_prediction: float,
    scaler: StandardScaler,
    top_n: int = 10,
    genre: str | None = None,
) -> pd.DataFrame:
    profiles = artist_profiles.copy()
    if genre:
        genre_matches = profiles[profiles["main_genre"].str.lower() == genre.lower()]
        if len(genre_matches) >= top_n:
            profiles = genre_matches

    input_row = pd.DataFrame([{feature: input_profile[feature] for feature in RECOMMENDER_FEATURES}])
    input_scaled = scaler.transform(input_row)
    profile_scaled = scaler.transform(profiles[RECOMMENDER_FEATURES])

    input_norm = np.linalg.norm(input_scaled, axis=1)[0]
    profile_norm = np.linalg.norm(profile_scaled, axis=1)
    similarity = (profile_scaled @ input_scaled.T).ravel() / np.maximum(profile_norm * input_norm, 1e-9)
    similarity = np.clip(similarity, -1, 1)
    similarity_0_1 = (similarity + 1) / 2

    result = profiles[
        ["artist", "track_count", "main_genre", "avg_popularity", "artist_lift_shrunk"]
    ].copy()
    result["style_similarity"] = similarity_0_1
    result["artist_assisted_popularity"] = np.clip(
        base_popularity_prediction + result["artist_lift_shrunk"] * result["style_similarity"],
        0,
        100,
    )
    result["ranking_score"] = (
        0.55 * result["style_similarity"] * 100
        + 0.35 * result["artist_assisted_popularity"]
        + 0.10 * result["avg_popularity"]
    )
    return result.sort_values("ranking_score", ascending=False).head(top_n)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_modeling_data()
    recommender_scaler, recommender, song_catalog = build_recommender(df)

    model_comparison = evaluate_popularity_models(df)
    selected_model_name = BASE_POPULARITY_MODEL_NAME
    final_popularity_model = train_final_popularity_model(df, selected_model_name)

    artist_profiles = build_artist_profiles(df, final_popularity_model)

    joblib.dump(recommender_scaler, MODEL_DIR / "spotify_recommender_scaler.joblib")
    joblib.dump(recommender, MODEL_DIR / "spotify_nearest_neighbors.joblib")
    joblib.dump(final_popularity_model, MODEL_DIR / "spotify_base_popularity_model.joblib")
    song_catalog.to_csv(MODEL_DIR / "spotify_song_catalog.csv", index=False)
    artist_profiles.to_csv(MODEL_DIR / "spotify_artist_profiles.csv", index=False)
    model_comparison.to_csv(MODEL_DIR / "spotify_popularity_model_comparison.csv", index=False)

    metadata = build_model_metadata(
        df=df,
        song_catalog=song_catalog,
        artist_profiles=artist_profiles,
        model_comparison=model_comparison,
        selected_model_name=selected_model_name,
    )
    (MODEL_DIR / "spotify_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    example = {
        "danceability": 0.70,
        "energy": 0.75,
        "loudness": -6.5,
        "speechiness": 0.06,
        "acousticness": 0.18,
        "instrumentalness": 0.00,
        "liveness": 0.12,
        "valence": 0.62,
        "tempo": 120.0,
        "duration_min": 3.4,
        "explicit_int": 0,
        "key": 1,
        "mode": 1,
        "time_signature": 4,
        "track_genre": "pop",
    }
    base_prediction, base_band = predict_base_popularity(example, final_popularity_model)
    similar = recommend_songs(example, recommender_scaler, recommender, song_catalog, top_n=5, genre="pop")
    artists = recommend_artists(example, artist_profiles, base_prediction, recommender_scaler, top_n=5, genre="pop")

    print("Training complete.")
    print("Model comparison:")
    print(model_comparison[["model", "mae", "rmse", "r2"]].round(4).to_string(index=False))
    print(f"Selected base popularity model: {selected_model_name}")
    print(f"Example base popularity prediction: {base_prediction:.2f} / 100 ({base_band})")
    print("Example similar songs:")
    print(similar[["track_name", "artists", "track_genre", "popularity", "similarity_score"]].to_string(index=False))
    print("Example artist recommendations:")
    print(artists.to_string(index=False))


if __name__ == "__main__":
    main()
