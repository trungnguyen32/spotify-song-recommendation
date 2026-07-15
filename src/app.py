"""Streamlit dashboard for composition similarity and popularity analysis."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spotify_train_models import (
    RECOMMENDER_FEATURES,
    deduplicate_song_results,
    predict_base_popularity,
    recommend_artists,
    recommend_songs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

PROFILE_FIELDS = [
    ("danceability", "Danceability", "0.700", 0.0, 1.0, "How suitable the track is for dancing."),
    ("energy", "Energy", "0.750", 0.0, 1.0, "Perceived intensity and activity."),
    ("valence", "Valence", "0.600", 0.0, 1.0, "Musical positivity, from darker to brighter."),
    ("acousticness", "Acousticness", "0.200", 0.0, 1.0, "Confidence that the track is acoustic."),
    ("instrumentalness", "Instrumentalness", "0.000", 0.0, 1.0, "Likelihood that the track has no vocals."),
    ("speechiness", "Speechiness", "0.060", 0.0, 1.0, "Presence of spoken words."),
    ("liveness", "Liveness", "0.120", 0.0, 1.0, "Likelihood of a live audience or performance."),
    ("tempo", "Tempo (BPM)", "120.0", 40.0, 220.0, "Estimated tempo in beats per minute."),
    ("loudness", "Loudness (dB)", "-6.50", -40.0, 5.0, "Average track loudness in decibels."),
    ("duration_min", "Duration (minutes)", "3.40", 1.0, 8.0, "Proposed track duration in minutes."),
]

FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "loudness": "Loudness",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumental",
    "liveness": "Liveness",
    "valence": "Valence",
    "tempo": "Tempo",
    "duration_min": "Duration",
    "explicit_int": "Explicit",
}

RADAR_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


st.set_page_config(
    page_title="Spotify Composer Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource
def load_models():
    """Load fitted estimators once per Streamlit process."""
    scaler = joblib.load(MODEL_DIR / "spotify_recommender_scaler.joblib")
    recommender = joblib.load(MODEL_DIR / "spotify_nearest_neighbors.joblib")
    popularity_model = joblib.load(MODEL_DIR / "spotify_base_popularity_model.joblib")
    return scaler, recommender, popularity_model


@st.cache_data
def load_tables_and_metadata():
    """Load dashboard data and model documentation."""
    song_catalog = pd.read_csv(MODEL_DIR / "spotify_song_catalog.csv")
    artist_profiles = pd.read_csv(MODEL_DIR / "spotify_artist_profiles.csv")
    metadata = json.loads((MODEL_DIR / "spotify_model_metadata.json").read_text(encoding="utf-8"))
    return song_catalog, artist_profiles, metadata


def apply_dashboard_styles() -> None:
    """Apply a restrained studio-workbench visual system."""
    st.markdown(
        """
        <style>
        :root {
            --studio-canvas: #f4f6f2;
            --studio-panel: #ffffff;
            --studio-inset: #edf0eb;
            --studio-ink: #171b18;
            --studio-muted: #5f6862;
            --studio-line: rgba(23, 27, 24, 0.10);
            --signal-green: #18864b;
            --reference-coral: #d96f55;
        }
        [data-testid="stAppViewContainer"] { background: var(--studio-canvas); }
        [data-testid="stSidebar"] { background: var(--studio-inset); }
        [data-testid="stHeader"] { background: rgba(244, 246, 242, 0.92); }
        .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--studio-ink); letter-spacing: 0; }
        h1 { font-size: 2rem; font-weight: 700; }
        h2 { font-size: 1.35rem; font-weight: 650; }
        h3 { font-size: 1rem; font-weight: 650; }
        p, label, [data-testid="stCaptionContainer"] { color: var(--studio-muted); }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        [data-testid="stMetricLabel"] { color: var(--studio-muted); }
        [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--studio-line);
            border-radius: 8px;
        }
        [data-baseweb="input"] { background: #e7ebe5; }
        [data-baseweb="tab-list"] { gap: 1rem; }
        [data-baseweb="tab"] { min-height: 44px; }
        .model-kicker {
            color: var(--signal-green);
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .interpretation-copy {
            color: var(--studio-ink);
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .score-track {
            width: 100%;
            height: 8px;
            margin: 1rem 0;
            overflow: hidden;
            background: #e5e9e3;
            border-radius: 4px;
        }
        .score-fill {
            height: 100%;
            background: var(--signal-green);
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_decimal_inputs(raw_values: dict[str, str]) -> tuple[dict[str, float], list[str]]:
    """Parse dot-decimal text fields and return all validation errors together."""
    values: dict[str, float] = {}
    errors: list[str] = []
    field_specs = {field[0]: field for field in PROFILE_FIELDS}

    for name, raw_value in raw_values.items():
        _, label, _, minimum, maximum, _ = field_specs[name]
        candidate = raw_value.strip()
        if "," in candidate:
            errors.append(f"{label}: use a period for decimals, for example 0.75.")
            continue
        try:
            value = float(candidate)
        except ValueError:
            errors.append(f"{label}: enter a numeric value.")
            continue
        if not math.isfinite(value) or not minimum <= value <= maximum:
            errors.append(f"{label}: enter a value from {minimum:g} to {maximum:g}.")
            continue
        values[name] = value

    return values, errors


def collect_song_profile(genres: list[str]) -> dict:
    """Render the sidebar form and return a validated composer profile."""
    with st.sidebar:
        st.markdown("### Song profile")
        st.caption("Enter proposed values with a period as the decimal separator.")

        with st.form("song-profile"):
            genre = st.selectbox("Target genre", genres)
            raw_values: dict[str, str] = {}

            st.markdown("#### Character")
            for name, label, default, _, _, help_text in PROFILE_FIELDS[:3]:
                raw_values[name] = st.text_input(label, value=default, help=help_text)

            st.markdown("#### Texture")
            for name, label, default, _, _, help_text in PROFILE_FIELDS[3:7]:
                raw_values[name] = st.text_input(label, value=default, help=help_text)

            st.markdown("#### Structure")
            for name, label, default, _, _, help_text in PROFILE_FIELDS[7:]:
                raw_values[name] = st.text_input(label, value=default, help=help_text)

            structure_columns = st.columns(2)
            with structure_columns[0]:
                key = st.selectbox("Key", range(12), index=1, help="Pitch-class integer from 0 to 11.")
                mode = st.selectbox("Mode", [1, 0], format_func=lambda value: "Major" if value else "Minor")
            with structure_columns[1]:
                time_signature = st.selectbox("Meter", [3, 4, 5], index=1)
                explicit_int = st.selectbox("Explicit", [0, 1], format_func=lambda value: "Yes" if value else "No")

            st.form_submit_button("Analyze song profile", width="stretch", type="primary")

    numeric_values, errors = parse_decimal_inputs(raw_values)
    if errors:
        with st.sidebar:
            for error in errors:
                st.error(error)
        st.stop()

    return {
        **numeric_values,
        "explicit_int": explicit_int,
        "key": key,
        "mode": mode,
        "time_signature": time_signature,
        "track_genre": genre,
    }


def evidence_support(mean_similarity: float, genre_track_count: int) -> tuple[str, str]:
    """Describe neighborhood support without presenting it as statistical confidence."""
    if mean_similarity >= 0.90 and genre_track_count >= 100:
        return "Strong", "The catalog contains many close same-genre reference tracks."
    if mean_similarity >= 0.78 and genre_track_count >= 30:
        return "Moderate", "The profile has usable same-genre references, with some feature distance."
    return "Limited", "Few close same-genre examples support this profile; interpret cautiously."


def build_radar_chart(input_profile: dict, similar_songs: pd.DataFrame) -> go.Figure:
    """Compare bounded audio features with the similar-song benchmark."""
    labels = [FEATURE_LABELS[feature] for feature in RADAR_FEATURES]
    input_values = [input_profile[feature] for feature in RADAR_FEATURES]
    benchmark_values = [similar_songs[feature].mean() for feature in RADAR_FEATURES]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=input_values + [input_values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(24, 134, 75, 0.16)",
            line={"color": "#18864b", "width": 3},
            name="Proposed song",
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=benchmark_values + [benchmark_values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(217, 111, 85, 0.10)",
            line={"color": "#d96f55", "width": 2, "dash": "dot"},
            name="Similar-song average",
        )
    )
    figure.update_layout(
        height=470,
        margin={"l": 60, "r": 60, "t": 35, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        polar={
            "bgcolor": "rgba(255,255,255,0)",
            "radialaxis": {"range": [0, 1], "tickvals": [0.25, 0.5, 0.75, 1.0], "gridcolor": "#d9ded8"},
            "angularaxis": {"gridcolor": "#d9ded8"},
        },
        legend={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
    )
    return figure


def build_difference_table(input_profile: dict, similar_songs: pd.DataFrame, scaler) -> pd.DataFrame:
    """Return raw and standardized differences from the recommendation benchmark."""
    features = [feature for feature in RECOMMENDER_FEATURES if feature != "explicit_int"]
    feature_positions = {feature: index for index, feature in enumerate(scaler.feature_names_in_)}
    rows = []
    for feature in features:
        benchmark = float(similar_songs[feature].mean())
        difference = float(input_profile[feature] - benchmark)
        scale = float(scaler.scale_[feature_positions[feature]])
        rows.append(
            {
                "feature": FEATURE_LABELS[feature],
                "proposed": float(input_profile[feature]),
                "benchmark": benchmark,
                "difference": difference,
                "standardized_difference": difference / scale if scale else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_difference")


def build_difference_chart(differences: pd.DataFrame) -> go.Figure:
    """Show feature gaps in standard-deviation units for comparable magnitudes."""
    colors = ["#d96f55" if value < 0 else "#18864b" for value in differences["standardized_difference"]]
    figure = go.Figure(
        go.Bar(
            x=differences["standardized_difference"],
            y=differences["feature"],
            orientation="h",
            marker_color=colors,
            customdata=differences[["difference"]],
            hovertemplate="%{y}<br>Standardized gap: %{x:.2f}<br>Raw gap: %{customdata[0]:.3f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#69726c", line_width=1)
    figure.update_layout(
        height=470,
        margin={"l": 20, "r": 20, "t": 35, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Difference from benchmark (standard deviations)",
        yaxis_title=None,
        xaxis={"gridcolor": "#e2e6e0", "zeroline": False},
    )
    return figure


def render_score_summary(
    base_score: float,
    popularity_band: str,
    support_label: str,
    support_text: str,
    mae: float,
) -> None:
    """Render the prediction with nearby interpretation and uncertainty context."""
    score_column, interpretation_column = st.columns([1.45, 1], gap="large")
    lower = max(0.0, base_score - mae)
    upper = min(100.0, base_score + mae)

    with score_column:
        with st.container(border=True):
            st.markdown('<div class="model-kicker">COMPOSITION-ONLY ESTIMATE</div>', unsafe_allow_html=True)
            metric_columns = st.columns(3)
            metric_columns[0].metric("Base popularity", f"{base_score:.1f} / 100")
            metric_columns[1].metric("Popularity band", popularity_band)
            metric_columns[2].metric("Reference support", support_label, help=support_text)
            st.markdown(
                f'<div class="score-track" role="img" aria-label="Popularity score {base_score:.1f} out of 100">'
                f'<div class="score-fill" style="width: {base_score:.1f}%"></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"A practical reading is roughly {lower:.0f}-{upper:.0f}, using the model's {mae:.1f}-point "
                "holdout MAE as context. This is not a calibrated prediction interval."
            )

    with interpretation_column:
        with st.container(border=True):
            st.markdown("### What this score means")
            st.markdown(
                '<p class="interpretation-copy">The estimate reflects composition and song attributes only. '
                "Artist identity, playlist placement, marketing, release timing, and social momentum are "
                "intentionally absent.</p>",
                unsafe_allow_html=True,
            )
            st.info("Treat the score as a comparison tool for song profiles, not a forecast of virality or commercial success.")


def render_similar_songs(similar_songs: pd.DataFrame, genre: str) -> None:
    """Render ranked, same-genre song references with reader-friendly labels."""
    st.subheader("Closest composition references")
    st.caption(f"Ranked by audio-feature similarity within the {genre} catalog; duplicate song-artist pairs are removed.")
    display = similar_songs.copy().reset_index(drop=True)
    display.insert(0, "rank", display.index + 1)
    display = display.rename(
        columns={
            "rank": "Rank",
            "track_name": "Track",
            "artists": "Artist",
            "popularity": "Popularity",
            "similarity_score": "Similarity",
            "danceability": "Danceability",
            "energy": "Energy",
            "valence": "Valence",
            "tempo": "Tempo",
        }
    )
    display["Similarity (%)"] = display["Similarity"] * 100
    st.dataframe(
        display[["Rank", "Track", "Artist", "Popularity", "Similarity (%)"]],
        hide_index=True,
        width="stretch",
        height=430,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="%d"),
            "Track": st.column_config.TextColumn(width="medium"),
            "Artist": st.column_config.TextColumn(width="medium"),
            "Popularity": st.column_config.NumberColumn(format="%.0f"),
            "Similarity (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    with st.expander("View recommendation feature values"):
        st.dataframe(
            display[["Rank", "Track", "Danceability", "Energy", "Valence", "Tempo"]],
            hide_index=True,
            width="stretch",
            column_config={
                "Danceability": st.column_config.NumberColumn(format="%.3f"),
                "Energy": st.column_config.NumberColumn(format="%.3f"),
                "Valence": st.column_config.NumberColumn(format="%.3f"),
                "Tempo": st.column_config.NumberColumn(format="%.1f BPM"),
            },
        )


def render_artist_fit(artist_recommendations: pd.DataFrame, base_score: float) -> None:
    """Render artist style matches and the residual-lift scenario estimate."""
    st.subheader("Artists whose catalog fits this profile")
    st.caption(
        "Fit combines style similarity with each artist's historical popularity above or below the composition-only model. "
        f"The assisted score starts from the {base_score:.1f} base estimate and is a scenario, not a guarantee."
    )
    display = artist_recommendations.copy().reset_index(drop=True)
    display.insert(0, "rank", display.index + 1)
    display = display.rename(
        columns={
            "rank": "Rank",
            "artist": "Artist",
            "main_genre": "Catalog genre",
            "track_count": "Tracks observed",
            "avg_popularity": "Historical average",
            "style_similarity": "Style fit",
            "artist_assisted_popularity": "Assisted scenario",
            "ranking_score": "Fit score",
        }
    )
    display["Style fit (%)"] = display["Style fit"] * 100
    st.dataframe(
        display[["Rank", "Artist", "Catalog genre", "Tracks observed", "Style fit (%)", "Historical average", "Assisted scenario", "Fit score"]],
        hide_index=True,
        width="stretch",
        height=430,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="%d"),
            "Artist": st.column_config.TextColumn(width="large"),
            "Tracks observed": st.column_config.NumberColumn(format="%d"),
            "Style fit (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Historical average": st.column_config.NumberColumn(format="%.1f"),
            "Assisted scenario": st.column_config.NumberColumn(format="%.1f"),
            "Fit score": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_feature_fingerprint(input_profile: dict, similar_songs: pd.DataFrame, scaler) -> None:
    """Render shape and magnitude views of the proposed song profile."""
    st.subheader("Feature fingerprint")
    st.caption("The radar shows the seven bounded audio features; the difference chart makes mixed-unit gaps comparable.")
    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        st.plotly_chart(build_radar_chart(input_profile, similar_songs), width="stretch", config=CHART_CONFIG)
    differences = build_difference_table(input_profile, similar_songs, scaler)
    with chart_columns[1]:
        st.plotly_chart(build_difference_chart(differences), width="stretch", config=CHART_CONFIG)

    with st.expander("View exact feature values"):
        exact_values = differences.sort_values("feature").rename(
            columns={
                "feature": "Feature",
                "proposed": "Proposed song",
                "benchmark": "Similar-song average",
                "difference": "Raw difference",
                "standardized_difference": "Standardized difference",
            }
        )
        st.dataframe(
            exact_values,
            hide_index=True,
            width="stretch",
            column_config={
                "Proposed song": st.column_config.NumberColumn(format="%.3f"),
                "Similar-song average": st.column_config.NumberColumn(format="%.3f"),
                "Raw difference": st.column_config.NumberColumn(format="%+.3f"),
                "Standardized difference": st.column_config.NumberColumn(format="%+.2f"),
            },
        )


def render_model_notes(metadata: dict) -> None:
    """Render concise model documentation and known limitations."""
    metrics = metadata["selected_base_popularity_metrics"]
    st.subheader("Model interpretation and limitations")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Holdout MAE", f"{metrics['mae']:.2f} points")
    metric_columns[1].metric("Holdout RMSE", f"{metrics['rmse']:.2f} points")
    metric_columns[2].metric("Holdout R-squared", f"{metrics['r2']:.2f}")

    explanation_columns = st.columns(2, gap="large")
    with explanation_columns[0]:
        st.markdown("#### Included evidence")
        st.write(
            "Audio features, duration, key, mode, time signature, explicit status, and target genre. "
            "The holdout split groups rows by track ID to reduce duplicate-song leakage."
        )
        st.markdown("#### Excluded on purpose")
        st.write("Artist, album, track name, and track ID are excluded from the base score to reduce direct fame bias.")
    with explanation_columns[1]:
        st.markdown("#### What remains unknown")
        st.write(
            "Promotion, editorial and algorithmic playlists, release timing, audience size, cultural trends, and social activity "
            "can strongly affect popularity but are unavailable here."
        )
        st.markdown("#### Appropriate use")
        st.write(
            "Compare composition choices and identify plausible references or artist fits. Do not interpret the score as causal, "
            "a probability of success, or financial advice."
        )


def main() -> None:
    """Run the dashboard."""
    apply_dashboard_styles()
    scaler, recommender, popularity_model = load_models()
    song_catalog, artist_profiles, metadata = load_tables_and_metadata()
    genres = sorted(song_catalog["track_genre"].dropna().unique().tolist())
    composer_input = collect_song_profile(genres)

    base_score, popularity_band = predict_base_popularity(composer_input, popularity_model)
    similar_songs = recommend_songs(
        input_profile=composer_input,
        scaler=scaler,
        recommender=recommender,
        song_catalog=song_catalog,
        top_n=50,
        genre=composer_input["track_genre"],
    )
    similar_songs = deduplicate_song_results(similar_songs, top_n=10)
    artist_recommendations = recommend_artists(
        input_profile=composer_input,
        artist_profiles=artist_profiles,
        base_popularity_prediction=base_score,
        scaler=scaler,
        top_n=10,
        genre=composer_input["track_genre"],
    )

    mean_similarity = float(similar_songs["similarity_score"].head(5).mean())
    genre_track_count = int((song_catalog["track_genre"] == composer_input["track_genre"]).sum())
    support_label, support_text = evidence_support(mean_similarity, genre_track_count)
    model_mae = float(metadata["selected_base_popularity_metrics"]["mae"])

    st.markdown('<div class="model-kicker">COMPOSITION WORKBENCH</div>', unsafe_allow_html=True)
    st.title("Spotify Composer Assistant")
    st.caption("Explore composition references, a fame-free popularity baseline, and artist catalog fit.")

    render_score_summary(base_score, popularity_band, support_label, support_text, model_mae)

    similar_tab, artist_tab, fingerprint_tab, notes_tab = st.tabs(
        ["Similar songs", "Artist fit", "Feature fingerprint", "Model notes"]
    )
    with similar_tab:
        render_similar_songs(similar_songs, composer_input["track_genre"])
    with artist_tab:
        render_artist_fit(artist_recommendations, base_score)
    with fingerprint_tab:
        render_feature_fingerprint(composer_input, similar_songs, scaler)
    with notes_tab:
        render_model_notes(metadata)


if __name__ == "__main__":
    main()
