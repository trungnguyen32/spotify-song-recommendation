## LOADING MODELS AND DATA
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

from spotify_train_models import (
    POPULARITY_FEATURES,
    deduplicate_song_results,
    recommend_artists,
    recommend_songs,
    predict_base_popularity,
)

@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / "spotify_recommender_scaler.joblib")
    recommender = joblib.load(MODEL_DIR / "spotify_nearest_neighbors.joblib")
    popularity_model = joblib.load(MODEL_DIR / "spotify_base_popularity_model.joblib")
    return scaler, recommender, popularity_model

@st.cache_data
def load_tables():
    song_catalog = pd.read_csv(MODEL_DIR / "spotify_song_catalog.csv")
    artist_profiles = pd.read_csv(MODEL_DIR / "spotify_artist_profiles.csv")
    return song_catalog, artist_profiles

## ADD USER INPUTS
st.set_page_config(page_title="Composer Song Recommender", layout="wide")

st.title("Composer Song Recommender")
st.caption("Find similar songs, estimate fame-free popularity, and identify suitable artists.")

scaler, recommender, popularity_model = load_models()
song_catalog, artist_profiles = load_tables()

genres = sorted(song_catalog["track_genre"].dropna().unique())

with st.sidebar:
    st.header("Song Inputs")

    genre = st.selectbox("Genre", genres)

    st.caption("Enter exact audio-feature values if you already have a proposed song profile.")

    feature_col_1, feature_col_2 = st.columns(2)

    with feature_col_1:
        danceability = st.number_input("Danceability", 0.0, 1.0, 0.70, 0.01, format="%.3f")
        energy = st.number_input("Energy", 0.0, 1.0, 0.75, 0.01, format="%.3f")
        valence = st.number_input("Valence", 0.0, 1.0, 0.60, 0.01, format="%.3f")
        acousticness = st.number_input("Acousticness", 0.0, 1.0, 0.20, 0.01, format="%.3f")
        instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, 0.00, 0.01, format="%.3f")

    with feature_col_2:
        speechiness = st.number_input("Speechiness", 0.0, 1.0, 0.06, 0.01, format="%.3f")
        liveness = st.number_input("Liveness", 0.0, 1.0, 0.12, 0.01, format="%.3f")
        tempo = st.number_input("Tempo", 40.0, 220.0, 120.0, 1.0, format="%.1f")
        loudness = st.number_input("Loudness", -40.0, 5.0, -6.5, 0.1, format="%.2f")
        duration_min = st.number_input("Duration (min)", 1.0, 8.0, 3.4, 0.1, format="%.2f")

    explicit_int = st.selectbox("Explicit", [0, 1], format_func=lambda x: "Yes" if x else "No")
    key = st.number_input("Key", 0, 11, 1, 1)
    mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
    time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)

    ## BUILD INPUT DICTIONARY
    composer_input = {
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "duration_min": duration_min,
    "explicit_int": explicit_int,
    "key": key,
    "mode": mode,
    "time_signature": time_signature,
    "track_genre": genre,
}

## POPULARITY MODEL PREDICTION
base_score, popularity_band = predict_base_popularity(composer_input, popularity_model)

col1, col2 = st.columns(2)

with col1:
    st.metric("Fame-Free Popularity Estimate", f"{base_score:.1f} / 100")

with col2:
    st.metric("Popularity Band", popularity_band)

## SIMLAR SONG RECOMMENDATIONS
similar_songs = recommend_songs(
    input_profile=composer_input,
    scaler=scaler,
    recommender=recommender,
    song_catalog=song_catalog,
    top_n=50,
    genre=genre,
)
similar_songs = deduplicate_song_results(similar_songs, 10)

st.subheader("Most Similar Existing Songs")

st.dataframe(
    similar_songs[
        [
            "track_name",
            "artists",
            "track_genre",
            "popularity",
            "similarity_score",
            "danceability",
            "energy",
            "valence",
            "tempo",
        ]
    ],
    use_container_width=True,
)

## ARTIST RECOMMENDATIONS
artist_recommendations = recommend_artists(
    input_profile=composer_input,
    artist_profiles=artist_profiles,
    base_popularity_prediction=base_score,
    scaler=scaler,
    top_n=10,
    genre=genre,
)

st.subheader("Recommended Artists To Pitch To")

st.dataframe(
    artist_recommendations[
        [
            "artist",
            "main_genre",
            "track_count",
            "avg_popularity",
            "style_similarity",
            "artist_assisted_popularity",
            "ranking_score",
        ]
    ],
    use_container_width=True,
)

## Add radar chart comparing input profile to top recommended song
radar_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",]

input_values = [composer_input[feature] for feature in radar_features]
similar_avg_values = [similar_songs[feature].mean() for feature in radar_features]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r = input_values,
    theta = radar_features,
    fill='toself',
    name='Your Song Input',
    )
)

fig.add_trace(go.Scatterpolar(
    r = similar_avg_values,
    theta = radar_features,
    fill='toself',
    name='Average of Similar Songs',
    )
)

fig.update_layout(
    polar = dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend = True,
    title="Comparison of Your Song Input to Similar Songs",
    height = 420,
)
st.plotly_chart(fig, use_container_width=True)
