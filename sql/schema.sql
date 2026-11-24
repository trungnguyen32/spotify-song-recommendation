-- Dedicated application schema. All changes participate in the loader transaction.
CREATE SCHEMA IF NOT EXISTS spotify;

CREATE TABLE IF NOT EXISTS spotify.tracks (
    track_id text PRIMARY KEY CHECK (length(btrim(track_id)) > 0),
    artists text NOT NULL CHECK (length(btrim(artists)) > 0),
    album_name text NOT NULL,
    track_name text NOT NULL CHECK (length(btrim(track_name)) > 0),
    popularity double precision NOT NULL CHECK (popularity BETWEEN 0 AND 100),
    duration_ms double precision NOT NULL CHECK (duration_ms > 0 AND duration_ms < 'Infinity'::float8),
    explicit boolean NOT NULL,
    danceability double precision NOT NULL CHECK (danceability BETWEEN 0 AND 1),
    energy double precision NOT NULL CHECK (energy BETWEEN 0 AND 1),
    key smallint NOT NULL CHECK (key BETWEEN -1 AND 11),
    loudness double precision NOT NULL CHECK (loudness > '-Infinity'::float8 AND loudness < 'Infinity'::float8),
    mode smallint NOT NULL CHECK (mode IN (0, 1)),
    speechiness double precision NOT NULL CHECK (speechiness BETWEEN 0 AND 1),
    acousticness double precision NOT NULL CHECK (acousticness BETWEEN 0 AND 1),
    instrumentalness double precision NOT NULL CHECK (instrumentalness BETWEEN 0 AND 1),
    liveness double precision NOT NULL CHECK (liveness BETWEEN 0 AND 1),
    valence double precision NOT NULL CHECK (valence BETWEEN 0 AND 1),
    tempo double precision NOT NULL CHECK (tempo > 0 AND tempo < 'Infinity'::float8),
    time_signature smallint NOT NULL CHECK (time_signature BETWEEN 0 AND 7)
);

CREATE TABLE IF NOT EXISTS spotify.genres (
    genre_name text PRIMARY KEY CHECK (length(btrim(genre_name)) > 0)
);

CREATE TABLE IF NOT EXISTS spotify.track_genres (
    track_id text NOT NULL REFERENCES spotify.tracks(track_id),
    genre_name text NOT NULL REFERENCES spotify.genres(genre_name),
    PRIMARY KEY (track_id, genre_name)
);
CREATE INDEX IF NOT EXISTS track_genres_genre_idx
    ON spotify.track_genres (genre_name, track_id);

-- One current snapshot, replaced transactionally with the catalog.
CREATE TABLE IF NOT EXISTS spotify.dataset_snapshot (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    source_name text NOT NULL,
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    quality_report jsonb NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
