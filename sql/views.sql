-- Global measures read tracks directly, avoiding track-genre multiplication.
CREATE OR REPLACE VIEW spotify.catalog_overview AS
SELECT COUNT(*) AS unique_tracks,
       (SELECT COUNT(*) FROM spotify.genres) AS genre_count,
       AVG(popularity) AS mean_popularity,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY popularity) AS median_popularity,
       COUNT(*) FILTER (WHERE explicit) AS explicit_tracks
FROM spotify.tracks;

CREATE OR REPLACE VIEW spotify.genre_summary AS
SELECT g.genre_name, COUNT(*) AS track_count,
       AVG(t.popularity) AS mean_popularity,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY t.popularity) AS median_popularity,
       percentile_cont(0.25) WITHIN GROUP (ORDER BY t.popularity) AS popularity_q25,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY t.popularity) AS popularity_q75,
       AVG(t.danceability) AS mean_danceability,
       AVG(t.energy) AS mean_energy,
       AVG(t.valence) AS mean_valence,
       AVG(t.acousticness) AS mean_acousticness,
       AVG(t.tempo) AS mean_tempo,
       AVG(t.loudness) AS mean_loudness,
       COUNT(*) FILTER (WHERE t.popularity >= 60)::float8 / COUNT(*) AS high_popularity_share
FROM spotify.track_genres g
JOIN spotify.tracks t USING (track_id)
GROUP BY g.genre_name;

-- Descriptive associations: these bands do not imply causal effects.
CREATE OR REPLACE VIEW spotify.energy_band_summary AS
SELECT CASE WHEN energy < 0.33 THEN 'low'
            WHEN energy < 0.67 THEN 'medium' ELSE 'high' END AS energy_band,
       COUNT(*) AS track_count, AVG(popularity) AS mean_popularity,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY popularity) AS median_popularity
FROM spotify.tracks
GROUP BY 1;
