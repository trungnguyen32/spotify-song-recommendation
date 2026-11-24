-- Run after the loader. Each SELECT answers a separate analytical question.

-- 1. Which genres have the highest median popularity? Dense ranks preserve ties.
WITH ranked AS (
    SELECT *, DENSE_RANK() OVER (ORDER BY median_popularity DESC) AS popularity_rank
    FROM spotify.genre_summary
    WHERE track_count >= 30
)
SELECT genre_name, track_count, median_popularity, popularity_q25, popularity_q75,
       high_popularity_share, popularity_rank
FROM ranked
WHERE popularity_rank <= 10
ORDER BY popularity_rank, genre_name;

-- 2. Which tracks exceed their genre's typical popularity most strongly?
-- A track can legitimately appear in several genres here.
WITH compared AS (
    SELECT g.genre_name, t.track_id, t.track_name, t.artists, t.popularity,
           t.popularity - s.mean_popularity AS popularity_above_genre_mean,
           ROW_NUMBER() OVER (
               PARTITION BY g.genre_name ORDER BY t.popularity DESC, t.track_id
           ) AS genre_position
    FROM spotify.track_genres g
    JOIN spotify.tracks t USING (track_id)
    JOIN spotify.genre_summary s ON s.genre_name = g.genre_name
)
SELECT * FROM compared WHERE genre_position <= 5
ORDER BY genre_name, genre_position;

-- 3. How does popularity vary across broad energy bands? One row per track.
SELECT * FROM spotify.energy_band_summary
ORDER BY CASE energy_band WHEN 'low' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END;

-- 4. Which audio profiles distinguish genres?
SELECT genre_name, track_count, mean_danceability, mean_energy, mean_valence,
       mean_acousticness, mean_tempo, mean_loudness
FROM spotify.genre_summary ORDER BY genre_name;
