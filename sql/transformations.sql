-- sql/transformations.sql
-- Idempotent transformations for Staging and Curated layers

-- 1. Create Staging Table from Raw Table
DROP TABLE IF EXISTS staging_spotify_tracks;

CREATE TABLE staging_spotify_tracks AS
WITH ranked_tracks AS (
    SELECT 
        raw_index,
        track_id,
        artists,
        album_name,
        track_name,
        popularity,
        duration_ms,
        explicit,
        danceability,
        energy,
        key,
        loudness,
        mode,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,
        time_signature,
        track_genre,
        _etl_loaded_at,
        ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY raw_index ASC) as rn
    FROM raw_spotify_tracks
    WHERE artists IS NOT NULL 
      AND album_name IS NOT NULL 
      AND track_name IS NOT NULL
)
SELECT 
    raw_index,
    track_id,
    artists,
    album_name,
    track_name,
    popularity,
    duration_ms,
    explicit,
    danceability,
    energy,
    key,
    loudness,
    mode,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    time_signature,
    track_genre,
    NOW() AS _staged_at
FROM ranked_tracks
WHERE rn = 1;

-- 2. Create Analytical / Curated Table with Categorical Target
DROP TABLE IF EXISTS curated_spotify_tracks;

CREATE TABLE curated_spotify_tracks AS
SELECT 
    *,
    CASE 
        WHEN popularity <= 33 THEN 'Low'
        WHEN popularity <= 66 THEN 'Mid'
        ELSE 'High'
    END AS popularity_tier,
    NOW() AS _curated_at
FROM staging_spotify_tracks;