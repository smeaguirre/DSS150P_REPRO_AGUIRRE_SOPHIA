DROP TABLE IF EXISTS curated_spotify_tracks;

CREATE TABLE curated_spotify_tracks (
    track_id VARCHAR(255),
    artists TEXT,
    album_name TEXT,
    track_name TEXT,
    popularity INTEGER,
    duration_ms INTEGER,
    explicit BOOLEAN,
    danceability FLOAT,
    energy FLOAT,
    key INTEGER,
    loudness FLOAT,
    mode INTEGER,
    speechiness FLOAT,
    acousticness FLOAT,
    instrumentalness FLOAT,
    liveness FLOAT,
    valence FLOAT,
    tempo FLOAT,
    time_signature INTEGER,
    track_genre VARCHAR(100),
    
    -- Audit & Pipeline Metadata Columns
    _etl_loaded_at TIMESTAMP,
    _ingest_mode VARCHAR(50),
    _staged_at TIMESTAMP,
    _staging_row_id INTEGER,
    
    -- Feature Engineered Columns
    popularity_tier VARCHAR(50),
    scaled_danceability FLOAT,
    scaled_energy FLOAT,
    scaled_valence FLOAT,
    scaled_acousticness FLOAT,
    scaled_liveness FLOAT,
    scaled_loudness FLOAT,
    scaled_tempo FLOAT,
    scaled_key FLOAT,
    scaled_mode FLOAT,
    scaled_time_signature FLOAT,
    scaled_instrumentalness FLOAT,
    scaled_speechiness FLOAT
);