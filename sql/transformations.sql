SELECT
    split,
    popularity_tier,
    COUNT(*) AS track_count,
    ROUND(AVG(popularity)::numeric, 2) AS avg_popularity,
    ROUND(AVG(danceability)::numeric, 4) AS avg_danceability,
    ROUND(AVG(energy)::numeric, 4) AS avg_energy
FROM curated_spotify_tracks
GROUP BY split, popularity_tier
ORDER BY split, popularity_tier;
