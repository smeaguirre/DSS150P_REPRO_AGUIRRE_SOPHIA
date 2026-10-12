# Source Data Profile: Spotify Tracks Dataset

## Overview
This dataset contains audio features and metadata for 114,000 Spotify tracks across 125 genres. It is used to predict track popularity using a comparative ablation study between Mood features (danceability, energy, valence, acousticness, liveness) and Technical features (loudness, tempo, key, mode, time signature, instrumentalness, speechiness).

## Source Systems
* **Platform**: Spotify Web API (Tracks Audio Features)
* **Provider**: Maharshi Pandya via Kaggle ("Spotify Tracks Dataset")

## Data Characteristics
* **Granularity**: Track-level per genre.
* **Update Frequency**: Static snapshot for reproducibility.
* **Privacy/Compliance**: Public catalog metadata and derived acoustic features. No PII.
* **Primary Keys**: 
  * The dataset contains 114,000 rows. 
  * `track_id` is NOT unique across the raw dataset (the same track can appear in multiple genres, resulting in 24,259 duplicates before cleaning). 
  * The true raw primary key is the row index (`Unnamed: 0`).

## Known Anomalies
* Explicit duplicates exist across `track_id` due to tracks belonging to multiple `track_genre` classifications. Deduplication must occur in the staging layer to prevent train/test leakage.