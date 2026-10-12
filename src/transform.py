import os
import json
import yaml
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MOOD_FEATURES = ['danceability', 'energy', 'valence', 'acousticness', 'liveness']
TECHNICAL_FEATURES = ['loudness', 'tempo', 'key', 'mode', 'time_signature', 'instrumentalness', 'speechiness']
ALL_AUDIO_FEATURES = MOOD_FEATURES + TECHNICAL_FEATURES

def load_config(config_path="config/settings.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_checksum(df: pd.DataFrame) -> str:
    """Computes an MD5 checksum over sorted serialized data to verify determinism."""
    serialized = df.to_json(orient='records', date_format='iso')
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def transform_staging():
    """
    Cleans raw ingested data:
    1. Drops rows missing core metadata (114,000 -> 113,999)
    2. Drops duplicate track_id occurrences (113,999 -> 89,740)
    3. Adds audit lineage columns
    """
    config = load_config()
    raw_path = os.path.join(config['paths']['raw_dir'], config['ingestion']['raw_filename'])
    staging_path = os.path.join(config['paths']['staging_dir'], "staging_spotify_tracks.parquet")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingestion first.")

    print(f"[{datetime.utcnow().isoformat()}] Loading raw dataset from {raw_path}...")
    df = pd.read_parquet(raw_path)
    initial_count = len(df)

    # 1. Drop rows missing critical metadata
    df_clean = df.dropna(subset=['artists', 'album_name', 'track_name']).copy()
    post_null_count = len(df_clean)
    print(f"Dropped {initial_count - post_null_count} row(s) with missing metadata. Remaining: {post_null_count}")

    # 2. Drop duplicate track_id entries (keep first occurrence)
    df_dedup = df_clean.drop_duplicates(subset=['track_id'], keep='first').copy()
    dedup_count = len(df_dedup)
    print(f"Dropped {post_null_count - dedup_count} duplicate track_id row(s). Remaining: {dedup_count}")

    # 3. Add audit columns
    df_dedup['_staged_at'] = datetime.utcnow()
    df_dedup['_staging_row_id'] = np.arange(len(df_dedup))

    os.makedirs(config['paths']['staging_dir'], exist_ok=True)
    df_dedup.to_parquet(staging_path, index=False)
    print(f"Staging table successfully created: {staging_path} ({len(df_dedup)} records)")
    return staging_path

def discretize_popularity(val: int) -> str:
    if val <= 33:
        return "Low"
    elif val <= 66:
        return "Mid"
    return "High"

def build_curated(random_state=42):
    """
    Builds the model-ready dataset:
    1. Discretizes popularity into Low, Mid, High tiers
    2. Stratified 80/20 train/test split on popularity_tier
    3. Fits StandardScaler ONLY on the train split to prevent leakage
    4. Writes curated train/test files and split_manifest.json
    """
    config = load_config()
    staging_path = os.path.join(config['paths']['staging_dir'], "staging_spotify_tracks.parquet")
    curated_dir = config['paths']['curated_dir']
    metadata_dir = config['paths']['metadata_dir']

    if not os.path.exists(staging_path):
        raise FileNotFoundError(f"Staging data not found at {staging_path}. Run staging transform first.")

    df = pd.read_parquet(staging_path)

    # 1. Target Construction
    df['popularity_tier'] = df['popularity'].apply(discretize_popularity)

    # 2. Stratified 80/20 Split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df['popularity_tier']
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # 3. Fit scaler ONLY on train set to prevent leakage
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_df[ALL_AUDIO_FEATURES])
    scaled_test_features = scaler.transform(test_df[ALL_AUDIO_FEATURES])

    scaled_col_names = [f"scaled_{col}" for col in ALL_AUDIO_FEATURES]
    train_scaled_df = pd.DataFrame(scaled_train_features, columns=scaled_col_names, index=train_df.index)
    test_scaled_df = pd.DataFrame(scaled_test_features, columns=scaled_col_names, index=test_df.index)

    # Combine back metadata, targets, unscaled, and scaled features
    train_final = pd.concat([train_df, train_scaled_df], axis=1)
    test_final = pd.concat([test_df, test_scaled_df], axis=1)

    os.makedirs(curated_dir, exist_ok=True)
    train_path = os.path.join(curated_dir, "curated_train.parquet")
    test_path = os.path.join(curated_dir, "curated_test.parquet")
    full_curated_path = os.path.join(curated_dir, "curated_full.parquet")

    train_final.to_parquet(train_path, index=False)
    test_final.to_parquet(test_path, index=False)
    pd.concat([train_final, test_final]).to_parquet(full_curated_path, index=False)

    # 4. Generate split_manifest.json for reproducibility audit
    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "random_state": random_state,
        "split_ratio": {"train": 0.8, "test": 0.2},
        "stratify_column": "popularity_tier",
        "counts": {
            "total_curated": len(df),
            "train": len(train_final),
            "test": len(test_final),
            "train_class_distribution": train_final['popularity_tier'].value_counts().to_dict(),
            "test_class_distribution": test_final['popularity_tier'].value_counts().to_dict()
        },
        "feature_sets": {
            "mood_features": MOOD_FEATURES,
            "technical_features": TECHNICAL_FEATURES,
            "all_features": ALL_AUDIO_FEATURES
        },
        "checksums": {
            "train_checksum": compute_checksum(train_final[['track_id', 'popularity']]),
            "test_checksum": compute_checksum(test_final[['track_id', 'popularity']])
        }
    }

    manifest_path = os.path.join(metadata_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Curated layers built successfully:")
    print(f"  - Train: {train_path} ({len(train_final)} rows)")
    print(f"  - Test: {test_path} ({len(test_final)} rows)")
    print(f"  - Manifest written to {manifest_path}")

if __name__ == "__main__":
    transform_staging()
    build_curated()