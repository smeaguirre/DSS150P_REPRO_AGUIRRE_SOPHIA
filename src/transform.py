import os
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    MOOD_FEATURES, TECHNICAL_FEATURES, ALL_AUDIO_FEATURES,
    KEY_COL, TARGET_COL, SPLIT_COL,
    load_config, get_random_state,
)

def compute_checksum(df: pd.DataFrame) -> str:
    serialized = df.to_json(orient='records', date_format='iso')
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def transform_staging():
    config = load_config()
    raw_path = os.path.join(config['paths']['raw_dir'], config['ingestion']['raw_filename'])
    staging_path = os.path.join(config['paths']['staging_dir'], "staging_spotify_tracks.parquet")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingestion first.")

    print(f"[{datetime.utcnow().isoformat()}] Loading raw dataset from {raw_path}...")
    df = pd.read_parquet(raw_path)
    initial_count = len(df)

    df_clean = df.dropna(subset=['artists', 'album_name', 'track_name']).copy()
    post_null_count = len(df_clean)
    print(f"Dropped {initial_count - post_null_count} row(s) with missing metadata. Remaining: {post_null_count}")

    df_dedup = df_clean.drop_duplicates(subset=['track_id'], keep='first').copy()
    dedup_count = len(df_dedup)
    print(f"Dropped {post_null_count - dedup_count} duplicate track_id row(s). Remaining: {dedup_count}")

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

def build_curated(random_state=None):
    config = load_config()
    if random_state is None:
        random_state = get_random_state(config)
    test_size = float(config.get("model", {}).get("test_size", 0.20))

    staging_path = os.path.join(config['paths']['staging_dir'], "staging_spotify_tracks.parquet")
    curated_dir = config['paths']['curated_dir']
    metadata_dir = config['paths']['metadata_dir']

    if not os.path.exists(staging_path):
        raise FileNotFoundError(f"Staging data not found at {staging_path}. Run staging transform first.")

    df = pd.read_parquet(staging_path)

    if not df[KEY_COL].is_unique:
        dupes = int(df[KEY_COL].duplicated().sum())
        raise ValueError(f"Staging data has {dupes} duplicate {KEY_COL} values; "
                         f"re-run transform_staging before building curated.")

    df[TARGET_COL] = df['popularity'].apply(discretize_popularity)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL]
    )

    train_df = train_df.sort_values(KEY_COL).reset_index(drop=True)
    test_df = test_df.sort_values(KEY_COL).reset_index(drop=True)

    overlap = set(train_df[KEY_COL]) & set(test_df[KEY_COL])
    if overlap:
        raise ValueError(f"Train/test leakage: {len(overlap)} {KEY_COL} values appear in both splits.")

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_df[ALL_AUDIO_FEATURES])
    scaled_test_features = scaler.transform(test_df[ALL_AUDIO_FEATURES])

    scaled_col_names = [f"scaled_{col}" for col in ALL_AUDIO_FEATURES]
    train_scaled_df = pd.DataFrame(scaled_train_features, columns=scaled_col_names, index=train_df.index)
    test_scaled_df = pd.DataFrame(scaled_test_features, columns=scaled_col_names, index=test_df.index)

    train_final = pd.concat([train_df, train_scaled_df], axis=1)
    test_final = pd.concat([test_df, test_scaled_df], axis=1)
    train_final[SPLIT_COL] = "train"
    test_final[SPLIT_COL] = "test"

    full_final = (pd.concat([train_final, test_final])
                  .sort_values(KEY_COL).reset_index(drop=True))

    os.makedirs(curated_dir, exist_ok=True)
    train_path = os.path.join(curated_dir, "curated_train.parquet")
    test_path = os.path.join(curated_dir, "curated_test.parquet")
    full_curated_path = os.path.join(curated_dir, "curated_full.parquet")

    train_final.to_parquet(train_path, index=False)
    test_final.to_parquet(test_path, index=False)
    full_final.to_parquet(full_curated_path, index=False)

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "random_state": random_state,
        "split_ratio": {"train": round(1 - test_size, 4), "test": test_size},
        "stratify_column": TARGET_COL,
        "split_column": SPLIT_COL,
        "row_order": f"sorted by {KEY_COL}",
        "counts": {
            "total_curated": len(df),
            "train": len(train_final),
            "test": len(test_final),
            "train_class_distribution": train_final[TARGET_COL].value_counts().to_dict(),
            "test_class_distribution": test_final[TARGET_COL].value_counts().to_dict()
        },
        "feature_sets": {
            "mood_features": MOOD_FEATURES,
            "technical_features": TECHNICAL_FEATURES,
            "all_features": ALL_AUDIO_FEATURES
        },
        "checksums": {
            "train_checksum": compute_checksum(train_final[[KEY_COL, 'popularity']]),
            "test_checksum": compute_checksum(test_final[[KEY_COL, 'popularity']])
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
    parser = argparse.ArgumentParser(description="Staging / curated transformations")
    parser.add_argument("--stage", choices=["staging", "curated", "all"], default="all",
                        help="Which transformation to run (default: all)")
    args = parser.parse_args()
    if args.stage in ("staging", "all"):
        transform_staging()
    if args.stage in ("curated", "all"):
        build_curated()
