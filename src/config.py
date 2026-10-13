import json
import os

import yaml
from sqlalchemy import create_engine

DEFAULT_CONFIG_PATH = "config/settings.yml"

MOOD_FEATURES = ["danceability", "energy", "valence", "acousticness", "liveness"]
TECHNICAL_FEATURES = ["loudness", "tempo", "key", "mode", "time_signature",
                      "instrumentalness", "speechiness"]
ALL_AUDIO_FEATURES = MOOD_FEATURES + TECHNICAL_FEATURES
SCALED_FEATURES = [f"scaled_{c}" for c in ALL_AUDIO_FEATURES]

KEY_COL = "track_id"
TARGET_COL = "popularity_tier"
SPLIT_COL = "split"

def load_config(config_path=None):
    path = config_path or os.getenv("PIPELINE_CONFIG", DEFAULT_CONFIG_PATH)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_random_state(config=None):
    env_value = os.getenv("PIPELINE_RANDOM_STATE")
    if env_value not in (None, ""):
        return int(env_value)
    config = config or load_config()
    return int(config.get("model", {}).get("random_state", 42))

def get_curated_table(config=None):
    config = config or load_config()
    return config.get("database", {}).get("curated_table", "curated_spotify_tracks")

def get_db_engine():
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "airflow")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

def expected_split_size(split, config=None):
    config = config or load_config()
    manifest_path = os.path.join(config["paths"]["metadata_dir"], "split_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)["counts"][split]


import json, os
from datetime import datetime, timezone

def log_structured_event(stage, status, row_count, duration_sec, details=None, error=None):
    os.makedirs("outputs", exist_ok=True)
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": stage, "status": status, "row_count": row_count, "duration_sec": duration_sec, "details": details or {}, "error": error}
    with open("outputs/structured_pipeline_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def update_pipeline_run_summary(*args, **kwargs):
    pass

ALL_AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness", "liveness", "loudness", "tempo", "key", "mode", "time_signature", "instrumentalness", "speechiness"]
CONTINUOUS_FEATURES = ["danceability", "energy", "valence", "acousticness", "liveness", "loudness", "tempo", "instrumentalness", "speechiness"]
SCALED_FEATURE_COLS = [f"scaled_{c}" for c in ALL_AUDIO_FEATURES]
LABEL_CLASSES = ["High", "Low", "Mid"]

