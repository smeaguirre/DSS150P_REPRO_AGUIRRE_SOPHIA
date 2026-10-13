import os
import json
import hashlib
import joblib
import sklearn
import xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    SCALED_FEATURES, TARGET_COL, KEY_COL,
    load_config, get_random_state, get_db_engine, expected_split_size,
)
from src.storage import read_split

def _fetch_train_split(config):
    print("Connecting to PostgreSQL to fetch the TRAIN split...")
    train_df = read_split("train", get_db_engine())
    expected = expected_split_size("train", config)
    if expected is not None and len(train_df) != expected:
        raise RuntimeError(
            f"Train split has {len(train_df)} rows but metadata/split_manifest.json says {expected}. "
            f"The database table is out of sync with the curated build (duplicates or a stale load?).")
    print(f"Loaded {len(train_df)} training rows. Preparing features...")
    return train_df

def _write_training_config(config, train_df, le, seed, model_type, model_file, params):
    model_dir = config["paths"]["model_dir"]
    ids_hash = hashlib.sha256("\n".join(train_df[KEY_COL].astype(str)).encode("utf-8")).hexdigest()
    training_config = {
        "model_type": model_type,
        "model_file": model_file,
        "model_params": params,
        "random_state": seed,
        "features": SCALED_FEATURES,
        "classes": le.classes_.tolist(),
        "train_samples": int(len(train_df)),
        "train_track_ids_sha256": ids_hash,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
    }
    if model_type == "xgboost":
        training_config["xgboost_params"] = params
    with open(os.path.join(model_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

def train_random_forest():
    config = load_config()
    seed = get_random_state(config)
    params = dict(config["model"]["random_forest"])
    model_dir = config["paths"]["model_dir"]

    train_df = _fetch_train_split(config)
    X_train = train_df[SCALED_FEATURES]
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])

    print(f"Training RandomForest Classifier (random_state={seed}, params={params})...")
    model = RandomForestClassifier(random_state=seed, **params)
    model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    model_file = "random_forest_model.joblib"
    joblib.dump(model, os.path.join(model_dir, model_file), compress=3)
    _write_training_config(config, train_df, le, seed, "random_forest", model_file, params)
    print(f"Model + training_config.json saved to {model_dir}/")

def train_xgboost():
    config = load_config()
    seed = get_random_state(config)
    params = dict(config["model"]["xgboost"])
    model_dir = config["paths"]["model_dir"]

    train_df = _fetch_train_split(config)
    X_train = train_df[SCALED_FEATURES]
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])

    print(f"Training XGBoost Classifier (random_state={seed}, params={params})...")
    model = XGBClassifier(random_state=seed, **params)
    model.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)
    model_file = "xgboost_model.json"
    model.save_model(os.path.join(model_dir, model_file))
    _write_training_config(config, train_df, le, seed, "xgboost", model_file, params)
    print(f"Model + training_config.json saved to {model_dir}/")

def train_model():
    config = load_config()
    model_type = config["model"].get("type", "random_forest")
    if model_type == "random_forest":
        return train_random_forest()
    if model_type == "xgboost":
        return train_xgboost()
    raise ValueError(f"Unknown model.type '{model_type}' in config/settings.yml")

if __name__ == "__main__":
    train_model()