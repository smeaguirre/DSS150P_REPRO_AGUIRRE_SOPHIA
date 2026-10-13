import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd
import yaml
from src.config import ALL_AUDIO_FEATURES, KEY_COL, SPLIT_COL, TARGET_COL, load_config

def log_event(stage, status, row_count, duration_sec, error=None, details=None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
        "stage": stage,
        "status": status,
        "row_count": int(row_count),
        "duration_sec": round(duration_sec, 4),
        "error": error,
        "details": details or {},
    }
    print(json.dumps(entry))
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/structured_pipeline_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_contract():
    for path in ("config/data_contract.yml", "metadata/data_contract.yml"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}

def run_five_assertions(df, stage):
    contract = load_contract()
    required_cols = [KEY_COL, "popularity", "duration_ms"] + ALL_AUDIO_FEATURES
    if stage == "curated":
        required_cols += [TARGET_COL, SPLIT_COL]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[Assertion 1 - Schema] Missing required columns: {missing_cols}")

    null_counts = df[required_cols].isnull().sum()
    bad_nulls = null_counts[null_counts > 0].to_dict()
    if bad_nulls:
        raise ValueError(f"[Assertion 2 - Completeness] Null values in required columns: {bad_nulls}")

    quarantined_count = 0
    if stage == "raw":
        text_cols = [c for c in ("artists", "album_name", "track_name") if c in df.columns]
        quarantine_mask = (df["duration_ms"] <= 0) | (df[text_cols].isnull().any(axis=1) if text_cols else False)
        if quarantine_mask.any():
            os.makedirs("data/quarantine", exist_ok=True)
            q_df = df[quarantine_mask]
            quarantined_count = int(len(q_df))
            q_df.to_parquet("data/quarantine/quarantined_rows.parquet", index=False)

    if stage == "curated" and not df[KEY_COL].is_unique:
        dupes = int(df[KEY_COL].duplicated().sum())
        raise ValueError(f"[Assertion 3 - Uniqueness] Found {dupes} duplicate {KEY_COL} rows in curated layer.")
    elif stage == "raw" and df[KEY_COL].nunique() == 0:
        raise ValueError("[Assertion 3 - Uniqueness] No unique track_id keys found.")

    if not df["popularity"].between(0, 100).all():
        raise ValueError("[Assertion 4 - Validity/Range] popularity outside [0, 100].")
    unit_cols = ["danceability", "energy", "valence", "acousticness", "liveness", "speechiness", "instrumentalness"]
    for col in unit_cols:
        if not df[col].between(0.0, 1.0).all():
            raise ValueError(f"[Assertion 4 - Validity/Range] {col} outside [0.0, 1.0].")

    if (df["duration_ms"] < 0).any() or (df["tempo"] < 0).any():
        raise ValueError("[Assertion 5 - Business Rule] Negative duration_ms or tempo detected.")
    if not set(df["mode"].dropna().unique()).issubset({0, 1}):
        raise ValueError("[Assertion 5 - Business Rule] Invalid musical mode outside {0, 1}.")
    if not df["key"].between(-1, 11).all():
        raise ValueError("[Assertion 5 - Business Rule] Invalid pitch class key outside [-1, 11].")

    if stage == "curated":
        valid_tiers = {"Low", "Mid", "High"}
        if not set(df[TARGET_COL].unique()).issubset(valid_tiers):
            raise ValueError(f"[Assertion 5 - Business Rule] Invalid {TARGET_COL} values.")
        train_ids = set(df.loc[df[SPLIT_COL] == "train", KEY_COL])
        test_ids = set(df.loc[df[SPLIT_COL] == "test", KEY_COL])
        if train_ids & test_ids:
            raise ValueError("[Assertion 5 - Business Rule] Train/test track_id leakage detected.")

    return {"contract_loaded": bool(contract), "assertions_verified": 5, "quarantined_rows": quarantined_count}

def write_pipeline_summary(config, status="SUCCESS"):
    def _count_pq(p):
        return int(len(pd.read_parquet(p))) if os.path.exists(p) else None

    def _count_csv(p):
        return int(len(pd.read_csv(p))) if os.path.exists(p) else None

    raw_p = os.path.join(config["paths"]["raw_dir"], config["ingestion"]["raw_filename"])
    stg_p = os.path.join(config["paths"]["staging_dir"], "staging_spotify_tracks.parquet")
    cur_p = os.path.join(config["paths"]["curated_dir"], "curated_full.parquet")
    trn_p = os.path.join(config["paths"]["curated_dir"], "curated_train.parquet")
    tst_p = os.path.join(config["paths"]["curated_dir"], "curated_test.parquet")
    prd_p = os.path.join(config["paths"]["outputs_dir"], "predictions.csv")

    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
        "status": status,
        "counts": {
            "raw": _count_pq(raw_p),
            "staging": _count_pq(stg_p),
            "curated": _count_pq(cur_p),
            "train": _count_pq(trn_p),
            "test": _count_pq(tst_p),
            "predictions": _count_csv(prd_p),
        },
    }
    os.makedirs(config["paths"]["outputs_dir"], exist_ok=True)
    with open(os.path.join(config["paths"]["outputs_dir"], "pipeline_run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def validate_data(stage="raw", inject_failure=False):
    start = time.time()
    config = load_config()
    path = (
        os.path.join(config["paths"]["raw_dir"], config["ingestion"]["raw_filename"])
        if stage == "raw"
        else os.path.join(config["paths"]["curated_dir"], "curated_full.parquet")
    )

    if not os.path.exists(path):
        err = f"Missing dataset for validation at {path}"
        log_event(f"validate_{stage}", "FAILED", 0, time.time() - start, error=err)
        raise FileNotFoundError(err)

    df = pd.read_parquet(path)
    if inject_failure:
        df = df.drop(columns=["danceability"])

    try:
        details = run_five_assertions(df, stage)
    except Exception as exc:
        if inject_failure:
            evidence = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": f"validate_{stage}",
                "injected_fault": "Dropped required column 'danceability' on temporary in-memory copy",
                "caught_exception": str(exc),
                "downstream_blocked": True,
                "status": "RECOVERABLE_FAILURE_CAPTURED",
            }
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/failure_injection_evidence.json", "w", encoding="utf-8") as f:
                json.dump(evidence, f, indent=2)
        log_event(f"validate_{stage}", "FAILED", len(df), time.time() - start, error=str(exc))
        raise

    write_pipeline_summary(config, status="SUCCESS")
    log_event(f"validate_{stage}", "SUCCESS", len(df), time.time() - start, details=details)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["raw", "curated"], default="raw")
    parser.add_argument("--inject-failure", action="store_true")
    args = parser.parse_args()
    try:
        validate_data(stage=args.stage, inject_failure=args.inject_failure)
    except Exception:
        sys.exit(1)
