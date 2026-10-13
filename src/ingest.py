import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yaml
from src.config import load_config

def compute_file_sha256(filepath):
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_source_manifest(source_path):
    for manifest_path in ("config/source_manifest.yml", "metadata/source_manifest.yml"):
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f) or {}
            expected = (
                manifest.get("source", {}).get("sha256_checksum")
                if isinstance(manifest.get("source"), dict)
                else manifest.get("sha256_checksum")
            )
            actual = compute_file_sha256(source_path)
            if expected and expected != "REPLACE_WITH_ACTUAL_HASH" and actual != expected:
                raise ValueError(f"Source checksum mismatch: expected {expected}, got {actual}")
            return actual
    return compute_file_sha256(source_path)

def log_structured_event(stage, status, row_count, duration_sec, details=None, error=None):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
        "stage": stage,
        "status": status,
        "row_count": int(row_count),
        "duration_sec": round(duration_sec, 4),
        "error": error,
        "details": details or {},
    }
    print(json.dumps(record))
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/structured_pipeline_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def run_ingestion(mode="full", force=False):
    start_time = time.time()
    config = load_config()
    source_path = os.path.join(config["paths"]["source_dir"], config["ingestion"]["source_filename"])
    raw_dir = config["paths"]["raw_dir"]
    raw_path = os.path.join(raw_dir, config["ingestion"]["raw_filename"])
    watermark_path = config["ingestion"]["watermark_file"]

    if not os.path.exists(source_path):
        err = f"Source dataset not found at {source_path}"
        log_structured_event("ingest_source", "FAILED", 0, time.time() - start_time, error=err)
        raise FileNotFoundError(err)

    source_checksum = verify_source_manifest(source_path)
    df = pd.read_csv(source_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    total_source_rows = len(df)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(watermark_path) or ".", exist_ok=True)

    now_ts = datetime.now(timezone.utc).isoformat()

    if mode == "full" or force or not os.path.exists(watermark_path) or not os.path.exists(raw_path):
        df["_etl_loaded_at"] = now_ts
        df["_ingest_mode"] = mode
        df.to_parquet(raw_path, index=False)
        watermark = {
            "last_ingested_row": total_source_rows,
            "total_source_rows": total_source_rows,
            "batches_completed": 3 if mode == "incremental" else 1,
            "total_batches": 3 if mode == "incremental" else 1,
            "mode": mode,
            "source_checksum": source_checksum,
            "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
            "updated_at_utc": now_ts,
        }
        with open(watermark_path, "w", encoding="utf-8") as f:
            json.dump(watermark, f, indent=2)
        log_structured_event(
            "ingest_source",
            "SUCCESS",
             len(df),
            time.time() - start_time,
            details={"mode": mode, "force": force, "source_checksum": source_checksum},
        )
        return

    with open(watermark_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    last_row = int(wm.get("last_ingested_row", 0))
    batches_done = int(wm.get("batches_completed", 0))

    if last_row >= total_source_rows:
        existing_df = pd.read_parquet(raw_path)
        log_structured_event(
            "ingest_source",
            "SKIPPED_UP_TO_DATE",
            len(existing_df),
            time.time() - start_time,
            details={"mode": "incremental", "watermark_row": last_row, "source_checksum": source_checksum},
        )
        return

    batch_bounds = np.linspace(0, total_source_rows, 4, dtype=int)
    next_batch_idx = min(batches_done + 1, 3)
    end_row = int(batch_bounds[next_batch_idx])

    new_slice = df.iloc[last_row:end_row].copy()
    new_slice["_etl_loaded_at"] = now_ts
    new_slice["_ingest_mode"] = "incremental"

    existing_df = pd.read_parquet(raw_path)
    combined_df = pd.concat([existing_df, new_slice], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["track_id", "album_name", "track_name"], keep="first")
    combined_df.to_parquet(raw_path, index=False)

    watermark = {
        "last_ingested_row": end_row,
        "total_source_rows": total_source_rows,
        "batches_completed": next_batch_idx,
        "total_batches": 3,
        "mode": "incremental",
        "source_checksum": source_checksum,
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
        "updated_at_utc": now_ts,
    }
    with open(watermark_path, "w", encoding="utf-8") as f:
        json.dump(watermark, f, indent=2)

    log_structured_event(
        "ingest_source",
        "SUCCESS",
        len(combined_df),
        time.time() - start_time,
        details={"batch": f"{next_batch_idx}/3", "rows_added": len(new_slice), "source_checksum": source_checksum},
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_ingestion(mode=args.mode, force=args.force)
