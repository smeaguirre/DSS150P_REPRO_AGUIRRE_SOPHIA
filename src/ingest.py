"""Source dataset verification (SHA-256, byte size, row count) and full/3-batch incremental ingestion."""
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


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest(source_path: str, row_count: int | None = None) -> dict:
    """Verify source CSV SHA-256 checksum, byte size, and expected row count against metadata/source_manifest.yml."""
    actual_sha = compute_file_sha256(source_path)
    actual_size = os.path.getsize(source_path)
    for manifest_path in ("metadata/source_manifest.yml", "config/source_manifest.yml"):
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f) or {}
            src_block = manifest.get("source", manifest) if isinstance(manifest.get("source"), dict) else manifest
            expected_sha = src_block.get("sha256_checksum") or src_block.get("expected_sha256")
            expected_size = src_block.get("expected_size_bytes") or src_block.get("file_size_bytes")
            expected_rows = src_block.get("expected_rows")
            if expected_sha and expected_sha != "REPLACE_WITH_ACTUAL_HASH" and actual_sha != expected_sha:
                raise ValueError(f"Source SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
            if expected_size and int(expected_size) != actual_size:
                raise ValueError(f"Source byte size mismatch: expected {expected_size}, got {actual_size}")
            if row_count is not None and expected_rows and int(expected_rows) != int(row_count):
                raise ValueError(f"Source row count mismatch: expected {expected_rows}, got {row_count}")
            return {"sha256": actual_sha, "size_bytes": actual_size, "manifest_path": manifest_path}
    return {"sha256": actual_sha, "size_bytes": actual_size, "manifest_path": None}


def log_structured_event(stage: str, status: str, row_count: int, duration_sec: float, details: dict | None = None, error: str | None = None) -> dict:
    """Write structured JSON log line to outputs/structured_pipeline_logs.jsonl."""
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
    return record


def run_ingestion(mode: str = "full", force: bool = False) -> dict:
    """Execute full or 3-batch incremental ingestion with watermark checkpointing."""
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

    df = pd.read_csv(source_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    total_source_rows = len(df)
    manifest_info = verify_source_manifest(source_path, row_count=total_source_rows)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(watermark_path) or ".", exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts_naive = pd.Timestamp.now("UTC").tz_localize(None)

    if mode == "full":
        df["_etl_loaded_at"] = now_ts_naive
        df["_ingest_mode"] = "full"
        df.to_parquet(raw_path, index=False)
        watermark = {
            "last_ingested_row": total_source_rows,
            "total_source_rows": total_source_rows,
            "batches_completed": 1,
            "total_batches": 1,
            "mode": "full",
            "source_checksum": manifest_info["sha256"],
            "source_size_bytes": manifest_info["size_bytes"],
            "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
            "updated_at_utc": now_iso,
        }
        with open(watermark_path, "w", encoding="utf-8") as f:
            json.dump(watermark, f, indent=2)
        return log_structured_event(
            "ingest_source",
            "SUCCESS",
            len(df),
            time.time() - start_time,
            details={"mode": "full", "force": force, "source_checksum": manifest_info["sha256"], "size_bytes": manifest_info["size_bytes"]},
        )

    wm = {}
    if not force and os.path.exists(watermark_path):
        with open(watermark_path, "r", encoding="utf-8") as f:
            wm = json.load(f)

    if force or not wm or wm.get("mode") != "incremental":
        if wm.get("mode") == "full" and not force:
            raise ValueError("Existing full mode watermark found. Run with --force to restart incremental ingestion.")
        last_row = 0
        batches_done = 0
    else:
        last_row = int(wm.get("last_ingested_row", 0))
        batches_done = int(wm.get("batches_completed", 0))

    if last_row >= total_source_rows and os.path.exists(raw_path) and not force:
        existing_df = pd.read_parquet(raw_path)
        return log_structured_event(
            "ingest_source",
            "SKIPPED_UP_TO_DATE",
            len(existing_df),
            time.time() - start_time,
            details={"mode": "incremental", "batch": f"{batches_done}/3", "watermark_row": last_row, "source_checksum": manifest_info["sha256"]},
        )

    batch_bounds = np.linspace(0, total_source_rows, 4, dtype=int)
    next_batch_idx = min(batches_done + 1, 3)
    end_row = int(batch_bounds[next_batch_idx])

    new_slice = df.iloc[last_row:end_row].copy()
    new_slice["_etl_loaded_at"] = now_ts_naive
    new_slice["_ingest_mode"] = "incremental"

    if last_row == 0 or not os.path.exists(raw_path):
        combined_df = new_slice.reset_index(drop=True)
    else:
        existing_df = pd.read_parquet(raw_path)
        combined_df = pd.concat([existing_df, new_slice], ignore_index=True)

    combined_df.to_parquet(raw_path, index=False)

    watermark = {
        "last_ingested_row": end_row,
        "total_source_rows": total_source_rows,
        "batches_completed": next_batch_idx,
        "total_batches": 3,
        "mode": "incremental",
        "source_checksum": manifest_info["sha256"],
        "source_size_bytes": manifest_info["size_bytes"],
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
        "updated_at_utc": now_iso,
    }
    with open(watermark_path, "w", encoding="utf-8") as f:
        json.dump(watermark, f, indent=2)

    return log_structured_event(
        "ingest_source",
        "SUCCESS",
        len(combined_df),
        time.time() - start_time,
        details={"mode": "incremental", "batch": f"{next_batch_idx}/3", "rows_added": len(new_slice), "cumulative_rows": len(combined_df)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Spotify dataset in full or 3-batch incremental mode.")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_ingestion(mode=args.mode, force=args.force)
