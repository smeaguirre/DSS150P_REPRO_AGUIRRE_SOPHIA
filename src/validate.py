"""YAML data-contract enforcement, 5 data-quality assertions, quarantine routing, and failure injection."""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd
import yaml
from src.config import ALL_AUDIO_FEATURES, KEY_COL, SPLIT_COL, TARGET_COL, load_config


def log_event(stage: str, status: str, row_count: int, duration_sec: float, error: str | None = None, details: dict | None = None) -> dict:
    """Append structured JSON validation event to outputs/structured_pipeline_logs.jsonl."""
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
    return entry


def load_contract() -> dict:
    """Load machine-readable data contract from metadata/data_contract.yml."""
    for path in ("metadata/data_contract.yml", "config/data_contract.yml"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError("Data contract YAML not found at metadata/data_contract.yml")


def run_five_assertions(df: pd.DataFrame, stage: str) -> dict:
    """Execute 5 data-quality assertions dynamically driven by metadata/data_contract.yml."""
    contract = load_contract()
    col_specs = contract.get("columns", {})
    assertions_passed = 0

    required_cols = [c for c, spec in col_specs.items() if not spec.get("quarantine_on_null", False)]
    if not required_cols:
        required_cols = [KEY_COL, "popularity", "duration_ms"] + ALL_AUDIO_FEATURES
    if stage == "curated":
        required_cols = [KEY_COL, "popularity", TARGET_COL, SPLIT_COL] + ALL_AUDIO_FEATURES

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[Assertion 1 - Schema] Missing contract-required columns: {missing_cols}")
    assertions_passed += 1

    non_nullable = [c for c in required_cols if not col_specs.get(c, {}).get("nullable", False)]
    null_counts = df[non_nullable].isnull().sum()
    bad_nulls = null_counts[null_counts > 0].to_dict()
    if bad_nulls:
        raise ValueError(f"[Assertion 2 - Completeness] Null values in non-nullable contract columns: {bad_nulls}")
    assertions_passed += 1

    quarantined_count = 0
    if stage == "raw":
        q_cols = [c for c, spec in col_specs.items() if spec.get("quarantine_on_null") and c in df.columns]
        q_mask = (df["duration_ms"] <= 0) | (df[q_cols].isnull().any(axis=1) if q_cols else False)
        if q_mask.any():
            os.makedirs("data/quarantine", exist_ok=True)
            q_df = df[q_mask]
            quarantined_count = int(len(q_df))
            q_df.to_parquet("data/quarantine/quarantined_rows.parquet", index=False)

    key_pattern = contract.get("key_pattern", "^[A-Za-z0-9]{22}$")
    invalid_keys = int((~df[KEY_COL].astype(str).str.match(key_pattern)).sum())
    if invalid_keys > 0:
        raise ValueError(f"[Assertion 3 - Uniqueness/Key Format] Found {invalid_keys} malformed {KEY_COL} keys violating {key_pattern}.")
    if stage == "curated" and not df[KEY_COL].is_unique:
        dupes = int(df[KEY_COL].duplicated().sum())
        raise ValueError(f"[Assertion 3 - Uniqueness] Found {dupes} duplicate {KEY_COL} rows in curated layer.")
    assertions_passed += 1

    for col_name, spec in col_specs.items():
        if col_name not in df.columns:
            continue
        series = df[col_name].dropna()
        if "min" in spec and "max" in spec:
            if not series.between(spec["min"], spec["max"]).all():
                raise ValueError(f"[Assertion 4 - Validity/Range] '{col_name}' outside contract range [{spec['min']}, {spec['max']}].")
        elif "min" in spec:
            if (series < spec["min"]).any():
                raise ValueError(f"[Assertion 4 - Validity/Range] '{col_name}' below contract min {spec['min']}.")
        if "allowed_values" in spec:
            if not set(series.unique()).issubset(set(spec["allowed_values"])):
                raise ValueError(f"[Assertion 4 - Validity/Range] '{col_name}' outside allowed values {spec['allowed_values']}.")
    assertions_passed += 1

    if (df["duration_ms"] < 0).any() or (df["tempo"] < 0).any():
        raise ValueError("[Assertion 5 - Business Rule] Negative duration_ms or tempo detected.")
    if stage == "curated":
        cur_rules = contract.get("curated_rules", {})
        allowed_tiers = set(cur_rules.get("allowed_targets", ["Low", "Mid", "High"]))
        if not set(df[TARGET_COL].unique()).issubset(allowed_tiers):
            raise ValueError(f"[Assertion 5 - Business Rule] Invalid {TARGET_COL} categories outside {allowed_tiers}.")
        train_ids = set(df.loc[df[SPLIT_COL] == "train", KEY_COL])
        test_ids = set(df.loc[df[SPLIT_COL] == "test", KEY_COL])
        if train_ids & test_ids:
            raise ValueError("[Assertion 5 - Business Rule] Data leakage: track_id overlap between train and test splits.")
    assertions_passed += 1

    return {
        "contract_loaded": True,
        "contract_version": contract.get("contract_version", "2.0.0"),
        "assertions_verified": assertions_passed,
        "quarantined_rows": quarantined_count,
    }


def write_pipeline_summary(config: dict, stage: str, start_iso: str, status: str = "SUCCESS") -> None:
    """Write pipeline_run_summary.json with start_time_utc, end_time_utc, and all layer row counts."""
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

    summary_path = os.path.join(config["paths"]["outputs_dir"], "pipeline_run_summary.json")
    existing_start = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_start = json.load(f).get("start_time_utc")
        except Exception:
            existing_start = None

    summary = {
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID", "local_run"),
        "start_time_utc": existing_start or start_iso,
        "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "last_validated_stage": f"validate_{stage}",
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
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def validate_data(stage: str = "raw", inject_failure: bool = False) -> dict:
    """Validate raw or curated Parquet dataset against contract assertions."""
    start = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
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

    if os.path.exists("outputs/failure_injection_evidence.json"):
        try:
            with open("outputs/failure_injection_evidence.json", "r", encoding="utf-8") as f:
                ev = json.load(f)
            ev["recovery_verification"] = {
                "recovery_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": f"validate_{stage}",
                "rows_validated": int(len(df)),
                "assertions_verified": details["assertions_verified"],
                "recovery_status": "RECOVERED_AND_PASSED_ALL_5_ASSERTIONS",
            }
            with open("outputs/failure_injection_evidence.json", "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2)
        except Exception:
            pass

    write_pipeline_summary(config, stage=stage, start_iso=start_iso, status="SUCCESS")
    return log_event(f"validate_{stage}", "SUCCESS", len(df), time.time() - start, details=details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate raw or curated data against data contract.")
    parser.add_argument("--stage", choices=["raw", "curated"], default="raw")
    parser.add_argument("--inject-failure", action="store_true")
    args = parser.parse_args()
    try:
        validate_data(stage=args.stage, inject_failure=args.inject_failure)
    except Exception:
        sys.exit(1)
