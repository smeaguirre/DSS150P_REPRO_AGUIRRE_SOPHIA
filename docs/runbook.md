# Operational Runbook — Spotify Track Popularity Reproducibility Pipeline

## 1. Normal Execution Procedure
1. Ensure Docker containers are healthy:
```bash
docker compose ps
```
2. Verify that `data/source/dataset.csv` is present and matches the SHA-256 hash in `metadata/source_manifest.yml`.
3. Trigger the `predictive_repro_pipeline` DAG in the Airflow Web UI (`http://localhost:8081`) or execute `python -m src.cli all` inside the `airflow-scheduler` container.
4. Confirm completion by inspecting `outputs/reproducibility_check.json` (`"passed": true`) and `outputs/final_release/manifest.json` (`"status": "SUCCESS"`).

## 2. Common Failure Modes and Diagnostics

| Failure Scenario | Task / Module | Root Cause | Diagnostic / Resolution |
| :--- | :--- | :--- | :--- |
| **Source Checksum Mismatch** | `ingest_source` (`src.ingest`) | `data/source/dataset.csv` was modified, corrupted, or replaced with a different version. | Restore the authoritative `dataset.csv` matching `metadata/source_manifest.yml`. |
| **Data Quality Assertion Failure** | `validate_raw` / `validate_curated` (`src.validate`) | Missing columns, null values in required columns, out-of-range audio features, or duplicate `track_id` rows. | Check `outputs/structured_pipeline_logs.jsonl` for the exact `[Assertion X]` error. Fix source/upstream transform and rerun. |
| **PostgreSQL Connection Refused** | `load_postgres` / `train_model` | `POSTGRES_HOST` set to `localhost` instead of `postgres` inside container, or `postgres` container not healthy. | Set `POSTGRES_HOST=postgres` in `.env`, run `docker compose up -d`, and retry the task. |
| **Outdated Table Schema Error** | `load_postgres` (`src.storage`) | Existing `curated_spotify_tracks` table lacks `track_id PRIMARY KEY` or `split` column. | Run one-time schema rebuild: `docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.storage --full-refresh`. |
| **Split Row Count Mismatch** | `train_model` / `evaluate_model` | Database table row count differs from `metadata/split_manifest.json` (`71,792` train / `17,948` test). | Re-run `python -m src.storage` to synchronize PostgreSQL with `curated_full.parquet`. |
| **Reproducibility Audit Failure** | `reproducibility_check` | Live inference disagrees with `outputs/metrics.json` or baseline drift exceeds `0.01` tolerance in `metadata/expected_metrics.json`. | Inspect `outputs/reproducibility_check.json`. Re-run `evaluate_model` if `metrics.json` is stale, or investigate seed/version drift. |

## 3. Restart, Rerun, and Idempotency Procedures
Every stage of the pipeline is designed to be idempotent and safe to rerun:
* **Airflow Task Retry:** Click any failed task in the Airflow UI -> **Clear** -> **OK**. Downstream dependencies will automatically wait and execute in order.
* **Database Rerun Safety:** `src/storage.py` loads into a temporary staging table (`_stg_curated_load`), performs an atomic `INSERT INTO curated_spotify_tracks ... ON CONFLICT (track_id) DO UPDATE`, deletes stale keys, and verifies `COUNT(*) == COUNT(DISTINCT track_id)` inside a single transaction before committing. Retrying `load_postgres` never creates duplicate records.

## 4. Database Reset Procedure (Without Losing Airflow History)
**Never** run `docker compose down -v` during normal recovery, because Airflow's metadata database resides in the same PostgreSQL instance and `-v` deletes DAG run history. Instead, reset only the pipeline target table:
```bash
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler bash -c "python -m src.transform --stage curated && python -m src.storage --full-refresh"
```

## 5. Incremental Checkpoint Recovery
* Incremental ingestion (`python -m src.ingest --mode incremental`) divides the 114,000 source records into 3 deterministic batches using `metadata/ingest_watermark.json`.
* **Resume after interruption:** Re-running `python -m src.ingest --mode incremental` reads `last_ingested_row` and `batches_completed` from `metadata/ingest_watermark.json` and ingests only the next uncompleted batch, deduplicating on `(track_id, album_name, track_name)`.
* **Force full reset of ingestion:** Run `python -m src.ingest --mode full --force` (or delete `metadata/ingest_watermark.json`) to rebuild `data/raw/raw_spotify_data.parquet` from scratch.

## 6. Failure Injection & Recovery Demonstration (A13 / A14)
1. **Inject Fault:** Run `docker compose exec airflow-scheduler python -m src.validate --stage raw --inject-failure`. This drops the required `danceability` column on an in-memory copy, triggers `[Assertion 1 - Schema]`, writes `outputs/failure_injection_evidence.json`, logs a `FAILED` entry in `outputs/structured_pipeline_logs.jsonl`, and exits with code `1` so bad data never reaches staging, curated, or model outputs.
2. **Recover:** Run `docker compose exec airflow-scheduler python -m src.validate --stage raw` (without `--inject-failure`). Validation passes all 5 assertions and updates `outputs/pipeline_run_summary.json`.

## 7. Automated Evidence Generation
Running `python -m src.cli evidence` will regenerate all JSON evidence artifacts. **Note:** This command runs the incremental ingestion trace, which overwrites `data/raw/raw_spotify_data.parquet` and `metadata/ingest_watermark.json` as a side effect.


## 8. Rollback & Cleanup
* **Partial Failure:** If a DAG task fails, Airflow retries up to 3 times. PostgreSQL UPSERTs ensure no duplicates.
* **Full Pipeline Reset:**
  1. `docker compose down -v`
  2. Delete all parquet files in `data/raw`, `data/staging`, `data/curated`, `data/partitioned`.
  3. Echo `{}` > `metadata/ingest_watermark.json`
  4. `docker compose up -d`