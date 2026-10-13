# Phase IV — Online Grading Evidence Map (Section 10.1)

Because this project is submitted asynchronously without a live defense, this document maps the 13 required Phase IV demonstration steps directly to the committed code, automated test output, and generated JSON evidence artifacts within this repository.

| Rubric Step (10.1) | Execution Evidence & Artifact Location |
| :--- | :--- |
| **1. Show Repo & Commit History** | 25+ milestone commits visible in the GitHub commit history. |
| **2. .env & Large Artifact Hygiene** | See `.gitignore` (excludes `outputs/benchmark.*`, `.env`, and raw datasets). Template provided in `.env.example`. |
| **3. Verify Source Checksum** | See `metadata/source_manifest.yml` and `tests/test_ingest_validate_dag.py::test_source_manifest_detects_checksum_or_row_mismatch`. Tested automatically. |
| **4. Build/Start Environment** | See `docker-compose.yml` (pins Airflow + PostgreSQL) and `Dockerfile`. |
| **5. Execute Pipeline (End-to-End)** | See `dags/predictive_repro_pipeline.py` (10 BashOperator tasks). Successful DAG run screenshot at `docs/airflow_success.png`. |
| **6. Show Outputs (Raw -> Curated)** | Layer counts tracked systematically in `outputs/pipeline_run_summary.json`. |
| **7. Show PostgreSQL Target** | See `outputs/second_run_idempotency_evidence.json` (shows `89,740` loaded rows and exact byte size). |
| **8. Run Quality & Code Tests** | Verified by 23 passing Pytest integration tests (see `tests/` directory) and `outputs/structured_pipeline_logs.jsonl`. |
| **9. Show Airflow Graph** | Verified in `docs/airflow_success.png`. |
| **10. Prove No Duplicates on Rerun** | Verified in `outputs/second_run_idempotency_evidence.json` (`duplicate_business_keys_introduced: 0` and `net_row_delta: 0`). |
| **11. Side-by-Side Metrics** | Compared machine-readably in `outputs/reproducibility_check.json` (`live_audit` diff is `0.0`). |
| **12. Failure/Recovery Scenario** | Trace captured in `outputs/failure_injection_evidence.json` (Dropping `danceability` triggers Assertion 1; immediately followed by successful recovery verification). |
| **13. Production Design Change** | See Question 13 & 14 in `docs/reproducibility_report.md` (Proposal to replace in-memory single-node Pandas staging loads with PySpark/Polars and PostgreSQL `COPY FROM STDIN`). |
