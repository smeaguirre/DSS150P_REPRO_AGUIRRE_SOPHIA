# Phase IV — Online Grading Evidence Map (Section 10.1)

Because this project is submitted asynchronously without a live defense, this document maps the 13 required Phase IV demonstration steps directly to the committed code, automated test output, and generated JSON evidence artifacts within this repository.

| Rubric Step (10.1) | Execution Evidence & Artifact Location |
| :--- | :--- |
| **1. Show Repo & Commit History** | 25+ milestone commits visible in the GitHub commit history. |
| **2. .env & Large Artifact Hygiene** | See `.gitignore` (excludes `outputs/benchmark.*`, `.env`, and raw datasets). Template provided in `.env.example`. |
| **3. Verify Source Checksum** | See `metadata/source_manifest.yml` and automated test in `outputs/pytest_log.txt`. |
| **4. Build/Start Environment** | See `docker-compose.yml` (pins Airflow + PostgreSQL) and `Dockerfile`. |
| **5. Execute Pipeline** | See `dags/predictive_repro_pipeline.py`. Output validated via `python -m src.cli evidence`. |
| **6. Show Outputs** | Layer counts tracked systematically in `outputs/pipeline_run_summary.json`. |
| **7. Show PostgreSQL Target** | See `outputs/second_run_idempotency_evidence.json` (Shows `postgres_table_size_mb`). |
| **8. Run Quality & Code Tests** | 24 passing Pytest integration tests documented explicitly in `outputs/pytest_log.txt`. |
| **9. Show Airflow Graph** | See DAG structure test in `tests/test_ingest_validate_dag.py`. |
| **10. Prove No Duplicates on Rerun** | Verified in `outputs/second_run_idempotency_evidence.json` (`idempotent_verified: true` and `net_row_delta: 0`). |
| **11. Side-by-Side Metrics** | Compared in `outputs/reproducibility_check.json` and explained rigorously in `outputs/ablation_study.json`. |
| **12. Failure/Recovery Scenario** | Trace captured in `outputs/failure_injection_evidence.json` (Dropping `danceability` triggers Assertion 1). |
| **13. Production Design Change** | See Questions 13 & 14 in `docs/reproducibility_report.md` (Proposal to replace in-memory Pandas staging with PySpark/Polars and PostgreSQL `COPY`). |

*Note: All dynamic JSON evidence artifacts above can be regenerated on a fresh clone by running `docker compose exec airflow-scheduler python -m src.cli evidence`.*
