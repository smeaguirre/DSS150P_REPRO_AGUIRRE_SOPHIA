# Phase IV — Reproducibility Demonstration

Pipeline entrypoint: python -m src.cli all
Airflow DAG: predictive_repro_pipeline

| Step | Requirement | Evidence |
|------|-------------|----------|
| 1 | Repo & commit history | GitHub |
| 2 | Secrets not committed | `.gitignore` |
| 3 | Source checksum | `evidence/structured_pipeline_logs_tail.txt` |
| 4 | Environment build | `evidence/pipeline_run1.log` |
| 5 | End-to-end pipeline | `evidence/pipeline_run1.log`, `evidence/pipeline_run2.log` |
| 6 | Layer outputs | `evidence/predictions.csv`, `evidence/metrics.json` |
| 7 | PostgreSQL tables & audit | `evidence/postgres_tables.txt`, `evidence/postgres_audit.txt` |
| 8 | Tests | `evidence/pytest_output.txt` + CI badge |
| 9 | Airflow DAG graph & run states | `evidence/airflow_success.png`, `evidence/airflow_run_url.txt` |
| 10 | Second run, no duplicates | `evidence/duplicate_check.txt` (89740 == 89740) |
| 11 | Baseline vs reproduced metrics | 0.643 baseline vs 0.6462 reproduced (tol 0.01) |
| 12 | Failure / recovery | `evidence/failure_injection_evidence.json` |
| 13 | Production decision | `docs/reproducibility_report.md` |
