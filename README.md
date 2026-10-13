# DSS150P - Fundamentals of Data Engineering
## Individual Laboratory Activity #4: Reproduction of a Predictive Analytics Project

| Field | Student Entry |
| :--- | :--- |
| **Name** | Sophia Aguirre |
| **Student Number** | 2024109563 |
| **Section** | CM17 |
| **Previous Predictive Analytics Group / Project Title** | Mood versus Technical Audio Features in Predicting Spotify Track Popularity: An Ablation Study Across 125 Genres |
| **Dataset Name / Source** | Spotify Tracks Dataset (public dataset available on Kaggle, containing tracks across 125 genres) |
| **Predictive Task** | Classification |

# Spotify Track Popularity — End-to-End Reproducible MLOps Pipeline

## 1. Purpose
*Note on Scope:* Per course guidelines, 'a model that is slightly less accurate but fully reproducible is preferable to an undocumented model.' This pipeline strictly reproduces the Classification tiering task to ensure deterministic execution, requesting formal deviation approval for historical regression metrics.
 and Overview
This repository implements a containerized, deterministic, end-to-end data engineering and machine learning pipeline for multiclass classification of Spotify track popularity (`popularity_tier`: `Low` [0–33], `Mid` [34–66], `High` [67–100]) using XGBoost, PostgreSQL 15, and Apache Airflow 2.8.1. It reproduces and re-engineers the prior group's exploratory notebook workflow into an automated production pipeline with strict data contracts, leakage-free feature scaling, idempotent database upserts, and automated reproducibility auditing.

## 2. Architecture & Data Lineage
* **Architecture Diagram:** `docs/architecture.png`
* **Dataset-Level Lineage Diagram:** `docs/lineage.png`
* **Operational Runbook:** `docs/runbook.md`
* **Reproducibility & Reflection Report:** `docs/reproducibility_report.md`
* **Source Profile & Schema:** `metadata/source_profile.md`
- **Source Manifest (SHA-256):** `metadata/source_manifest.yml`
- **Data Contract:** `metadata/data_contract.yml`
- **Storage Benchmark Report:** `docs/benchmark_report.md`
- **Data Quality Failure Evidence:** `docs/data_quality_fail.png` and `outputs/failure_injection_evidence.json`

### Data Layer Semantics
1. **Source (`data/source/dataset.csv`):** Immutable raw CSV input (114,000 rows) verified via SHA-256 against `metadata/source_manifest.yml`.
2. **Raw (`data/raw/raw_spotify_data.parquet`):** Parquet ingestion layer with audit metadata (`_etl_loaded_at`, `_ingest_mode`) and watermark checkpointing (`metadata/ingest_watermark.json`).
3. **Quarantine (`data/quarantine/quarantined_rows.parquet`):** Automated isolation of malformed or incomplete source records during raw validation.
4. **Staging (`data/staging/staging_spotify_tracks.parquet`):** Cleaned and deduplicated dataset (89,740 unique `track_id` rows) with `_staged_at` and `_staging_row_id`.
5. **Curated (`data/curated/curated_{train,test,full}.parquet`):** Model-ready layer containing discretized `popularity_tier`, deterministic `split` assignment (`71,792 train` / `17,948 test`), and 12 `scaled_*` audio features fit strictly on the training split.
6. **Relational Target (`curated_spotify_tracks` in PostgreSQL):** Primary-key enforced (`track_id PRIMARY KEY`) table loaded via transactional staging upsert (`INSERT ... ON CONFLICT DO UPDATE`).
7. **Outputs & Release (`outputs/` & `outputs/final_release/`):** Trained model (`outputs/model/xgboost_model.json`), training metadata (`training_config.json`), holdout predictions (`outputs/predictions.csv`), evaluation metrics (`outputs/metrics.json`), audit report (`outputs/reproducibility_check.json`), and SHA-256 release bundle (`outputs/final_release/manifest.json`).

## 3. Prerequisites & Clean-Machine Setup
* **Docker Desktop** (with Docker Compose v2)
* **Git** and **Python 3.10+** (optional for host commands; all pipeline execution runs inside the Docker container)

### Step-by-Step Setup from a Clean Clone
1. **Clone the repository:**
```bash
git clone https://github.com/smeaguirre/DSS150P_REPRO_AGUIRRE_SOPHIA.git
cd DSS150P_REPRO_AGUIRRE_SOPHIA
```
2. **Configure environment variables:**
   Copy `.env.example` to `.env` and ensure `POSTGRES_HOST=postgres`:
```bash
cp .env.example .env
```
3. **Place the source dataset:**
   Place the Kaggle Spotify Tracks dataset CSV at `data/source/dataset.csv`.
4. **Build and start the containerized environment:**
```bash
docker compose up -d --build
```

## 4. Pipeline Execution Commands

### Option A: Execute via Apache Airflow UI
1. Open **http://localhost:8081** (default credentials from `.env`: `airflow` / `airflow`).
2. Unpause and trigger the **`predictive_repro_pipeline`** DAG.
3. Monitor all 10 tasks (`ingest_source` -> `validate_raw` -> `transform_staging` -> `build_curated` -> `load_postgres` -> `validate_curated` -> `train_model` -> `evaluate_model` -> `reproducibility_check` -> `publish_outputs`) to completion.

### Option B: Execute Manually Inside the Container
```bash
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.ingest --mode full
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.validate --stage raw
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.transform --stage all
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.storage
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.validate --stage curated
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.train
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.evaluate
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.reproducibility_check
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.publish
```

## 5. Testing, Data Quality & Storage Benchmark
* **Run the 19 Unit & Integration Tests (against PostgreSQL):**
```bash
docker compose exec -e TEST_DATABASE_URL="postgresql+psycopg2://airflow:airflow@postgres:5432/airflow" airflow-scheduler python -m pytest -v
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.storage
```
* **Run Failure Injection Verification:**
```bash
docker compose exec airflow-scheduler python -m src.validate --stage raw --inject-failure
```
* **Run Storage Format Benchmark & Partition Workflow:**
```bash
docker compose exec -e POSTGRES_HOST=postgres airflow-scheduler python -m src.cli benchmark
```

## 6. Attribution
This pipeline re-engineers the exploratory Spotify Track Popularity classification study into a deterministic, production-grade data engineering architecture for DSS150P.

## 7. Documentation Index
* **Phase IV Evidence Map:** `docs/evidence.md`

## 8. Academic Integrity & AI Use Acknowledgment
In accordance with the DSS150P syllabus policy, AI assistance was used for code review, debugging duplicate-row data leakage, and designing the automated ablation study. All code, SQL queries, data contracts, ablation regimes, and tests were executed and verified by the author.
