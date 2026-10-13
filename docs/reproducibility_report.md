# Reproducibility & Engineering Report — Spotify Track Popularity Pipeline

## 1. Executive Summary & Baseline Comparison
This study re-engineered an exploratory multiclass Spotify track popularity prediction project (`Low`, `Mid`, `High` tiers using XGBoost) into a fully containerized, Airflow-orchestrated data engineering pipeline.

| Metric / Property | Prior Exploratory / Pre-Patch Baseline | Engineered Reproducible Pipeline | Deviation Explanation |
| :--- | :--- | :--- | :--- |
| **Total Raw Rows** | 114,000 | 114,000 | Identical source dataset (`dataset.csv`), verified via SHA-256. |
| **Deduplicated Curated Rows** | Unenforced in DB (179,480 after repeated loads) | **89,740** (`track_id PRIMARY KEY`) | Prior `if_exists='append'` duplicated rows on reruns; fixed via transactional `ON CONFLICT (track_id) DO UPDATE`. |
| **Train / Test Split Sizes** | 143,584 / 35,896 (2x duplicated) | **71,792 train / 17,948 test** | Stored once in `split` column during `build_curated` and verified against `metadata/split_manifest.json`. |
| **Holdout Accuracy** | `~0.71` (inflated by global scaling & duplicate leakage) | **`0.6249`** (`0.624861`) | Prior accuracy was artificially inflated because identical tracks appeared in both train and test sets and `StandardScaler` was not isolated to a single stored split. |
| **Macro-F1 / Balanced Acc.** | Not logged | **`0.4275` / `0.4385`** | Added comprehensive multiclass metrics and `outputs/predictions.csv`. |
| **Live Audit Drift** | Up to `0.037` drift across runs | **`0.000000` (`0.00e+00`)** | Sorting by `track_id` (`mergesort`), pinning `n_jobs=1`, `random_state=42`, `xgboost==3.2.0`, and `scikit-learn==1.7.2` achieved 100% mathematical determinism. |

## 2. Engineering Gap Analysis
1. **Elimination of Data Leakage:** Previously, `train.py`, `evaluate.py`, and `reproducibility_check.py` each called `pd.read_sql_table` (which returns rows in arbitrary PostgreSQL heap order) and independently executed `train_test_split`. Combined with duplicate `append` loads, identical songs crossed the train/test boundary. Now, `transform.build_curated` performs the stratified split once, fits `StandardScaler` exclusively on `train_df`, tags every row with `split IN ('train', 'test')`, and writes MD5 split checksums to `metadata/split_manifest.json`.
2. **Idempotent Relational Storage:** `sql/schema.sql` enforces `track_id VARCHAR(255) PRIMARY KEY` and `CHECK (split IN ('train', 'test'))`. `src/storage.py` loads data via a staging table and executes an atomic `INSERT ... ON CONFLICT (track_id) DO UPDATE` plus stale-row cleanup inside a single transaction.
3. **Two-Stage Reproducibility Gate:** `src/reproducibility_check.py` performs both a live inference audit (`audit_tolerance: 1e-6`) and a baseline comparison against `metadata/expected_metrics.json` (`tolerance: 0.01`). `src/publish.py` refuses to build `outputs/final_release/` unless the audit passes.

---

## 3. Required Analysis and Reflection Questions (Section 14)

### 1. What aspects of the previous predictive analytics project prevented another person from reproducing it immediately?
The prior workflow relied on manual, interactive execution without pinned library versions (`xgboost` and `scikit-learn` were unpinned), hardcoded local file paths and credentials, lacked source dataset SHA-256 verification, re-created train/test splits independently across scripts without deterministic row sorting, and appended records into PostgreSQL without a primary key (`if_exists='append'`), causing duplicate-row data leakage and shifting accuracy scores across runs.

### 2. Which artifact is the true authoritative source: the original file, the raw ingested copy, the curated dataset, or the model-ready dataset? Explain the role of each.
* **True Authoritative Source — Original File (`data/source/dataset.csv`):** The immutable external ground truth verified by SHA-256 in `metadata/source_manifest.yml`. All downstream layers can be deterministically reconstructed from it.
* **Raw Ingested Copy (`data/raw/raw_spotify_data.parquet`):** Preserves the exact source records in a fast columnar format while appending ingestion audit metadata (`_etl_loaded_at`, `_ingest_mode`) to track lineage.
* **Staging Dataset (`data/staging/staging_spotify_tracks.parquet`):** Enforces structural hygiene by removing null text records and deduplicating tracks to one canonical row per `track_id` (89,740 rows).
* **Curated / Model-Ready Dataset (`data/curated/curated_full.parquet` & PostgreSQL `curated_spotify_tracks`):** Encapsulates feature engineering (`scaled_*` audio features fit on train only), target discretization (`popularity_tier`), and the authoritative `split` assignment (`train` vs. `test`) consumed by training and evaluation.

### 3. What identifier or rule did you use to prevent duplicates, and what assumption makes that rule valid?
We used `track_id` as the natural business key (`track_id VARCHAR(255) PRIMARY KEY` in PostgreSQL and `drop_duplicates(subset=['track_id'], keep='first')` after deterministic sorting in staging). This rule is valid under the assumption that Spotify's 22-character base-62 `track_id` uniquely identifies a distinct audio recording and its intrinsic acoustic properties, even if the track is listed under multiple genre playlists in the raw scrape.

### 4. If your dataset has no natural update timestamp, how did you simulate incremental ingestion without pretending that the generated checkpoint is a real business timestamp?
Because the static Kaggle Spotify dataset has no natural business event timestamp, `src/ingest.py --mode incremental` partitions the deterministically ordered source file into 3 sequential row-index batches (`np.linspace(0, total_rows, 4, dtype=int)`) and records `last_ingested_row` and `batches_completed` in `metadata/ingest_watermark.json`. The ingestion timestamp is explicitly named `_etl_loaded_at` to document that it represents pipeline processing time rather than when a song was released or streamed.

### 5. Which transformations belong in staging versus curated, and why?
* **Staging (`transform_staging`):** Schema standardization, dropping corrupt/null records, deduplicating on `track_id`, and adding staging audit metadata (`_staged_at`, `_staging_row_id`). These are task-agnostic data-hygiene operations that do not depend on modeling assumptions.
* **Curated (`build_curated`):** Target discretization (`popularity_tier`), stratified 80/20 train/test splitting (`split` column), and fitting/applying `StandardScaler` on the 12 audio features. These transformations depend on the supervised learning formulation and must occur *after* deduplication and splitting so test distribution statistics never leak into training features.

### 6. Which storage format performed best for your workload? Why should your result not be generalized to every data engineering workload?
In our storage benchmark (`src/storage.py benchmark`), **Parquet** achieved the smallest disk footprint and fastest full-columnar scan and filtered read times compared to CSV, JSON, and PostgreSQL, due to dictionary/run-length compression and columnar projection pushdown. However, this result cannot be generalized to every workload: Parquet is poorly suited for high-frequency point lookups, row-level transactional updates (`OLTP`), or multi-user concurrent writes, where a relational database like PostgreSQL with B-tree indexes (`PRIMARY KEY`, `idx_curated_split`) is required.

### 7. What happens if Airflow retries your load task after the database committed but the task process did not receive confirmation?
When Airflow retries `load_postgres`, `src/storage.py` reloads `curated_full.parquet` into the temporary table `_stg_curated_load` and re-executes the transactional `INSERT INTO curated_spotify_tracks ... ON CONFLICT (track_id) DO UPDATE`, deletes any keys not in `_stg_curated_load`, and verifies `COUNT(*) == 89,740` and `COUNT(DISTINCT track_id) == 89,740`. The table state remains identical to the first commit with zero duplicate rows.

### 8. How does your design make that retry safe?
Retry safety (idempotency) is guaranteed by four mechanisms:
1. `sql/schema.sql` uses `CREATE TABLE IF NOT EXISTS` and enforces `track_id PRIMARY KEY`.
2. `src/storage.py` uses `INSERT ... ON CONFLICT (track_id) DO UPDATE SET ...` rather than blind `append`.
3. Stale rows are pruned via `DELETE FROM curated_spotify_tracks WHERE track_id NOT IN (SELECT track_id FROM _stg_curated_load)`.
4. All DDL/DML and post-load count assertions (`n_rows == len(df)` and `n_keys == n_rows`) run inside a single `with engine.begin() as conn:` transaction block that rolls back automatically on any error.

### 9. Which data-quality check is most likely to detect a change that would silently damage model accuracy?
**Assertion 4 (Validity/Range check on audio features and popularity)** combined with **Assertion 5 (Train/Test leakage and target category check)** in `src/validate.py`. While missing columns (Assertion 1) cause immediate crashes in Scikit-Learn/XGBoost, silent unit shifts (e.g., `loudness` or `danceability` scaled 0–100 instead of 0.0–1.0, or corrupt `popularity` values skewing class balance) and train/test `track_id` overlap do not crash training—they silently degrade or artificially inflate model accuracy.

### 10. How do you know that your training and test rows are the same logical rows used in your baseline result?
`src/transform.py` sorts both splits by `track_id` and records MD5 checksums of `(track_id, popularity)` along with exact row counts (`71,792` train, `17,948` test) and class distributions in `metadata/split_manifest.json`. Downstream, `src/train.py` computes and logs `train_track_ids_sha256` in `outputs/model/training_config.json` and verifies `len(train_df) == expected_split_size("train")`, while `src/evaluate.py` verifies `len(test_df) == expected_split_size("test")` after reading via deterministic `mergesort` on `track_id`.

### 11. What sources of nondeterminism remain in your model or environment?
We eliminated split nondeterminism (`random_state=42`), PostgreSQL heap-order nondeterminism (`ORDER BY track_id` via `mergesort`), multi-threaded floating-point reduction order (`n_jobs=1` in XGBoost), and package drift (`xgboost==3.2.0`, `scikit-learn==1.7.2`). Remaining minor sources of nondeterminism across heterogeneous hardware include CPU architecture differences (x86_64 vs. ARM64 SIMD floating-point instruction rounding) and OS kernel/glibc versions if the Docker base image tag is rebuilt on different host architectures.

### 12. If the reproduced metric differs from the original, how can you determine whether the cause is data, preprocessing, split logic, dependencies, model parameters, or nondeterministic execution?
We isolate the cause systematically through our pipeline's layered checkpoints:
1. **Data:** Compare the SHA-256 of `dataset.csv` against `metadata/source_manifest.yml`.
2. **Preprocessing & Split Logic:** Compare `train_checksum`, `test_checksum`, and `counts` in `metadata/split_manifest.json` and `train_track_ids_sha256` in `training_config.json`.
3. **Dependencies & Model Parameters:** Inspect `xgboost_version`, `pandas_version`, `random_state`, and `xgboost_params` recorded in `outputs/model/training_config.json`.
4. **Nondeterministic Execution:** Run `src/reproducibility_check.py`, which re-scores the saved model on the live test split (`live_audit`). If `live_audit` passes (`diff == 0.0`) on repeated runs with the same artifacts, execution is deterministic and any deviation stems from an upstream layer above.

### 13. What would you monitor if this pipeline ran every day in production?
1. **Data Drift & Feature Distribution Shifts:** Population Stability Index (PSI) or Kolmogorov-Smirnov statistics on the 12 audio features and target class proportions (`High` tier is only ~5% of tracks).
2. **Pipeline Operational Metrics:** Task duration trends, quarantine row counts (`data/quarantine/`), and row-count deltas in `outputs/structured_pipeline_logs.jsonl`.
3. **Model Performance Decay:** Macro-F1, per-class recall (especially on the minority `High` popularity tier), and calibration drift against delayed actual streaming popularity.

### 14. Which part of your current architecture would become the first bottleneck if the dataset grew by 100x?
At 100x scale (~11.4 million rows), **in-memory Pandas transformations (`pd.read_csv`, `pd.read_parquet`) and full-table staging loads (`df.to_sql` into `_stg_curated_load`)** inside a single Airflow container would hit memory and I/O bottlenecks first. To scale, we would replace single-node Pandas with Polars/DuckDB or PySpark, partition Parquet files by ingestion date/genre, and replace `pandas.to_sql` with PostgreSQL native `COPY FROM STDIN` on incremental delta partitions.

### 15. What would you change if the dataset became sensitive or regulated data?
1. **PII Anonymization & Tokenization:** Hash or tokenize artist/user identifiers and strip free-text fields before landing in staging.
2. **Encryption & Access Control:** Enforce TLS (`sslmode=require`) for PostgreSQL connections, encrypt Parquet/S3 storage volumes at rest (AES-256/KMS), and replace `.env` files with a secrets manager (AWS Secrets Manager / HashiCorp Vault) with least-privilege RBAC roles.
3. **Compliance & Lineage Auditing:** Implement automated right-to-erasure (`DELETE` propagation across Parquet partitions, PostgreSQL, and model retraining) and immutable access audit logs.
