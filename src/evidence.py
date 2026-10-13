import json, os, shutil, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, xgboost as xgb
from sqlalchemy import text
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import get_db_engine
from src.storage import load_to_postgres
from src.transform import discretize_popularity
from src.ingest import run_ingestion

KEY_COL = "track_id"
TARGET_COL = "popularity_tier"
LABEL_CLASSES = ["High", "Low", "Mid"]
ALL_AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness", "liveness", "loudness", "tempo", "key", "mode", "time_signature", "instrumentalness", "speechiness"]
CONTINUOUS_FEATURES = ["danceability", "energy", "valence", "acousticness", "liveness", "loudness", "tempo", "instrumentalness", "speechiness"]

def generate_all_evidence():
    os.makedirs("outputs", exist_ok=True)
    seed = 42
    engine = get_db_engine()
    
    print("1/5 Generating Incremental Trace...")
    trace = [run_ingestion(mode="incremental", force=True)] + [run_ingestion(mode="incremental") for _ in range(3)]
    with open("outputs/incremental_ingestion_evidence.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "batches": trace}, f, indent=2)

    print("2/5 Sweeping Alternative Metrics (Honest Ablation)...")
    raw_df = pd.read_parquet("data/raw/raw_spotify_data.parquet").dropna(subset=[KEY_COL] + ALL_AUDIO_FEATURES).copy()
    raw_df[TARGET_COL] = raw_df["popularity"].apply(discretize_popularity)
    
    le = LabelEncoder().fit(LABEL_CLASSES)
    r_tr, r_te = train_test_split(raw_df, test_size=0.20, random_state=seed, stratify=raw_df[TARGET_COL])
    
    sc = StandardScaler()
    X_tr_s = r_tr[ALL_AUDIO_FEATURES].copy()
    X_te_s = r_te[ALL_AUDIO_FEATURES].copy()
    X_tr_s[CONTINUOUS_FEATURES] = sc.fit_transform(r_tr[CONTINUOUS_FEATURES])
    X_te_s[CONTINUOUS_FEATURES] = sc.transform(r_te[CONTINUOUS_FEATURES])
    
    m_cls = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric="mlogloss")
    m_cls.fit(X_tr_s, le.transform(r_tr[TARGET_COL]))
    acc_val = float(accuracy_score(le.transform(r_te[TARGET_COL]), m_cls.predict(X_te_s)))
    roc_auc = float(roc_auc_score(le.transform(r_te[TARGET_COL]), m_cls.predict_proba(X_te_s), multi_class="ovr"))
    
    m_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed)
    m_reg.fit(X_tr_s, r_tr["popularity"])
    r2_val = float(r2_score(r_te["popularity"], m_reg.predict(X_te_s)))

    with open("outputs/metrics.json") as f: live_metrics = json.load(f)
    with open("metadata/split_manifest.json") as f: split_meta = json.load(f)

    ablation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regimes": {
            "1_metric_mismatch_hypothesis_sweep": {
                "multiclass_accuracy": round(acc_val, 6),
                "regression_r2_score": round(r2_val, 6),
                "one_vs_rest_roc_auc": round(roc_auc, 6),
                "historical_reported_metric": 0.7145,
                "conclusion": f"ROC-AUC ({round(roc_auc, 4)}) is a candidate for the historical metric, but R2 ({round(r2_val, 4)}) is far off. Neither strictly reproduces 0.7145. The exact 0.7145 baseline remains unexplained and pending instructor approval."
            },
            "2_engineered_deduplicated_pipeline": {
                "total_rows": split_meta["counts"]["total_curated"],
                "train_rows": split_meta["counts"]["train"],
                "test_rows": split_meta["counts"]["test"],
                "reproduced_accuracy": live_metrics["accuracy"]
            }
        }
    }
    with open("outputs/ablation_study.json", "w", encoding="utf-8") as f: json.dump(ablation, f, indent=2)

    print("3/5 Generating Idempotency Proof...")
    with engine.connect() as conn:
        before_rows = conn.execute(text("SELECT COUNT(*) FROM curated_spotify_tracks")).scalar()
    load_to_postgres(full_refresh=False, engine=engine)
    with engine.connect() as conn:
        after_rows = conn.execute(text("SELECT COUNT(*) FROM curated_spotify_tracks")).scalar()
        after_keys = conn.execute(text("SELECT COUNT(DISTINCT track_id) FROM curated_spotify_tracks")).scalar()
        pg_bytes = conn.execute(text("SELECT pg_total_relation_size('curated_spotify_tracks')")).scalar()
    
    with open("outputs/second_run_idempotency_evidence.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "table_name": "curated_spotify_tracks",
            "postgres_table_size_mb": round(float(pg_bytes) / (1024 * 1024), 2),
            "before_load_rows": int(before_rows),
            "after_second_load_rows": int(after_rows),
            "duplicate_business_keys_introduced": int(after_rows - after_keys),
            "net_row_delta": int(after_rows - before_rows),
            "idempotent_verified": bool(after_rows == before_rows and (after_rows - after_keys) == 0)
        }, f, indent=2)

    print("4/5 Generating Partition Verification...")
    cur_df = pd.read_parquet("data/curated/curated_full.parquet")
    part_dir = "data/partitioned"
    if os.path.exists(part_dir): shutil.rmtree(part_dir)
    cur_df.to_parquet(part_dir, partition_cols=["popularity_tier"], index=False)
    df_m = pd.read_parquet(os.path.join(part_dir, "popularity_tier=Mid"))
    df_h = pd.read_parquet(os.path.join(part_dir, "popularity_tier=High"))
    
    sum_matches = (len(df_m) + len(df_h)) == len(cur_df)
    with open("outputs/partition_verification.json", "w", encoding="utf-8") as f:
        json.dump({
            "partition_column": "popularity_tier",
            "total_curated_rows": int(len(cur_df)),
            "mid_tier_rows": int(len(df_m)),
            "high_tier_rows": int(len(df_h)),
            "independent_partition_read_verified": bool(sum_matches)
        }, f, indent=2)

    print("5/5 Dynamic PostgreSQL Benchmark...")
    bench_dir = "data/benchmark_temp"
    os.makedirs(bench_dir, exist_ok=True)
    results = []
    
    for fmt, fname in [("CSV", "b.csv"), ("JSON", "b.json"), ("Parquet", "b.parquet")]:
        path = os.path.join(bench_dir, fname)
        w_times, r_times = [], []
        for _ in range(3):
            t0 = time.time()
            if fmt=="CSV": cur_df.to_csv(path, index=False)
            elif fmt=="JSON": cur_df.to_json(path, orient="records")
            else: cur_df.to_parquet(path, index=False)
            w_times.append(time.time() - t0)
            
            t0 = time.time()
            if fmt=="CSV": pd.read_csv(path)
            elif fmt=="JSON": pd.read_json(path, orient="records")
            else: pd.read_parquet(path)
            r_times.append(time.time() - t0)
            
        results.append({"Format": fmt, "Rows": int(len(cur_df)), "Size_MB": round(os.path.getsize(path)/(1024*1024), 2), "Write_Sec": round(np.median(w_times), 4), "Read_Sec": round(np.median(r_times), 4)})
    shutil.rmtree(bench_dir, ignore_errors=True)

    t0 = time.time()
    with engine.begin() as conn: cur_df.to_sql("bench_spotify_tracks", conn, if_exists="replace", index=False, chunksize=5000)
    pg_w = time.time() - t0
    t0 = time.time()
    pd.read_sql("SELECT * FROM bench_spotify_tracks", engine)
    pg_r = time.time() - t0
    with engine.connect() as conn:
        pg_mb = conn.execute(text("SELECT pg_total_relation_size('bench_spotify_tracks')")).scalar() / (1024*1024)
        conn.execute(text("DROP TABLE bench_spotify_tracks"))
        
    results.append({"Format": "PostgreSQL", "Rows": int(len(cur_df)), "Size_MB": round(pg_mb, 2), "Write_Sec": round(pg_w, 4), "Read_Sec": round(pg_r, 4)})
    
    with open("outputs/storage_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "results": results}, f, indent=2)
    with open("docs/benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("# Dynamic Storage Format Benchmark\n\nGenerated automatically via `python -m src.cli evidence`.\n\n" + pd.DataFrame(results).to_markdown(index=False) + "\n")

if __name__ == "__main__":
    generate_all_evidence()
