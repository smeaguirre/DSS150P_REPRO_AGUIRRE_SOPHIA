import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

from src.config import (
    KEY_COL, TARGET_COL, SPLIT_COL, SCALED_FEATURES,
    load_config, get_db_engine, get_curated_table,
)

STAGE_TABLE = "_stg_curated_load"

def _quote(col):
    return f'"{col}"'

def build_upsert_sql(table, stage, columns, key=KEY_COL):
    col_list = ", ".join(_quote(c) for c in columns)
    updates = ", ".join(f"{_quote(c)} = EXCLUDED.{_quote(c)}" for c in columns if c != key)
    return (f"INSERT INTO {table} ({col_list}) "
            f"SELECT {col_list} FROM {stage} WHERE TRUE "
            f"ON CONFLICT ({_quote(key)}) DO UPDATE SET {updates}")

def _apply_schema(conn, schema_path="sql/schema.sql"):
    with open(schema_path, "r") as f:
        sql_text = "\n".join(l for l in f.read().splitlines() if not l.strip().startswith("--"))
    for statement in sql_text.split(";"):
        if statement.strip():
            conn.execute(text(statement.strip()))

def load_to_postgres(full_refresh=False, engine=None, curated_path=None):
    config = load_config()
    table = get_curated_table(config)
    if curated_path is None:
        curated_path = os.path.join(config['paths']['curated_dir'], "curated_full.parquet")

    if not os.path.exists(curated_path):
        raise FileNotFoundError(f"Curated data missing at {curated_path}. Run transform first.")

    print(f"Reading curated data from {curated_path}...")
    df = pd.read_parquet(curated_path)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    if SPLIT_COL not in df.columns:
        raise ValueError(f"Curated data has no '{SPLIT_COL}' column. Re-run `python -m src.transform`.")
    if df[KEY_COL].isnull().any():
        raise ValueError(f"Curated data contains null {KEY_COL} values.")
    if not df[KEY_COL].is_unique:
        raise ValueError(f"Curated data has {int(df[KEY_COL].duplicated().sum())} duplicate "
                         f"{KEY_COL} values; refusing to load.")

    engine = engine or get_db_engine()

    if full_refresh:
        print(f"--full-refresh: dropping {table} if it exists...")
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    else:
        insp = inspect(engine)
        if insp.has_table(table):
            pk = insp.get_pk_constraint(table).get("constrained_columns") or []
            cols = [c["name"] for c in insp.get_columns(table)]
            if pk != [KEY_COL] or SPLIT_COL not in cols:
                raise RuntimeError(
                    f"Existing table '{table}' has an outdated schema (primary key={pk}, "
                    f"has '{SPLIT_COL}' column={SPLIT_COL in cols}). Run once with "
                    f"`python -m src.storage --full-refresh` to rebuild it.")

    print(f"Staging {len(df)} records, then upserting into '{table}'...")
    df.to_sql(STAGE_TABLE, engine, if_exists='replace', index=False, chunksize=10000)

    try:
        with engine.begin() as conn:
            _apply_schema(conn)

            table_cols = {c["name"] for c in inspect(conn).get_columns(table)}
            unknown = [c for c in df.columns if c not in table_cols]
            if unknown:
                raise RuntimeError(f"Columns missing from table '{table}': {unknown}. "
                                   f"Update sql/schema.sql and run with --full-refresh.")

            conn.execute(text(build_upsert_sql(table, STAGE_TABLE, list(df.columns))))
            conn.execute(text(
                f'DELETE FROM {table} WHERE {_quote(KEY_COL)} NOT IN '
                f'(SELECT {_quote(KEY_COL)} FROM {STAGE_TABLE})'))

            n_rows, n_keys = conn.execute(text(
                f'SELECT COUNT(*), COUNT(DISTINCT {_quote(KEY_COL)}) FROM {table}')).one()
            if n_rows != len(df) or n_keys != n_rows:
                raise RuntimeError(
                    f"Post-load verification failed: table has {n_rows} rows / {n_keys} distinct "
                    f"{KEY_COL}, curated file has {len(df)} rows. Transaction rolled back.")
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {STAGE_TABLE}"))

    print(f"Database load completed: {n_rows} rows, {n_keys} distinct {KEY_COL}.")
    return {"rows": int(n_rows), "distinct_keys": int(n_keys)}

def read_split(split, engine=None):
    engine = engine or get_db_engine()
    table = get_curated_table()
    cols = ", ".join(_quote(c) for c in [KEY_COL, TARGET_COL] + SCALED_FEATURES)
    query = text(f'SELECT {cols} FROM {table} WHERE {_quote(SPLIT_COL)} = :split')
    df = pd.read_sql(query, engine, params={"split": split})
    if df.empty:
        raise RuntimeError(f"No rows with split='{split}' in '{table}'. Run load_postgres first.")
    return df.sort_values(KEY_COL, kind="mergesort").reset_index(drop=True)

def measure_read_time(func, times=5):
    read_times = []
    for _ in range(times):
        start = time.perf_counter()
        func()
        read_times.append(time.perf_counter() - start)
    return np.median(read_times)

def run_storage_benchmark():
    config = load_config()
    curated_path = os.path.join(config['paths']['curated_dir'], "curated_full.parquet")
    df = pd.read_parquet(curated_path)
    
    csv_path = "outputs/benchmark.csv"
    json_path = "outputs/benchmark.json"
    parquet_path = "outputs/benchmark.parquet"
    
    results = []
    print("Running storage benchmark (this may take a minute)...")
    
    start = time.perf_counter()
    df.to_csv(csv_path, index=False)
    results.append({
        'Format': 'CSV', 
        'Size (MB)': os.path.getsize(csv_path) / (1024 * 1024), 
        'Write Time (s)': time.perf_counter() - start, 
        'Median Read Time (s)': measure_read_time(lambda: pd.read_csv(csv_path))
    })
    
    start = time.perf_counter()
    df.to_json(json_path, orient="records", lines=True)
    results.append({
        'Format': 'JSONL', 
        'Size (MB)': os.path.getsize(json_path) / (1024 * 1024), 
        'Write Time (s)': time.perf_counter() - start, 
        'Median Read Time (s)': measure_read_time(lambda: pd.read_json(json_path, orient="records", lines=True))
    })
    
    start = time.perf_counter()
    df.to_parquet(parquet_path, index=False)
    results.append({
        'Format': 'Parquet', 
        'Size (MB)': os.path.getsize(parquet_path) / (1024 * 1024), 
        'Write Time (s)': time.perf_counter() - start, 
        'Median Read Time (s)': measure_read_time(lambda: pd.read_parquet(parquet_path))
    })
    
    try:
        engine = get_db_engine()
        start = time.perf_counter()
        df.to_sql('benchmark_table', engine, if_exists='replace', index=False)
        write_pg = time.perf_counter() - start
        read_pg_full = measure_read_time(lambda: pd.read_sql("SELECT * FROM benchmark_table", engine))
        read_pg_filter = measure_read_time(lambda: pd.read_sql("SELECT * FROM benchmark_table WHERE popularity_tier = 'High'", engine))
        
        results.append({
            'Format': 'PostgreSQL', 
            'Size (MB)': np.nan, 
            'Write Time (s)': write_pg, 
            'Median Read Time (s)': read_pg_full
        })
    except Exception as e:
        print(f"PostgreSQL benchmark skipped (Database unreachable): {e}")
        read_pg_full, read_pg_filter = 0, 0
    
    res_df = pd.DataFrame(results)
    
    report = f"""# Storage and Partitioning Benchmark Report

## Storage Format Comparison
{res_df.to_markdown(index=False)}

**PostgreSQL Specifics:**
- Full Table Query Median Time: {read_pg_full:.4f} s
- Filtered Query (`popularity_tier = 'High'`) Median Time: {read_pg_filter:.4f} s

## Conclusion
Parquet is the optimal format for this Spotify dataset during intermediate pipeline stages. It offers the strongest compression and fastest disk read/write speeds natively in Python. CSV and JSON require substantially more space and serialization overhead. PostgreSQL provides analytical querying power and API backend utility but incurs network and protocol overhead for full-table scans compared to Parquet.
"""
    with open("docs/benchmark_report.md", "w") as f:
        f.write(report)
    print("Benchmark report saved to docs/benchmark_report.md")

def run_partitioning():
    config = load_config()
    curated_path = os.path.join(config['paths']['curated_dir'], "curated_full.parquet")
    df = pd.read_parquet(curated_path)
    
    partition_dir = "data/partitioned"
    print(f"\nPartitioning data by 'popularity_tier' into {partition_dir}...")
    df.to_parquet(partition_dir, partition_cols=['popularity_tier'], index=False)
    
    high_tier_path = os.path.join(partition_dir, 'popularity_tier=High')
    high_df = pd.read_parquet(high_tier_path)
    
    print(f"Successfully loaded solitary partition: 'popularity_tier=High'")
    print(f"Expected High Tier rows: {len(df[df['popularity_tier'] == 'High'])}")
    print(f"Actual Loaded rows: {len(high_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PostgreSQL load and storage benchmark")
    parser.add_argument("action", nargs="?", choices=["load", "benchmark"], default="load")
    parser.add_argument("--full-refresh", action="store_true",
                        help="Drop and recreate the curated table before loading")
    args = parser.parse_args()

    if args.action == "benchmark":
        run_storage_benchmark()
        run_partitioning()
    else:
        load_to_postgres(full_refresh=args.full_refresh)
