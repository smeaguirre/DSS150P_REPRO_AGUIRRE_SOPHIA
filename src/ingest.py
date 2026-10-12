import os
import json
import yaml
import pandas as pd
from datetime import datetime
import argparse

def load_config(config_path="config/settings.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def read_watermark(watermark_path):
    """Reads the last ingested index row to support simulated incremental loads."""
    if os.path.exists(watermark_path):
        with open(watermark_path, "r") as f:
            data = json.load(f)
            return data.get("last_ingested_index", -1)
    return -1

def write_watermark(watermark_path, latest_index):
    """Saves the maximum index ingested to the watermark file."""
    with open(watermark_path, "w") as f:
        json.dump({"last_ingested_index": int(latest_index)}, f)

def ingest_data(mode="full"):
    config = load_config()
    source_path = os.path.join(config['paths']['source_dir'], config['ingestion']['source_filename'])
    raw_path = os.path.join(config['paths']['raw_dir'], config['ingestion']['raw_filename'])
    watermark_path = config['ingestion']['watermark_file']

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file missing at {source_path}. Please place 'dataset.csv' there.")

    print(f"Starting {mode} ingestion...")
    df = pd.read_csv(source_path)

    if mode == "incremental":
        last_index = read_watermark(watermark_path)
        print(f"Filtering records after index {last_index}")
        df = df[df['Unnamed: 0'] > last_index]
        
        if df.empty:
            print("No new records to ingest.")
            return

    # Add Audit / Lineage Columns
    df['_etl_loaded_at'] = datetime.utcnow()
    df['_ingest_mode'] = mode

    # Prevent Duplicates on Incremental Appends
    if mode == "incremental" and os.path.exists(raw_path):
        existing_df = pd.read_parquet(raw_path)
        combined_df = pd.concat([existing_df, df])
        # Deduplicate based on the raw primary key (Unnamed: 0)
        combined_df = combined_df.sort_values('_etl_loaded_at').drop_duplicates(
            subset=['Unnamed: 0'], keep='last'
        )
        final_df = combined_df
    else:
        final_df = df

    # Save to Raw (Parquet format for type safety and compression)
    final_df.to_parquet(raw_path, index=False)
    
    # Update Watermark
    max_index = final_df['Unnamed: 0'].max()
    write_watermark(watermark_path, max_index)
    
    print(f"Ingested {len(df)} new rows. Total raw records: {len(final_df)}.")
    print(f"Data saved to {raw_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Data Ingestion")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full", 
                        help="Ingestion mode: full (overwrite) or incremental (append new)")
    args = parser.parse_args()
    
    ingest_data(mode=args.mode)