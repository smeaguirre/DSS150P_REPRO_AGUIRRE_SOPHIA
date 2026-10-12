import pandas as pd
import sys
import os
import yaml

def load_config(config_path="config/settings.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_raw_data():
    config = load_config()
    raw_path = os.path.join(config['paths']['raw_dir'], config['ingestion']['raw_filename'])
    
    print(f"Loading raw data for validation from {raw_path}...")
    df = pd.read_parquet(raw_path)
    
    print("Running Data Quality Contracts...")
    
    # Check 1: Primary Key must not be null
    null_ids = df['track_id'].isnull().sum()
    check_1 = (null_ids == 0)
    if not check_1:
        print(f"❌ FAIL: 'track_id' contains {null_ids} null values!")
    else:
        print("✅ PASS: 'track_id' has no nulls.")
        
    # Check 2: Popularity must be a valid percentage (0-100)
    out_of_bounds = df[~df['popularity'].between(0, 100)]
    check_2 = len(out_of_bounds) == 0
    if not check_2:
        bad_vals = out_of_bounds['popularity'].head(5).tolist()
        print(f"❌ FAIL: 'popularity' is out of bounds! Invalid values found: {bad_vals}")
    else:
        print("✅ PASS: 'popularity' is between 0 and 100.")
        
    # If any check fails, exit with a status of 1 to tell Airflow to halt the pipeline
    if not (check_1 and check_2):
        print("\n🚨 DATA CONTRACT VIOLATED. Halting pipeline to prevent database corruption.")
        sys.exit(1)
        
    print("\n🎉 All Data Quality Contracts PASSED!")

if __name__ == "__main__":
    validate_raw_data()