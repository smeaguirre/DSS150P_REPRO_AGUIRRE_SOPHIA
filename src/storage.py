import os
import yaml
import pandas as pd
from sqlalchemy import create_engine, text

def load_config(config_path="config/settings.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_db_engine():
    """Constructs database URI from environment variables."""
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")
    host = os.getenv("POSTGRES_HOST", "localhost") # Use localhost when running locally
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "airflow")
    
    uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(uri)

def load_to_postgres():
    """
    Loads the curated dataset into PostgreSQL.
    Ensures rerun-safety by executing the DROP/CREATE schema script first.
    """
    config = load_config()
    curated_path = os.path.join(config['paths']['curated_dir'], "curated_full.parquet")
    
    if not os.path.exists(curated_path):
        raise FileNotFoundError(f"Curated data missing at {curated_path}. Run transform.py first.")
        
    print(f"Reading curated data from {curated_path}...")
    df = pd.read_parquet(curated_path)
    
    engine = get_db_engine()
    
    # 1. Rerun-safe: Execute schema.sql to recreate the table cleanly
    schema_path = "sql/schema.sql"
    print("Applying PostgreSQL schema (Rerun-Safe)...")
    with engine.begin() as conn:
        with open(schema_path, "r") as f:
            sql_commands = f.read()
            # Split by semicolon to execute commands individually
            for command in sql_commands.split(';'):
                if command.strip():
                    conn.execute(text(command))
                    
    # 2. Insert the data
    print(f"Loading {len(df)} records into PostgreSQL table 'curated_spotify_tracks'...")
    # Use if_exists='append' because our schema.sql already provided a clean slate
    df.to_sql('curated_spotify_tracks', engine, if_exists='append', index=False)
    
    print("Database load completed successfully.")

if __name__ == "__main__":
    load_to_postgres()