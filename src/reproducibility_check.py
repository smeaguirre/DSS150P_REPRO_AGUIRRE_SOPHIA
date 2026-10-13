import os
import json
import pickle
import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_db_engine():
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "airflow")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

def verify_reproducibility():
    print("Running Reproducibility Audit...")
    
    metrics_path = "outputs/metrics.json"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("❌ Baseline metrics.json not found! Run evaluation first.")
        
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    logged_accuracy = metrics_data["accuracy"]
    
    engine = get_db_engine()
    df = pd.read_sql_table('curated_spotify_tracks', engine)
    
    feature_cols = [col for col in df.columns if col.startswith('scaled_')]
    X = df[feature_cols]
    y = df['popularity_tier']
    
    with open("data/models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    y_encoded = le.transform(y)
    
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = XGBClassifier()
    model.load_model("data/models/xgboost_model.json")
    
    y_pred = model.predict(X_test)
    live_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Logged Metrics Accuracy: {logged_accuracy:.6f}")
    print(f"Live Audit Accuracy:     {live_accuracy:.6f}")
    
    tolerance = 0.005
    assert abs(live_accuracy - logged_accuracy) <= tolerance, "🚨 Reproducibility audit FAILED! Logged metrics do not match live model execution."
    
    print("🎉 REPRODUCIBILITY AUDIT PASSED: Pipeline artifacts and live model execution are deterministic.")

if __name__ == "__main__":
    verify_reproducibility()