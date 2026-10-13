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
        baseline_metrics = json.load(f)
    baseline_accuracy = baseline_metrics["accuracy"]
    
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
    current_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Baseline Accuracy: {baseline_accuracy:.6f}")
    print(f"Current Accuracy:  {current_accuracy:.6f}")
    
    # Allow a minor 1% tolerance window for environment row-ordering variations
    tolerance = 0.01
    assert abs(current_accuracy - baseline_accuracy) <= tolerance, "🚨 Reproducibility audit FAILED! Accuracy drifted beyond tolerance."
    
    print("🎉 REPRODUCIBILITY AUDIT PASSED: Model accuracy is within acceptable deterministic limits.")

if __name__ == "__main__":
    verify_reproducibility()