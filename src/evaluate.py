import os
import json
import pickle
import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def get_db_engine():
    """Constructs database URI from Airflow environment variables."""
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "airflow")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

def evaluate_model():
    print("Fetching data from PostgreSQL to reconstruct test set...")
    engine = get_db_engine()
    df = pd.read_sql_table('curated_spotify_tracks', engine)
    
    feature_cols = [col for col in df.columns if col.startswith('scaled_')]
    X = df[feature_cols]
    y = df['popularity_tier']
    
    print("Loading Label Encoder and Trained XGBoost Model...")
    with open("data/models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
        
    y_encoded = le.transform(y)
    
    # We use the exact same random_state=42 to perfectly recreate the test set
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = XGBClassifier()
    model.load_model("data/models/xgboost_model.json")
    
    print("Making predictions on the holdout test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    
    # Save metrics to a file for reproducibility proof
    os.makedirs("outputs", exist_ok=True)
    metrics = {
        "accuracy": float(accuracy),
        "model_type": "XGBoost",
        "test_samples": len(y_test)
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("✅ Metrics securely saved to outputs/metrics.json")

if __name__ == "__main__":
    evaluate_model()