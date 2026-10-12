import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_db_engine():
    """Constructs database URI from Airflow environment variables."""
    user = os.getenv("POSTGRES_USER", "airflow")
    password = os.getenv("POSTGRES_PASSWORD", "airflow")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "airflow")
    
    uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(uri)

def train_xgboost():
    print("Connecting to PostgreSQL to fetch curated data...")
    engine = get_db_engine()
    
    # Read the data we just loaded in Milestone 07
    df = pd.read_sql_table('curated_spotify_tracks', engine)
    
    print(f"Loaded {len(df)} records. Preparing features...")
    
    # Isolate all the scaled features you built in transform.py
    feature_cols = [col for col in df.columns if col.startswith('scaled_')]
    X = df[feature_cols]
    y = df['popularity_tier']
    
    # XGBoost requires numeric labels (0, 1, 2) instead of text (Low, Mid, High)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Strict reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        random_state=42, 
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=6,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    # Ensure our output directory exists
    os.makedirs("data/models", exist_ok=True)
    
    # Save the model and the label encoder natively
    model.save_model("data/models/xgboost_model.json")
    with open("data/models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    print("✅ Model trained and securely saved to data/models/")

if __name__ == "__main__":
    train_xgboost()