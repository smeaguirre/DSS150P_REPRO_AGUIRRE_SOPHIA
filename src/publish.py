import os
import json
import shutil
from datetime import datetime

def publish_pipeline_outputs():
    print("Packaging pipeline outputs and artifacts...")
    
    # Define artifact paths
    artifacts = {
        "model": "data/models/xgboost_model.json",
        "encoder": "data/models/label_encoder.pkl",
        "metrics": "outputs/metrics.json"
    }
    
    # Verify all artifacts exist before publishing
    missing = [name for name, path in artifacts.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"❌ Missing critical artifacts for publishing: {missing}")
        
    # Create a timestamped release/export directory
    release_dir = "outputs/final_release"
    os.makedirs(release_dir, exist_ok=True)
    
    # Copy artifacts to release bundle
    for name, path in artifacts.items():
        shutil.copy(path, release_dir)
        print(f"✔ Packed {name} -> {release_dir}/")
        
    # Generate a final execution manifest
    manifest = {
        "project": "Spotify Track Popularity Reproducibility Pipeline",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "SUCCESS",
        "pipeline_version": "1.0.0"
    }
    
    manifest_path = os.path.join(release_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"🎉 Pipeline outputs successfully published to {release_dir}/!")

if __name__ == "__main__":
    publish_pipeline_outputs()