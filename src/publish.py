import os
import json
import shutil
import hashlib
from datetime import datetime, timezone

from src.config import load_config

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def publish_pipeline_outputs():
    config = load_config()
    paths, repro = config["paths"], config["reproducibility"]
    model_dir, outputs_dir = paths["model_dir"], paths["outputs_dir"]
    release_dir = paths.get("release_dir", os.path.join(outputs_dir, "final_release"))

    with open(os.path.join(model_dir, "training_config.json"), "r") as f:
        training_config = json.load(f)
    model_file = training_config.get("model_file", "xgboost_model.json")
    ship_model = training_config.get("model_type", "xgboost") != "random_forest"

    artifacts = {
        "model": os.path.join(model_dir, model_file),
        "training_config": os.path.join(model_dir, "training_config.json"),
        "metrics": repro["metrics_file"],
        "predictions": os.path.join(outputs_dir, "predictions.csv"),
        "reproducibility_report": repro["check_report_file"],
        "split_manifest": os.path.join(paths["metadata_dir"], "split_manifest.json"),
        "expected_metrics": repro["expected_metrics_file"],
    }

    print("Packaging pipeline outputs and artifacts...")
    missing = [name for name, path in artifacts.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing artifacts for publishing: {missing}")

    model_info = None
    if not ship_model:
        model_path = artifacts.pop("model")
        model_info = {"file": model_file, "sha256": _sha256(model_path),
                      "bytes": os.path.getsize(model_path), "included_in_release": False,
                      "reason": "Model is too large to commit; regenerate with `python -m src.cli train`."}

    with open(artifacts["reproducibility_report"], "r") as f:
        report = json.load(f)
    if not report.get("passed"):
        raise RuntimeError("Refusing to publish: the reproducibility audit did not pass "
                           f"({artifacts['reproducibility_report']}).")
    with open(artifacts["metrics"], "r") as f:
        metrics = json.load(f)

    if os.path.isdir(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)

    files = {}
    for name, path in artifacts.items():
        target = shutil.copy(path, release_dir)
        files[name] = {"file": os.path.basename(target), "sha256": _sha256(target),
                       "bytes": os.path.getsize(target)}
        print(f"Packed {name} -> {release_dir}/{files[name]['file']}")

    manifest = {
        "project": "Spotify Track Popularity Reproducibility Pipeline",
        "status": "SUCCESS",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
        "pipeline_version": "1.0.0",
        "summary": {k: metrics.get(k) for k in
                    ("model_type", "accuracy", "macro_f1", "test_samples", "train_samples", "random_state")},
        "files": files,
    }

    if model_info:
        manifest["model"] = model_info

    with open(os.path.join(release_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Pipeline outputs published to {release_dir}/")
    return manifest

if __name__ == "__main__":
    publish_pipeline_outputs()
