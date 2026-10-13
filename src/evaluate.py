import os
import json
import xgboost
import pandas as pd
import joblib
import sklearn
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

from src.config import (
    SCALED_FEATURES, TARGET_COL, KEY_COL,
    load_config, get_db_engine, expected_split_size,
)
from src.storage import read_split

def score_test_split(model, X_test, y_test):
    """Evaluate model on test split and return metrics dictionary for testing/audit."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

MODEL_LABELS = {"random_forest": "RandomForest", "xgboost": "XGBoost"}

def _load_model(model_dir, training_config):
    model_type = training_config.get("model_type", "xgboost")
    path = os.path.join(model_dir, training_config.get("model_file", "xgboost_model.json"))
    if model_type == "random_forest":
        return joblib.load(path)
    model = XGBClassifier()
    model.load_model(path)
    return model

def run_evaluation(engine=None):
    config = load_config()
    model_dir = config["paths"]["model_dir"]

    print("Fetching the TEST split from PostgreSQL...")
    test_df = read_split("test", engine or get_db_engine())

    expected = expected_split_size("test", config)
    if expected is not None and len(test_df) != expected:
        raise RuntimeError(
            f"Test split has {len(test_df)} rows but metadata/split_manifest.json says {expected}. "
            f"The database table is out of sync with the curated build (duplicates or a stale load?).")

    with open(os.path.join(model_dir, "training_config.json"), "r") as f:
        training_config = json.load(f)
    classes = training_config["classes"]
    class_index = {c: i for i, c in enumerate(classes)}

    X_test = test_df[SCALED_FEATURES]
    y_test = test_df[TARGET_COL].map(class_index)
    if y_test.isnull().any():
        raise RuntimeError("Test split contains labels the model was not trained on.")
    y_test = y_test.astype(int)

    model = _load_model(model_dir, training_config)

    print("Making predictions on the holdout test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    print("\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))

    metrics = {
        "model_type": MODEL_LABELS.get(training_config.get("model_type", "xgboost"), "XGBoost"),
        "sklearn_version": sklearn.__version__,
        "primary_metric": "accuracy",
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "per_class": {c: {k: float(v) for k, v in report_dict[c].items()} for c in classes},
        "labels": classes,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": int(len(y_test)),
        "train_samples": training_config["train_samples"],
        "random_state": training_config["random_state"],
        "xgboost_version": xgboost.__version__,
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
    }

    predictions = pd.DataFrame({
        KEY_COL: test_df[KEY_COL],
        "actual": test_df[TARGET_COL],
        "predicted": [classes[i] for i in y_pred],
    })
    return metrics, predictions

def evaluate_model():
    config = load_config()
    outputs_dir = config["paths"]["outputs_dir"]
    metrics, predictions = run_evaluation()

    os.makedirs(outputs_dir, exist_ok=True)
    with open(config["reproducibility"]["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=4)
    predictions.to_csv(os.path.join(outputs_dir, "predictions.csv"), index=False)

    print(f"Metrics saved to {config['reproducibility']['metrics_file']}; "
          f"predictions saved to {outputs_dir}/predictions.csv")
    return metrics, predictions

def check_reproducibility(metrics_path=None, expected_path=None, report_path=None, write_report=True):
    config = load_config()
    repro = config["reproducibility"]
    metrics_path = metrics_path or repro["metrics_file"]
    expected_path = expected_path or repro["expected_metrics_file"]
    report_path = report_path or repro["check_report_file"]

    with open(metrics_path, "r") as f:
        actual = json.load(f)
    with open(expected_path, "r") as f:
        expected = json.load(f)

    tolerance = float(expected["tolerance"])
    baseline = {k: v for k, v in expected["metrics"].items() if v is not None}
    primary = expected.get("primary_metric", "accuracy")
    if primary not in baseline:
        raise ValueError(
            f"Baseline metric '{primary}' is not set in {expected_path}. Fill in the previous "
            f"group's reported value (see 'baseline_source') before running the check.")

    checks = []
    for name, expected_value in baseline.items():
        actual_value = actual.get(name)
        if actual_value is None:
            checks.append({"metric": name, "expected": expected_value, "actual": None,
                           "abs_diff": None, "passed": False})
            continue
        diff = abs(float(actual_value) - float(expected_value))
        checks.append({"metric": name, "expected": expected_value, "actual": actual_value,
                       "abs_diff": diff, "passed": diff <= tolerance})

    report = {"tolerance": tolerance, "primary_metric": primary,
              "passed": all(c["passed"] for c in checks), "checks": checks}
    if write_report:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"[{status}] {c['metric']}: expected={c['expected']} actual={c['actual']} "
              f"diff={c['abs_diff']} (tolerance {tolerance})")
    return report

if __name__ == "__main__":
    evaluate_model()