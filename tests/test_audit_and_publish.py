import hashlib
import json
import os

import pytest

import src.evaluate as evaluate_mod
import src.publish as publish_mod
import src.reproducibility_check as audit_mod
import src.train as train_mod

def _train_and_evaluate(baseline_accuracy="actual", tolerance=0.01):
    train_mod.train_xgboost()
    metrics, _ = evaluate_mod.evaluate_model()
    if baseline_accuracy == "actual":
        baseline_accuracy = metrics["accuracy"]
    json.dump({"primary_metric": "accuracy", "metrics": {"accuracy": baseline_accuracy},
               "tolerance": tolerance}, open("metadata/expected_metrics.json", "w"))
    return metrics

def test_audit_passes_when_artifacts_and_baseline_agree(loaded_db):
    _train_and_evaluate()
    report = audit_mod.verify_reproducibility()
    assert report["passed"] is True
    assert report["live_audit"]["passed"] and report["baseline_check"]["passed"]
    assert json.load(open("outputs/reproducibility_check.json"))["passed"] is True

def test_audit_detects_tampered_metrics_file(loaded_db):
    metrics = _train_and_evaluate()
    tampered = dict(metrics, accuracy=metrics["accuracy"] + 0.05)
    json.dump(tampered, open("outputs/metrics.json", "w"))
    json.dump({"primary_metric": "accuracy", "metrics": {"accuracy": tampered["accuracy"]},
               "tolerance": 0.01}, open("metadata/expected_metrics.json", "w"))
    report = audit_mod.verify_reproducibility()
    assert report["live_audit"]["passed"] is False and report["passed"] is False

def test_audit_fails_when_result_differs_from_baseline(loaded_db):
    _train_and_evaluate(baseline_accuracy=0.99)
    report = audit_mod.verify_reproducibility()
    assert report["live_audit"]["passed"] is True
    assert report["baseline_check"]["passed"] is False and report["passed"] is False

def test_audit_errors_while_baseline_is_unset(loaded_db):
    _train_and_evaluate(baseline_accuracy=None)
    with pytest.raises(ValueError, match="not set"):
        audit_mod.verify_reproducibility()

def test_publish_refuses_without_or_with_failed_audit(loaded_db):
    _train_and_evaluate()
    with pytest.raises(FileNotFoundError):
        publish_mod.publish_pipeline_outputs()
    audit_mod.verify_reproducibility()
    report = json.load(open("outputs/reproducibility_check.json"))
    report["passed"] = False
    json.dump(report, open("outputs/reproducibility_check.json", "w"))
    with pytest.raises(RuntimeError, match="did not pass"):
        publish_mod.publish_pipeline_outputs()

def test_publish_builds_verifiable_bundle_and_is_rerun_safe(loaded_db, monkeypatch):
    monkeypatch.setenv("PIPELINE_RUN_ID", "manual__test")
    _train_and_evaluate()
    audit_mod.verify_reproducibility()

    publish_mod.publish_pipeline_outputs()
    release = "outputs/final_release"
    manifest = json.load(open(f"{release}/manifest.json"))
    assert manifest["status"] == "SUCCESS" and manifest["pipeline_run_id"] == "manual__test"
    for entry in manifest["files"].values():
        data = open(os.path.join(release, entry["file"]), "rb").read()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]

    open(f"{release}/stale.txt", "w").write("left over from an old run")
    publish_mod.publish_pipeline_outputs()
    assert not os.path.exists(f"{release}/stale.txt")
    assert len(os.listdir(release)) == len(manifest["files"]) + 1
