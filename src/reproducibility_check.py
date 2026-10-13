import json, os
from datetime import datetime, timezone
from src.evaluate import run_evaluation

def verify_reproducibility(engine=None):
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists("metadata/expected_metrics.json"):
        raise ValueError("baseline not set")
    with open("metadata/expected_metrics.json", "r", encoding="utf-8") as f:
        exp = json.load(f)

    exp_acc = exp.get("metrics", {}).get("accuracy")
    tolerance = exp.get("tolerance", exp.get("tolerances", {}).get("accuracy"))
    if exp_acc is None or tolerance is None:
        raise ValueError("baseline not set")

    if not os.path.exists("outputs/metrics.json"):
        raise ValueError("metrics.json missing")
    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        logged = json.load(f)

    live_metrics, _ = run_evaluation(engine=engine)
    live_acc = live_metrics["accuracy"]
    logged_acc = logged.get("accuracy")

    live_diff = None if logged_acc is None else abs(live_acc - logged_acc)
    live_passed = bool(live_diff is not None and live_diff <= 1e-6)

    diff = abs(live_acc - exp_acc)
    baseline_passed = bool(diff <= tolerance)
    passed = bool(live_passed and baseline_passed)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS" if passed else "FAILED",
        "passed": passed,
        "live_audit": {"logged_accuracy": logged_acc, "live_accuracy": live_acc,
                       "difference": live_diff, "passed": live_passed},
        "baseline_check": {"expected_accuracy": exp_acc, "live_accuracy": live_acc,
                           "difference": round(diff, 6), "tolerance": tolerance,
                           "passed": baseline_passed},
        "historical_notebook_comparison": exp.get("historical_notebook_baseline", {}),
    }
    with open("outputs/reproducibility_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    try:
        from src.config import load_config, update_pipeline_run_summary
        update_pipeline_run_summary(load_config(), "reproducibility_check", report["status"])
    except Exception:
        pass
    return report

if __name__ == "__main__":
    verify_reproducibility()