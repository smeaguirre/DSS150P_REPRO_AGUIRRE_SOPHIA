import json, os
from datetime import datetime, timezone
from src.evaluate import score_test_split

def verify_reproducibility(engine=None):
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists("metadata/expected_metrics.json"): raise ValueError("baseline not set")
    with open("metadata/expected_metrics.json", "r", encoding="utf-8") as f: exp = json.load(f)
    
    exp_acc = exp.get("metrics", {}).get("accuracy", 0.624861)
    tolerance = exp.get("tolerances", {}).get("accuracy", 0.01)
    
    live_metrics, _ = score_test_split(engine=engine)
    live_acc = live_metrics["accuracy"]
    
    if not os.path.exists("outputs/metrics.json"): raise ValueError("metrics.json missing")
    with open("outputs/metrics.json", "r", encoding="utf-8") as f: logged = json.load(f)
    if abs(live_acc - logged.get("accuracy", 0)) > 1e-6: raise ValueError("Tampering detected")

    diff = abs(live_acc - exp_acc)
    passed = bool(diff <= tolerance)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS" if passed else "FAILED",
        "passed": passed,
        "baseline_check": {"expected_accuracy": exp_acc, "live_accuracy": live_acc, "difference": diff, "tolerance": tolerance, "passed": passed},
        "historical_notebook_comparison": exp.get("historical_notebook_baseline", {})
    }
    with open("outputs/reproducibility_check.json", "w", encoding="utf-8") as f: json.dump(report, f, indent=4)
    return report

if __name__ == "__main__": verify_reproducibility()
