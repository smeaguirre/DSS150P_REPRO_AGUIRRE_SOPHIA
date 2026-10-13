import json, os
from datetime import datetime, timezone

def verify_reproducibility():
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/metrics.json", "r", encoding="utf-8") as f: 
        live_metrics = json.load(f)
        
    with open("metadata/expected_metrics.json", "r", encoding="utf-8") as f: 
        exp_metrics = json.load(f)
        
    exp_acc = exp_metrics.get("metrics", {}).get("accuracy", 0.624861)
    live_acc = live_metrics.get("accuracy", 0.0)
    
    diff = abs(live_acc - exp_acc)
    passed = bool(diff <= 1e-4)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS" if passed else "FAILED",
        "passed": passed,
        "baseline_check": {
            "expected_accuracy": exp_acc,
            "live_accuracy": live_acc,
            "difference": diff,
            "tolerance": 1e-4,
            "passed": passed
        },
        "historical_notebook_comparison": {
            "historical_notebook_metric": 0.7145,
            "outcome_classification": "Unexplained Baseline Deviation - Instructor Approval Pending",
            "explanation": "ROC-AUC (~0.74) is a candidate for the historical metric, but strict multiclass accuracy is ~0.62. The 0.7145 figure remains exactly unexplained."
        }
    }
    with open("outputs/reproducibility_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    return report

if __name__ == "__main__":
    verify_reproducibility()
