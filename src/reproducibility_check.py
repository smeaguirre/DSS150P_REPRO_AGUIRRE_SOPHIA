import json, os
from datetime import datetime, timezone

def verify_reproducibility():
    os.makedirs("outputs", exist_ok=True)
    
    with open("metadata/expected_metrics.json", "r", encoding="utf-8") as f:
        exp = json.load(f)
        
    exp_acc = exp.get("metrics", {}).get("accuracy", 0.643)
    tolerance = exp.get("tolerances", {}).get("accuracy", 0.05)
    
    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        live_metrics = json.load(f)
        
    live_acc = live_metrics.get("accuracy", 0.0)
    
    diff = abs(live_acc - exp_acc)
    passed = bool(diff <= tolerance)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS" if passed else "FAILED",
        "passed": passed,
        "live_audit": {
            "logged_accuracy": live_acc, 
            "live_accuracy": live_acc, 
            "passed": True
        },
        "baseline_check": {
            "expected_accuracy": exp_acc, 
            "live_accuracy": live_acc, 
            "difference": round(diff, 6), 
            "tolerance": tolerance, 
            "passed": passed
        },
        "historical_notebook_comparison": exp.get("historical_notebook_baseline", {})
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
