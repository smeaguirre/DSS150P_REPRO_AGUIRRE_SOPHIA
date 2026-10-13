import os
import sys
import json
from datetime import datetime, timezone

from src.config import load_config
from src.evaluate import run_evaluation, check_reproducibility

def verify_reproducibility():
    config = load_config()
    repro = config["reproducibility"]
    metrics_path = repro["metrics_file"]
    report_path = repro["check_report_file"]

    print("Running Reproducibility Audit...")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{metrics_path} not found. Run evaluation first.")
    with open(metrics_path, "r") as f:
        logged_accuracy = float(json.load(f)["accuracy"])

    live_metrics, _ = run_evaluation()
    live_accuracy = float(live_metrics["accuracy"])
    audit_tolerance = float(repro.get("audit_tolerance", 1e-6))
    audit_diff = abs(live_accuracy - logged_accuracy)
    audit = {"logged_accuracy": logged_accuracy, "live_accuracy": live_accuracy,
             "abs_diff": audit_diff, "tolerance": audit_tolerance,
             "passed": audit_diff <= audit_tolerance}
    print(f"[{'PASS' if audit['passed'] else 'FAIL'}] live audit: logged={logged_accuracy:.6f} "
          f"live={live_accuracy:.6f} diff={audit_diff:.2e}")

    baseline = check_reproducibility(write_report=False)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
        "passed": bool(audit["passed"] and baseline["passed"]),
        "live_audit": audit,
        "baseline_check": baseline,
    }
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("REPRODUCIBILITY AUDIT PASSED" if report["passed"] else "REPRODUCIBILITY AUDIT FAILED")
    return report

if __name__ == "__main__":
    try:
        sys.exit(0 if verify_reproducibility()["passed"] else 1)
    except (ValueError, FileNotFoundError) as exc:
        print(f"REPRODUCIBILITY AUDIT ERROR: {exc}")
        sys.exit(1)


# --- AUTOMATED WRAPPER ---
import json
_orig_verify = verify_reproducibility
def verify_reproducibility(*args, **kwargs):
    report = _orig_verify(*args, **kwargs)
    try:
        with open('metadata/expected_metrics.json', 'r', encoding='utf-8') as f:
            exp = json.load(f)
        if isinstance(report, dict):
            report['historical_notebook_comparison'] = exp.get('historical_notebook_baseline', {})
            with open('outputs/reproducibility_check.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
    except Exception:
        pass
    return report
