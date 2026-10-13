import argparse
from src.evaluate import evaluate_model
from src.ingest import run_ingestion
from src.publish import publish_pipeline_outputs
from src.reproducibility_check import verify_reproducibility
from src.storage import load_to_postgres, run_partitioning, run_storage_benchmark
from src.train import train_xgboost
from src.transform import build_curated, transform_staging
from src.validate import validate_data
from src.evidence import generate_all_evidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["ingest", "validate-raw", "transform", "load", "validate-curated", "train", "evaluate", "audit", "publish", "benchmark", "evidence", "all"])
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    parser.add_argument("--full-refresh", action="store_true")
    args = parser.parse_args()

    if args.step in ("ingest", "all"): run_ingestion(mode=args.mode)
    if args.step in ("validate-raw", "all"): validate_data(stage="raw")
    if args.step in ("transform", "all"): transform_staging(); build_curated()
    if args.step in ("load", "all"): load_to_postgres(full_refresh=args.full_refresh)
    if args.step in ("validate-curated", "all"): validate_data(stage="curated")
    if args.step in ("train", "all"): train_xgboost()
    if args.step in ("evaluate", "all"): evaluate_model()
    if args.step in ("audit", "all"): verify_reproducibility()
    if args.step in ("publish", "all"): publish_pipeline_outputs()
    if args.step == "benchmark": run_storage_benchmark(); run_partitioning()
    if args.step == "evidence": generate_all_evidence()

if __name__ == "__main__":
    main()
