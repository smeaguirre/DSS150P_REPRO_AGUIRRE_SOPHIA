import ast
import json
import os
import shutil
import pandas as pd
import pytest
import yaml
from src.ingest import run_ingestion, verify_source_manifest
from src.transform import build_curated, transform_staging
from src.validate import validate_data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    for d in ["data/source", "data/raw", "data/staging", "data/curated", "data/quarantine", "metadata", "config", "outputs"]:
        os.makedirs(tmp_path / d, exist_ok=True)
    
    cfg_src = os.path.join(REPO_ROOT, "config", "settings.yml")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, tmp_path / "config" / "settings.yml")
        
    contract_src = os.path.join(REPO_ROOT, "metadata", "data_contract.yml")
    if os.path.exists(contract_src):
        shutil.copy2(contract_src, tmp_path / "metadata" / "data_contract.yml")
        
    real_raw = os.path.join(REPO_ROOT, "data", "raw", "raw_spotify_data.parquet")
    df = pd.read_parquet(real_raw).head(400).copy()
    df["track_id"] = [f"trk{i:019d}" for i in range(len(df))]
    df.to_parquet(tmp_path / "data" / "raw" / "raw_spotify_data.parquet", index=False)
    df.to_csv(tmp_path / "data" / "source" / "dataset.csv", index=False)
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_CONFIG", str(tmp_path / "config" / "settings.yml"))
    return tmp_path


def test_incremental_ingestion_runs_three_batches_and_is_idempotent(isolated_env):
    wm_path = isolated_env / "metadata" / "ingest_watermark.json"
    raw_path = isolated_env / "data" / "raw" / "raw_spotify_data.parquet"
    if raw_path.exists():
        raw_path.unlink()

    run_ingestion(mode="incremental", force=True)
    with open(wm_path) as f:
        wm1 = json.load(f)
    assert wm1["batches_completed"] == 1
    assert wm1["mode"] == "incremental"
    count1 = len(pd.read_parquet(raw_path))
    assert count1 > 0

    run_ingestion(mode="incremental")
    with open(wm_path) as f:
        wm2 = json.load(f)
    assert wm2["batches_completed"] == 2
    count2 = len(pd.read_parquet(raw_path))
    assert count2 > count1

    run_ingestion(mode="incremental")
    with open(wm_path) as f:
        wm3 = json.load(f)
    assert wm3["batches_completed"] == 3
    count3 = len(pd.read_parquet(raw_path))
    assert count3 == wm3["total_source_rows"]

    res4 = run_ingestion(mode="incremental")
    assert res4["status"] == "SKIPPED_UP_TO_DATE"
    assert len(pd.read_parquet(raw_path)) == count3


def test_source_manifest_detects_checksum_or_row_mismatch(isolated_env):
    src_csv = isolated_env / "data" / "source" / "dataset.csv"
    manifest_file = isolated_env / "metadata" / "source_manifest.yml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"sha256_checksum": "0000badhash", "expected_rows": 400}, f)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_source_manifest(str(src_csv))


def test_validate_enforces_all_five_contract_assertions_and_failure_injection(isolated_env):
    raw_path = isolated_env / "data" / "raw" / "raw_spotify_data.parquet"
    df = pd.read_parquet(raw_path)

    res = validate_data(stage="raw")
    assert res["status"] == "SUCCESS"
    assert res["details"]["assertions_verified"] == 5

    with pytest.raises(ValueError, match="Assertion 1 - Schema"):
        validate_data(stage="raw", inject_failure=True)
    assert (isolated_env / "outputs" / "failure_injection_evidence.json").exists()

    bad_range = df.copy()
    bad_range.loc[0, "danceability"] = 1.5
    bad_range.to_parquet(raw_path, index=False)
    with pytest.raises(ValueError, match="Assertion 4 - Validity/Range"):
        validate_data(stage="raw")

    df.to_parquet(raw_path, index=False)
    transform_staging()
    build_curated()
    res_cur = validate_data(stage="curated")
    assert res_cur["status"] == "SUCCESS"
    assert res_cur["details"]["assertions_verified"] == 5


def test_airflow_dag_has_ten_bash_operators_and_no_empty_operators():
    dag_file = os.path.join(REPO_ROOT, "dags", "predictive_repro_pipeline.py")
    with open(dag_file, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)
    assert "EmptyOperator" not in src
    bash_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "BashOperator"
    ]
    assert len(bash_calls) == 10
