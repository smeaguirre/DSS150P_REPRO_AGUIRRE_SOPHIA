import ast, json, os, shutil, pandas as pd, pytest, yaml
from airflow.models import DagBag
from src.ingest import run_ingestion, verify_source_manifest
from src.transform import build_curated, transform_staging
from src.validate import validate_data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _prep_source_and_contract(workdir):
    os.makedirs(workdir / "data" / "source", exist_ok=True)
    os.makedirs(workdir / "metadata", exist_ok=True)
    os.makedirs(workdir / "data" / "raw", exist_ok=True)
    contract_src = os.path.join(REPO_ROOT, "metadata", "data_contract.yml")
    if os.path.exists(contract_src):
        shutil.copy2(contract_src, workdir / "metadata" / "data_contract.yml")
    df = pd.DataFrame({
        "Unnamed: 0": range(400),
        "track_id": [f"trk{i:019d}" for i in range(400)],
        "popularity": [50] * 400, "duration_ms": [200000] * 400,
        "explicit": [False] * 400, "danceability": [0.5] * 400, "energy": [0.5] * 400,
        "key": [1] * 400, "loudness": [-5.0] * 400, "mode": [1] * 400,
        "speechiness": [0.1] * 400, "acousticness": [0.1] * 400, "instrumentalness": [0.1] * 400,
        "liveness": [0.1] * 400, "valence": [0.5] * 400, "tempo": [120.0] * 400,
        "time_signature": [4] * 400, "track_genre": ["pop"] * 400,
        "artists": ["A"] * 400, "album_name": ["B"] * 400, "track_name": ["C"] * 400
    })
    df.to_parquet(workdir / "data" / "raw" / "raw_spotify_data.parquet", index=False)
    df.to_csv(workdir / "data" / "source" / "dataset.csv", index=False)
    return df

@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    for d in ["data/source", "data/raw", "data/staging", "data/curated", "data/quarantine", "metadata", "config", "outputs"]:
        os.makedirs(tmp_path / d, exist_ok=True)
    cfg_src = os.path.join(REPO_ROOT, "config", "settings.yml")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, tmp_path / "config" / "settings.yml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_CONFIG", str(tmp_path / "config" / "settings.yml"))
    return tmp_path

def test_incremental_ingestion_runs_three_batches_and_is_idempotent(isolated_env):
    _prep_source_and_contract(isolated_env)
    wm_path = isolated_env / "metadata" / "ingest_watermark.json"
    raw_path = isolated_env / "data" / "raw" / "raw_spotify_data.parquet"
    raw_path.unlink()

    run_ingestion(mode="incremental", force=True)
    with open(wm_path) as f: wm1 = json.load(f)
    assert wm1["batches_completed"] == 1
    
    run_ingestion(mode="incremental")
    with open(wm_path) as f: wm2 = json.load(f)
    assert wm2["batches_completed"] == 2
    
    run_ingestion(mode="incremental")
    with open(wm_path) as f: wm3 = json.load(f)
    assert wm3["batches_completed"] == 3
    
    res4 = run_ingestion(mode="incremental")
    assert res4["status"] == "SKIPPED_UP_TO_DATE"

def test_ingest_full_guard_raises_error(isolated_env):
    _prep_source_and_contract(isolated_env)
    run_ingestion(mode="full", force=True)
    with pytest.raises(ValueError, match="Existing full mode watermark found"):
        run_ingestion(mode="incremental")

def test_source_manifest_detects_checksum_or_row_mismatch(isolated_env):
    _prep_source_and_contract(isolated_env)
    manifest_file = isolated_env / "metadata" / "source_manifest.yml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"sha256_checksum": "0000badhash", "expected_rows": 400}, f)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_source_manifest(str(isolated_env / "data" / "source" / "dataset.csv"))

def test_validate_enforces_all_five_contract_assertions_and_failure_injection(isolated_env):
    df = _prep_source_and_contract(isolated_env)
    raw_path = isolated_env / "data" / "raw" / "raw_spotify_data.parquet"
    
    res = validate_data(stage="raw")
    assert res["status"] == "SUCCESS"
    assert res["details"]["assertions_verified"] == 5

    with pytest.raises(ValueError, match="Assertion 1 - Schema"):
        validate_data(stage="raw", inject_failure=True)
    assert (isolated_env / "outputs" / "failure_injection_evidence.json").exists()

    df.loc[0, "danceability"] = 1.5
    df.to_parquet(raw_path, index=False)
    with pytest.raises(ValueError, match="Assertion 4 - Validity/Range"):
        validate_data(stage="raw")
        
    df.loc[0, "danceability"] = 0.5
    df.to_parquet(raw_path, index=False)
    transform_staging()
    build_curated()
    res_cur = validate_data(stage="curated")
    assert res_cur["status"] == "SUCCESS"
    assert res_cur["details"]["assertions_verified"] == 5

def test_airflow_dag_has_ten_bash_operators_and_no_empty_operators():
    pytest.importorskip('airflow')
    dagbag = DagBag(dag_folder=os.path.join(REPO_ROOT, "dags"), include_examples=False)
    dag = dagbag.get_dag("predictive_repro_pipeline")
    assert dag is not None
    assert len(dag.tasks) == 10
    for task in dag.tasks:
        assert task.task_type == "BashOperator"
