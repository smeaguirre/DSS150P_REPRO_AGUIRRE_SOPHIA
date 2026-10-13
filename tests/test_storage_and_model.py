import json

import pandas as pd
import pytest
from sqlalchemy import text

import src.evaluate as evaluate_mod
import src.storage as storage_mod
import src.train as train_mod
from src.transform import build_curated

def _count(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT track_id) FROM curated_spotify_tracks")).one()

def test_load_is_rerun_safe_and_matches_curated_file(loaded_db):
    n_file = len(pd.read_parquet("data/curated/curated_full.parquet"))
    assert tuple(_count(loaded_db)) == (n_file, n_file)
    storage_mod.load_to_postgres(engine=loaded_db)
    storage_mod.load_to_postgres(engine=loaded_db)
    assert tuple(_count(loaded_db)) == (n_file, n_file)

def test_reload_updates_changed_rows_and_removes_stale_rows(loaded_db):
    with loaded_db.begin() as conn:
        conn.execute(text("UPDATE curated_spotify_tracks SET popularity = -1 WHERE track_id = 'T00001'"))
        conn.execute(text("INSERT INTO curated_spotify_tracks (track_id, split) VALUES ('STALE', 'train')"))
    storage_mod.load_to_postgres(engine=loaded_db)
    with loaded_db.connect() as conn:
        assert conn.execute(text("SELECT popularity FROM curated_spotify_tracks WHERE track_id='T00001'")).scalar() != -1
        assert conn.execute(text("SELECT COUNT(*) FROM curated_spotify_tracks WHERE track_id='STALE'")).scalar() == 0

def test_outdated_table_schema_is_rejected_with_clear_message(workdir, db_engine):
    build_curated()
    with db_engine.begin() as conn:
        conn.execute(text("CREATE TABLE curated_spotify_tracks (track_id TEXT, popularity INTEGER)"))
    with pytest.raises(RuntimeError, match="--full-refresh"):
        storage_mod.load_to_postgres(engine=db_engine)
    storage_mod.load_to_postgres(engine=db_engine, full_refresh=True)
    assert _count(db_engine)[0] == len(pd.read_parquet("data/curated/curated_full.parquet"))

def test_read_split_sizes_match_manifest(loaded_db):
    manifest = json.load(open("metadata/split_manifest.json"))
    train, test = storage_mod.read_split("train", loaded_db), storage_mod.read_split("test", loaded_db)
    assert len(train) == manifest["counts"]["train"] and len(test) == manifest["counts"]["test"]
    assert not set(train.track_id) & set(test.track_id)

def test_train_evaluate_end_to_end_is_deterministic(loaded_db):
    train_mod.train_xgboost()
    m1, _ = evaluate_mod.evaluate_model()
    train_mod.train_xgboost()
    m2, _ = evaluate_mod.evaluate_model()

    manifest = json.load(open("metadata/split_manifest.json"))
    assert m1["test_samples"] == manifest["counts"]["test"]
    assert m1["accuracy"] == m2["accuracy"]
    preds = pd.read_csv("outputs/predictions.csv")
    assert len(preds) == m1["test_samples"] and list(preds.columns) == ["track_id", "actual", "predicted"]
    assert {"accuracy", "macro_f1", "balanced_accuracy", "per_class"} <= set(m1)

def test_evaluate_detects_database_out_of_sync_with_manifest(loaded_db):
    train_mod.train_xgboost()
    manifest = json.load(open("metadata/split_manifest.json"))
    manifest["counts"]["test"] += 1
    json.dump(manifest, open("metadata/split_manifest.json", "w"))
    with pytest.raises(RuntimeError, match="out of sync"):
        evaluate_mod.evaluate_model()

def _write_expected(accuracy, tolerance=0.01):
    json.dump({"primary_metric": "accuracy", "metrics": {"accuracy": accuracy}, "tolerance": tolerance},
              open("metadata/expected_metrics.json", "w"))

def test_reproducibility_check_pass_fail_and_pending(workdir):
    json.dump({"accuracy": 0.7144}, open("outputs/metrics.json", "w"))

    _write_expected(0.7150)
    assert evaluate_mod.check_reproducibility()["passed"] is True
    _write_expected(0.80)
    assert evaluate_mod.check_reproducibility()["passed"] is False
    _write_expected(None)
    with pytest.raises(ValueError, match="not set"):
        evaluate_mod.check_reproducibility()
