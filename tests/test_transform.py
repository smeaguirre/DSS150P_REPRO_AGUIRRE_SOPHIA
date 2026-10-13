import json

import pandas as pd
import pytest

from src.transform import build_curated, discretize_popularity

def test_discretize_popularity_boundaries():
    assert [discretize_popularity(v) for v in (0, 33, 34, 66, 67, 100)] == \
           ["Low", "Low", "Mid", "Mid", "High", "High"]

def test_build_curated_split_has_no_overlap_and_is_sorted(workdir):
    build_curated()
    train = pd.read_parquet("data/curated/curated_train.parquet")
    test = pd.read_parquet("data/curated/curated_test.parquet")
    full = pd.read_parquet("data/curated/curated_full.parquet")

    assert not set(train.track_id) & set(test.track_id)
    assert train.track_id.is_monotonic_increasing and test.track_id.is_monotonic_increasing
    assert full.track_id.is_unique and len(full) == len(train) + len(test)
    assert set(train["split"]) == {"train"} and set(test["split"]) == {"test"}

    manifest = json.load(open("metadata/split_manifest.json"))
    assert manifest["counts"]["train"] == len(train)
    assert manifest["counts"]["test"] == len(test)

def test_scaler_is_fit_on_train_only(workdir):
    build_curated()
    train = pd.read_parquet("data/curated/curated_train.parquet")
    test = pd.read_parquet("data/curated/curated_test.parquet")
    assert abs(train["scaled_loudness"].mean()) < 1e-9
    assert abs(test["scaled_loudness"].mean()) > 1e-9

def test_build_curated_is_deterministic(workdir):
    build_curated()
    first = json.load(open("metadata/split_manifest.json"))["checksums"]
    build_curated()
    assert json.load(open("metadata/split_manifest.json"))["checksums"] == first

def test_build_curated_rejects_duplicate_track_ids(workdir):
    path = "data/staging/staging_spotify_tracks.parquet"
    df = pd.read_parquet(path)
    pd.concat([df, df.head(5)]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        build_curated()

def test_seed_env_override_changes_split(workdir, monkeypatch):
    build_curated()
    a = json.load(open("metadata/split_manifest.json"))["checksums"]["test_checksum"]
    monkeypatch.setenv("PIPELINE_RANDOM_STATE", "7")
    build_curated()
    manifest = json.load(open("metadata/split_manifest.json"))
    assert manifest["random_state"] == 7 and manifest["checksums"]["test_checksum"] != a
