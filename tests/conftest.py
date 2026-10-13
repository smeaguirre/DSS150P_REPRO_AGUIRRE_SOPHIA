import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[1]

def make_staging_df(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    popularity = np.concatenate([
        rng.integers(0, 34, n // 2),
        rng.integers(34, 67, n // 2 - n // 20),
        rng.integers(67, 101, n - n // 2 - (n // 2 - n // 20)),
    ])
    rng.shuffle(popularity)
    return pd.DataFrame({
        "track_id": [f"T{i:05d}" for i in range(n)],
        "artists": "artist", "album_name": "album", "track_name": "track",
        "popularity": popularity,
        "duration_ms": rng.integers(60_000, 400_000, n),
        "explicit": rng.random(n) > 0.8,
        "danceability": rng.random(n), "energy": rng.random(n),
        "key": rng.integers(0, 12, n), "loudness": rng.normal(-8, 4, n),
        "mode": rng.integers(0, 2, n), "speechiness": rng.random(n),
        "acousticness": rng.random(n), "instrumentalness": rng.random(n),
        "liveness": rng.random(n), "valence": rng.random(n),
        "tempo": rng.normal(120, 25, n), "time_signature": rng.integers(3, 6, n),
        "track_genre": "pop",
        "_etl_loaded_at": pd.Timestamp("2026-01-01"), "_ingest_mode": "full",
        "_staged_at": pd.Timestamp("2026-01-02"), "_staging_row_id": np.arange(n),
    })

@pytest.fixture
def workdir(tmp_path, monkeypatch):
    for sub in ("config", "sql"):
        shutil.copytree(REPO_ROOT / sub, tmp_path / sub)
    for sub in ("metadata", "data/staging", "data/curated", "outputs/model"):
        (tmp_path / sub).mkdir(parents=True)
    make_staging_df().to_parquet(tmp_path / "data/staging/staging_spotify_tracks.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PIPELINE_RANDOM_STATE", raising=False)
    return tmp_path

@pytest.fixture
def db_engine(tmp_path):
    url = os.getenv("TEST_DATABASE_URL")
    engine = create_engine(url or f"sqlite:///{tmp_path / 'test.db'}")

    def _clean():
        with engine.begin() as conn:
            for table in ("curated_spotify_tracks", "_stg_curated_load"):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))

    if url:
        _clean()
    yield engine
    if url:
        _clean()
    engine.dispose()

@pytest.fixture
def loaded_db(workdir, db_engine, monkeypatch):
    import src.evaluate as evaluate_mod
    import src.storage as storage_mod
    import src.train as train_mod
    from src.transform import build_curated

    for mod in (storage_mod, train_mod, evaluate_mod):
        monkeypatch.setattr(mod, "get_db_engine", lambda: db_engine)
    build_curated()
    storage_mod.load_to_postgres(engine=db_engine)
    return db_engine
