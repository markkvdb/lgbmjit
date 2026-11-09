import json
from pathlib import Path

import pytest
from numpy.testing import assert_array_equal

import lgbmjit._tune
from lgbmjit import Model
from lgbm_data import booster, edge_data, model_path


def test_tuning_keeps_results():
    b = booster("airline")
    X = edge_data(b, 3_000)
    m = Model(model_path("airline")).compile(tune=X)
    assert m.settings["block_trees"] in lgbmjit._tune.GRID["block_trees"]
    assert_array_equal(m.predict(X), b.predict(X))
    with pytest.raises(ValueError): Model(model_path("airline")).compile(tune=X[:0])


def test_tuning_cache(tmp_path):
    "Cached settings are used without tuning; a missing or invalid entry is tuned and stored."
    cache, path = tmp_path / "tune.json", model_path("NYC_taxi")
    key = lgbmjit._tune.cache_key(Path(path).read_text())
    b = booster("NYC_taxi")
    X = edge_data(b, 500)
    # values outside the grid can only come from the cache
    chosen = dict(block_trees=3, tile_rows=5, branch_free_leaves=1, cold_fraction=0.5)
    cache.write_text(json.dumps({key: chosen}))
    m = Model(path).compile(tune=X, tune_cache=cache)
    assert m.settings == chosen
    assert_array_equal(m.predict(X), b.predict(X))
    cache.write_text(json.dumps({key: dict(block_trees="bad")}))
    m = Model(path).compile(tune=X, tune_cache=cache)
    assert m.settings["block_trees"] in lgbmjit._tune.GRID["block_trees"]
    assert json.loads(cache.read_text())[key] == m.settings
    assert Model(path).compile(tune_cache=tmp_path / "missing.json").settings == Model(path).settings
