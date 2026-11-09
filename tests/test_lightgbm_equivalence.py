from pathlib import Path

import lightgbm, numpy as np, pandas as pd, pytest
from numpy.testing import assert_array_equal

from lgbmjit import Model, _lgbmjit

from lgbm_data import MODEL_NAMES, booster, edge_data, model_path


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bundled_models(name, dtype):
    b = booster(name)
    rows = 300 if b.num_trees() > 500 else 2_000
    X = edge_data(b, rows).astype(dtype)
    m = Model(model_path(name))
    assert_array_equal(m.predict(X), b.predict(X))
    assert_array_equal(m.predict(X, raw_score=True), b.predict(X, raw_score=True))
    iterations = b.num_trees() // b.num_model_per_iteration()
    if iterations > 2:
        kw = dict(start_iteration=1, num_iteration=2)
        assert_array_equal(m.predict(X, **kw), b.predict(X, **kw))


@pytest.mark.parametrize("name", ["NYC_taxi", "airline", "mixed_categorical"])
def test_input_layouts(name):
    "Any NumPy memory layout gives LightGBM's result: views, strides, odd byte strides and unaligned data."
    b, m = booster(name), Model(model_path(name))
    X = edge_data(b, 500)
    want = b.predict(X)
    wide = np.empty((X.shape[0], 2 * X.shape[1]))
    wide[:, ::2] = X
    assert_array_equal(m.predict(np.asfortranarray(X)), want)
    assert_array_equal(m.predict(wide[:, ::2]), want)
    assert_array_equal(m.predict(X[::-1])[::-1], want)
    assert_array_equal(m.predict(X[::3]), want[::3])
    # A structured array's column: a byte stride that is not a whole number of float64s
    S = np.zeros(len(X), dtype=[("x", "f8", X.shape[1]), ("pad", "i4")])
    S["x"] = X
    assert S["x"].strides[0] % 8
    assert_array_equal(m.predict(S["x"]), want)
    buf = bytearray(X.nbytes + 1)
    unaligned = np.frombuffer(buf, dtype=np.float64, offset=1, count=X.size).reshape(X.shape)
    unaligned[:] = X
    assert not unaligned.flags.aligned
    assert_array_equal(m.predict(unaligned), want)


def test_extension_rejects_unreadable_arrays():
    "The extension module itself refuses arrays it cannot read element by element, and the wrong dtype."
    f = _lgbmjit.Forest(Path(model_path("NYC_taxi")).read_text())
    c = f.compile("float64", 0, 0, **_lgbmjit.default_settings())
    odd = np.lib.stride_tricks.as_strided(np.zeros(2_000), shape=(10, 5), strides=(12, 4))
    with pytest.raises(ValueError, match="stride"): c.predict(odd, False, 1)
    with pytest.raises(TypeError): c.predict(np.zeros((2, 5), np.float32), False, 1)


def _train(params, X, y, rounds=5, **dataset_kw):
    params = {"verbose": -1, "num_leaves": 8, "min_data_in_leaf": 5, **params}
    return lightgbm.train(params, lightgbm.Dataset(X, y, **dataset_kw), rounds)


def _check(b, X, **kw):
    m = Model.from_booster(b)
    assert_array_equal(m.predict(X, **kw), b.predict(X, **kw))


def _regression_data(rows=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 4))
    y = X[:, 0] * 2 + np.sin(X[:, 1]) + rng.normal(scale=0.1, size=rows)
    return X, y


@pytest.mark.parametrize(
    "objective",
    [
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "quantile",
        "mape",
        "poisson",
        "gamma",
        "tweedie",
    ])
@pytest.mark.parametrize("reg_sqrt", [False, True])
def test_regression_objectives(objective, reg_sqrt):
    X, y = _regression_data()
    if objective in ("poisson", "gamma", "tweedie", "mape"): y = np.abs(y) + 0.5
    b = _train({"objective": objective, "reg_sqrt": reg_sqrt}, X, y)
    _check(b, X)
    _check(b, X, raw_score=True)


@pytest.mark.parametrize("objective", ["binary", "cross_entropy", "cross_entropy_lambda"])
def test_probability_objectives(objective):
    X, y = _regression_data()
    b = _train({"objective": objective, "sigmoid": 1.7}, X, (y > 0).astype(float))
    _check(b, X)
    _check(b, X, raw_score=True)


@pytest.mark.parametrize("objective", ["multiclass", "multiclassova"])
def test_multiclass_objectives(objective):
    X, y = _regression_data()
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    b = _train({"objective": objective, "num_class": 3}, X, labels)
    _check(b, X)
    _check(b, X, raw_score=True)
    _check(b, X, start_iteration=1, num_iteration=3)


def test_lambdarank():
    X, y = _regression_data(600)
    relevance = np.digitize(y, np.quantile(y, [0.5, 0.8]))
    b = _train({"objective": "lambdarank"}, X, relevance, group=[30] * 20)
    _check(b, X)


@pytest.mark.parametrize("num_class", [1, 3])
@pytest.mark.parametrize("raw_score", [False, True])
def test_random_forest(num_class, raw_score):
    X, y = _regression_data()
    params = dict(boosting="rf", bagging_freq=1, bagging_fraction=0.8)
    if num_class == 1:
        params["objective"] = "binary"
        y = (y > 0).astype(float)
    else:
        params.update(objective="multiclass", num_class=3)
        y = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    b = _train(params, X, y, rounds=7)
    _check(b, X, raw_score=raw_score)
    _check(b, X, raw_score=raw_score, num_iteration=3)


@pytest.mark.parametrize("zero_as_missing", [False, True])
@pytest.mark.parametrize("use_missing", [False, True])
def test_missing_values(zero_as_missing, use_missing):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2_000, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[rng.random(X.shape) < 0.2] = 0.0
    y = np.nan_to_num(X[:, 0], nan=3.0) + (X[:, 1] == 0) * 2 + rng.normal(scale=0.1, size=2_000)
    b = _train(dict(objective='regression', zero_as_missing=zero_as_missing, use_missing=use_missing), X, y, 10)
    types = {int(d) >> 2 & 3 for line in b.model_to_string().splitlines() if line.startswith('decision_type=') for d in line.split('=')[1].split()}
    expected = 0 if not use_missing else (1 if zero_as_missing else 2)
    assert expected in types, f"no node with missing type {expected}: {types}"
    T = rng.choice([np.nan, 0.0, -0.0, 1e-36, -1e-36, 1e-35, -1e-35, 0.5, -0.5], size=(500, 3))
    _check(b, T)
    _check(b, T.astype(np.float32))
    _check(b, X)


def test_categorical_numpy():
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.integers(0, 100, 3_000), rng.normal(size=3_000)]).astype(float)
    y = (X[:, 0] % 7 == 0) * 3.0 + X[:, 1] + rng.normal(scale=0.1, size=3_000)
    b = _train(dict(objective='regression', max_cat_to_onehot=1, cat_smooth=1), X, y, 10, categorical_feature=[0])
    T = X.copy()
    T[:50, 0] = [-0.5, -1.0, 1e10, np.nan, 0.5, 99.9, 150.0, -1e10, 31.0, 32.0] * 5
    _check(b, T)
    _check(b, T.astype(np.float32))


def test_pandas_categorical():
    rng = np.random.default_rng(5)
    cats = [f"c{i}" for i in range(40)]
    df = pd.DataFrame(dict(a=pd.Categorical(rng.choice(cats, 2000), categories=cats), x=rng.normal(size=2000),
        b=pd.Categorical(rng.choice(['p', 'q', 'r'], 2000))))
    y = df["a"].cat.codes.to_numpy() % 5 + (df["b"] == "q") * 2.0 + df["x"].to_numpy()
    b = _train({"objective": "regression"}, df, y, 10)
    _check(b, df)
    test = df.copy()
    test["a"] = pd.Categorical(test["a"].astype(str).where(rng.random(2_000) > 0.1, "unseen"))
    test["b"] = test["b"].cat.reorder_categories(["r", "q", "p"])
    _check(b, test)
