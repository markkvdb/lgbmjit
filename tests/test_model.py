import subprocess, sys, threading
from pathlib import Path

import lightgbm, numpy as np, pandas as pd, pytest
from numpy.testing import assert_array_equal

from lgbmjit import Model
from lgbm_data import booster, edge_data, model_path


def test_arguments():
    "Construction and `predict` arguments behave as in LightGBM; unsupported ones raise."
    path = model_path("NYC_taxi")
    text = Path(path).read_text()
    b, m = booster("NYC_taxi"), Model(model_str=text)
    X = edge_data(b, 1_000)
    want = b.predict(X)
    assert m.model_to_string() == text
    for alias in ("num_threads", "num_thread", "nthread", "nthreads", "n_jobs"): assert_array_equal(m.predict(X, **{alias: 3}), want)
    assert_array_equal(m.predict(X[:3], num_threads=64), want[:3])
    assert m.predict(np.zeros((0, 5))).shape == (0,)
    mc = Model(model_path("multiclass"))
    assert mc.predict(np.zeros((0, mc.num_feature()))).shape == (0, 3)
    df = pd.DataFrame(X, columns=m.feature_name())
    assert_array_equal(m.predict(df, validate_features=True), want)
    with pytest.raises(ValueError, match="feature"): m.predict(df.rename(columns={df.columns[0]: "other"}), validate_features=True)
    with pytest.raises(TypeError): Model()
    with pytest.raises(TypeError): Model(path, model_str=text)
    with pytest.raises(TypeError, match="pred_early_stop"): m.predict(X, pred_early_stop=True)
    with pytest.raises(NotImplementedError): m.predict(X, pred_leaf=True)
    with pytest.raises(NotImplementedError): m.predict(X, pred_contrib=True)
    with pytest.raises(ValueError, match="columns"): m.predict(np.zeros((2, 4)))


def test_concurrent_predictions():
    b, m = booster("airline"), Model(model_path("airline"))
    X = edge_data(b, 2_000)
    want = b.predict(X)
    results = [None] * 4

    def work(i): results[i] = m.predict(X, num_threads=1)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    for r in results: assert_array_equal(r, want)


def test_from_booster_with_early_stopping():
    "`from_booster` keeps every tree, and `predict` defaults to the best iteration as `Booster.predict` does."
    rng = np.random.default_rng(6)
    X, Xv = rng.normal(size=(400, 4)), rng.normal(size=(200, 4))
    y, yv = X[:, 0] + rng.normal(size=400), rng.normal(size=200)
    b = lightgbm.train(dict(objective="regression", verbose=-1, learning_rate=0.3), lightgbm.Dataset(X, y), 50,
        valid_sets=[lightgbm.Dataset(Xv, yv)], callbacks=[lightgbm.early_stopping(3, verbose=False)], keep_training_booster=True)
    assert 0 < b.best_iteration < b.current_iteration()
    m = Model.from_booster(b)
    for kw in [{}, dict(num_iteration=b.current_iteration()), dict(start_iteration=1), dict(num_iteration=2)]:
        assert_array_equal(m.predict(X, **kw), b.predict(X, **kw), err_msg=str(kw))


def test_model_file_is_read_as_utf8(tmp_path):
    "Model files are UTF-8 whatever the locale, as LightGBM writes them."
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(200, 2)), columns=["température", "x"])
    b = lightgbm.train(dict(objective="regression", verbose=-1), lightgbm.Dataset(df, df["x"]), 2)
    path = tmp_path / "model.txt"
    b.save_model(path)
    code = "import lgbmjit, sys; print(lgbmjit.Model(sys.argv[1]).feature_name()[0])"
    env = dict(LC_ALL="en_US.ISO8859-1", LANG="en_US.ISO8859-1", PYTHONUTF8="0", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, encoding="utf-8", env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "température"
