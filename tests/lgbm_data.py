"Bundled models and edge-case data for comparing lgbmjit with LightGBM."

from pathlib import Path

import lightgbm, numpy as np

MODELS = Path(__file__).parent / "models"
MODEL_NAMES = sorted(p.name for p in MODELS.iterdir() if (p / "model.txt").exists())


def model_path(name): return str(MODELS / name / "model.txt")


def _thresholds(booster):
    'Numerical thresholds per feature and the set of categorical features of `booster`.'
    model = booster.dump_model()
    nf = booster.num_feature()
    thresholds = [[] for _ in range(nf)]
    categorical = set()

    def walk(node):
        if "split_feature" not in node: return
        f = node["split_feature"]
        if node["decision_type"] == "==": categorical.add(f)
        else: thresholds[f].append(node["threshold"])
        walk(node["left_child"])
        walk(node["right_child"])

    for tree in model["tree_info"]: walk(tree["tree_structure"])
    return thresholds, categorical


def edge_data(booster, rows, seed=0):
    "Random rows around the model's thresholds, with missing values, zeros and odd categories."
    rng = np.random.default_rng(seed)
    thresholds, categorical = _thresholds(booster)
    X = np.empty((rows, len(thresholds)))
    for f, ts in enumerate(thresholds):
        if f in categorical:
            col = rng.integers(0, 64, rows) + rng.random(rows) * 0.9
            odd = rng.random(rows) < 0.2
            col[odd] = rng.choice([-0.5, -3.0, 1e10, np.nan], odd.sum())
        else:
            ts = np.array(ts or [0.0])
            lo, hi = ts.min() - 1, ts.max() + 1
            col = lo + rng.random(rows) * (hi - lo)
            near = rng.random(rows) < 0.4
            picks = rng.choice(ts, near.sum())
            col[near] = np.where(rng.random(near.sum()) < 0.5, picks, np.nextafter(picks, np.inf))
            special = rng.random(rows) < 0.1
            col[special] = rng.choice([0.0, -0.0, 1e-36, -1e-36, 1e-35], special.sum())
        col[rng.random(rows) < 0.1] = np.nan
        X[:, f] = col
    return X


def booster(name): return lightgbm.Booster(model_file=model_path(name))
