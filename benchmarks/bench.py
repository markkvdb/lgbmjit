"""Compare lgbmjit with LightGBM on a model and data: speed and identical outputs.

Usage: python benchmarks/bench.py MODEL.txt X.npy [--expected E.npy] [--rows N]
                                  [--threads T] [--runs R] [--tune]
"""

import argparse, time

import lightgbm, numpy as np

import lgbmjit


def best_of(f, runs):
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = f()
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("data", help=".npy file with one row per prediction")
    p.add_argument("--expected", help=".npy file with LightGBM's predictions for the same rows")
    p.add_argument("--rows", type=int, default=None, help="use the first ROWS rows")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tune", action="store_true", help="tune settings on the first 20,000 rows")
    a = p.parse_args()

    X = np.load(a.data, mmap_mode="r")
    X = np.ascontiguousarray(X[: a.rows] if a.rows else X)
    n = X.shape[0]
    shape = f"{n} rows x {X.shape[1]} features, {X.dtype}"
    print(f"{shape}, {a.threads} thread(s), lgbmjit {lgbmjit.__version__}")

    booster = lightgbm.Booster(model_file=a.model)
    t_lgb, want = best_of(lambda: booster.predict(X, num_threads=a.threads), a.runs)
    print(f"lightgbm {lightgbm.__version__}: {t_lgb / n * 1e6:8.3f} s per million rows")

    t0 = time.perf_counter()
    model = lgbmjit.Model(a.model).compile(tune=X if a.tune else None)
    t_compile = time.perf_counter() - t0
    t_jit, got = best_of(lambda: model.predict(X, num_threads=a.threads), a.runs)
    rate, speedup = t_jit / n * 1e6, t_lgb / t_jit
    print(f"lgbmjit:          {rate:8.3f} s per million rows, {speedup:5.1f} times faster")
    print(f"lgbmjit compile:  {t_compile:8.3f} s, settings {model.settings}")
    print(f"identical to lightgbm: {np.array_equal(got, want)}")
    if a.expected:
        expected = np.load(a.expected)[:n]
        print(f"identical to {a.expected}: {np.array_equal(got, expected)}")


if __name__ == "__main__": main()
