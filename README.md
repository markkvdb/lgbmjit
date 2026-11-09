# lgbmjit

`lgbmjit` compiles trained LightGBM models to machine code and predicts with them. Its predictions are bit-identical to `lightgbm.Booster.predict`.

I wrote it for my own batch forecasting, where LightGBM prediction took most of the run time. On my production model (500 trees of up to 255 leaves, 82 features), `lgbmjit` predicts between 18 and 25 times faster than LightGBM on one thread.

`lgbmjit` builds on the idea of [lleaves](https://github.com/siboehm/lleaves), which compiles LightGBM models with LLVM. Compared with lleaves:

- It compiles in seconds instead of minutes. My model compiles in 1.6 s, where lleaves took 6 minutes on the same Mac and needed about 14 GB of memory on an x86 machine.
- It depends only on NumPy. The compiler, [Cranelift](https://cranelift.dev), is built into the wheel, so there is no llvmlite or LLVM to install.
- It predicts faster: 0.69 s per million rows against 0.97 s for lleaves on the same model and machine.
- You can tune the generated code for your machine and your data, and cache the result.

## Install

```bash
pip install lgbmjit
```

Wheels are built for Linux (x86-64, aarch64), macOS (arm64) and Windows (x86-64), for Python 3.11 and later. The only runtime dependency is NumPy 1.24 or later.

## Example

```python
import lightgbm as lgb
import lgbmjit

booster = lgb.train(params, lgb.Dataset(X_train, y_train))
model = lgbmjit.Model.from_booster(booster)    # or lgbmjit.Model("model.txt")

predictions = model.predict(X)                 # same result as booster.predict(X)
```

`predict` takes the arguments of `Booster.predict`: `start_iteration`, `num_iteration`, `raw_score`, `validate_features` and `num_threads`. `X` can be a NumPy array, a list of rows or a pandas DataFrame, including categorical columns. The model compiles on the first `predict` for float32 or float64 input, which took 1 to 3 seconds for my model.

To pick the fastest code layout for your machine, compile with a sample of your data:

```python
model = lgbmjit.Model("model.txt").compile(tune=X[:20_000], tune_cache="lgbmjit-tuning.json")
```

Tuning compiles the model several times and times each version, which took 14 seconds on my Mac and 30 seconds on an older x86 machine. The cache file keeps the chosen settings per model and CPU, so later runs skip the timing. Results never depend on the settings, only speed does.

## Speed

One thread, 192,000 rows of float32 data with 11% missing values, on the model above:

| Machine | LightGBM 4.7 | lgbmjit | Speedup | Compile |
|---|---:|---:|---:|---:|
| Apple M5 Pro | 12.9 s | 0.69 s | 18.6 times | 1.6 s |
| AMD Zen 1 (virtual machine) | 38.1 s | 1.65 s | 23.1 times | 5.7 s |

Times are seconds per million rows. Every prediction was identical to LightGBM's. `benchmarks/bench.py` runs the same comparison on your own model and data.

## What is supported

`lgbmjit` handles numerical splits with every missing-value setting, categorical splits, and the output of every built-in objective, including multiclass, ranking and random forests. It does not train models, and it raises an error for linear trees, `pred_leaf=True` and `pred_contrib=True`. It never silently returns results that differ from LightGBM's.

## Development

```bash
uv run --extra test pytest tests   # build in place and run the Python tests
uv run tasks.py check              # formatting, style, types, lints and Rust tests (also the pre-commit hook)
uv run tasks.py build              # a release wheel for this machine into dist/
uv run tasks.py test               # test that wheel with the newest dependencies and with the oldest supported ones
uv run release.py v0.2.0 --check   # check dist/ for that release; without --check it also publishes
```

To release, push a tag such as `v0.2.0` on a commit of `main` whose CI run passed, then bump the version in `Cargo.toml` to the next one. The release workflow builds nothing. It fetches the files that CI built and tested for that commit, checks them with `release.py` and publishes them to PyPI. If that build has expired, the workflow stops and asks you to re-run the commit's CI run.

The Rust core in `crates/lgbmjit-core` has no Python dependency. `crates/lgbmjit-python` holds the bindings and `python/lgbmjit` the Python API.

## Credits

`lgbmjit` follows the approach of [lleaves](https://github.com/siboehm/lleaves) by Simon Boehm. Its test models come from lleaves under the MIT license (`tests/models/LICENSE-lleaves`).

## License

MIT
