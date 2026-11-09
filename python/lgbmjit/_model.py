'The `Model` class.'

from __future__ import annotations

import json, os, threading
from pathlib import Path
from typing import Any

from lgbmjit import _lgbmjit, _tune
from lgbmjit._data import is_dataframe, to_array

_THREAD_ALIASES = ("num_threads", "num_thread", "nthread", "nthreads", "n_jobs")


def _threads(kwargs: dict[str, Any]) -> int:
    unknown = sorted(set(kwargs) - set(_THREAD_ALIASES))
    if unknown: raise TypeError(f"unsupported keyword argument(s): {', '.join(unknown)}")
    values = [kwargs[k] for k in _THREAD_ALIASES if k in kwargs]
    n = values[0] if values else None
    return n if n is not None and n > 0 else (os.cpu_count() or 1)


class Model:
    """A LightGBM model compiled to machine code. Predictions equal `lightgbm.Booster.predict`.

    Pass the path of a model file saved with `Booster.save_model`, or its text as `model_str`.
    """

    def __init__(self, model_file: str|os.PathLike|None=None, *, model_str: str|None=None):
        if (model_file is None) == (model_str is None): raise TypeError("pass exactly one of model_file and model_str")
        if model_str is None:
            assert model_file is not None
            model_str = Path(model_file).read_text(encoding="utf-8")
        self._forest = _lgbmjit.Forest(model_str)
        pc = self._forest.pandas_categorical
        self._pandas_categorical = None if pc is None else json.loads(pc)
        self._settings = dict(_lgbmjit.default_settings())
        self._compiled = {}
        self._lock = threading.Lock()
        self._best_iteration = -1

    @classmethod
    def from_booster(cls, booster: Any) -> Model:
        """A `Model` with all trees of a `lightgbm.Booster`.

        Like `Booster.predict`, `predict` then defaults to the booster's best iteration, if any.
        """
        model = cls(model_str=booster.model_to_string(num_iteration=-1))
        model._best_iteration = booster.best_iteration
        return model

    def num_feature(self) -> int: return self._forest.num_feature

    def num_trees(self) -> int: return self._forest.num_trees

    def num_model_per_iteration(self) -> int: return self._forest.trees_per_iteration

    def feature_name(self) -> list[str]: return self._forest.feature_names

    def model_to_string(self) -> str: return self._forest.text

    @property
    def settings(self) -> dict[str, int | float]:
        'The code generation settings used for compiling.'
        return dict(self._settings)

    def _range(self, start_iteration: int, num_iteration: int) -> tuple[int, int]:
        total = self._forest.num_trees // self._forest.trees_per_iteration
        start = min(max(start_iteration, 0), total)
        left = total - start
        return start, (min(num_iteration, left) if num_iteration > 0 else left)

    def _get(self, dtype: str, start_iteration: int, num_iteration: int) -> _lgbmjit.CompiledForest:
        key = (dtype, *self._range(start_iteration, num_iteration))
        with self._lock:
            compiled = self._compiled.get(key)
            if compiled is None:
                s = self._settings
                compiled = self._forest.compile(dtype, key[1], key[2], int(s['block_trees']), int(s['tile_rows']),
                    int(s['branch_free_leaves']), float(s['cold_fraction']))
                self._compiled[key] = compiled
        return compiled

    def compile(self, *, tune: Any = None, tune_cache: str | os.PathLike | None = None) -> Model:
        """Compiles the model for float32 and float64 input now, not at the first `predict`.

        With `tune`, a sample of prediction input, it first times a grid of code generation
        settings on up to 20,000 rows of the sample and keeps the fastest for this machine.
        With `tune_cache`, a JSON file path, it reuses settings stored there for this model and
        CPU, and stores new ones there.
        """
        key = _tune.cache_key(self._forest.text)
        settings = _tune.read_cache(tune_cache, key) if tune_cache is not None else None
        if settings is None and tune is not None:
            settings = _tune.search(self._forest, to_array(tune, self._pandas_categorical))
            if tune_cache is not None: _tune.write_cache(tune_cache, key, settings)
        if settings is not None:
            with self._lock:
                self._settings = dict(settings)
                self._compiled.clear()
        for dtype in ("float32", "float64"): self._get(dtype, 0, 0)
        return self

    def predict(
        self,
        data, # 2-D NumPy array, list of rows or pandas DataFrame
        start_iteration=0, # First boosting iteration to use
        num_iteration=None, # Iterations to use; `None` means the best iteration of a `from_booster` model, else all
        raw_score=False, # Return raw scores, without the objective's output transform
        pred_leaf=False, # Not supported
        pred_contrib=False, # Not supported
        validate_features=False, # Check a DataFrame's column names against the model's feature names
        **kwargs # `num_threads`, or one of LightGBM's aliases for it; the default is every CPU
    ):
        "Predictions for `data`, with the arguments and results of `lightgbm.Booster.predict`."
        if pred_leaf or pred_contrib: raise NotImplementedError("lgbmjit does not support pred_leaf or pred_contrib")
        threads = _threads(kwargs)
        X = to_array(data, self._pandas_categorical)
        if validate_features and is_dataframe(data):
            names = [str(c) for c in data.columns]
            if names != self.feature_name():
                raise ValueError(
                    f"The feature names {names} differ from the model's {self.feature_name()}")
        if num_iteration is None:
            # Booster.predict's default: the best iteration, unless a start iteration is given
            num_iteration = self._best_iteration if start_iteration <= 0 else -1
        compiled = self._get(X.dtype.name, start_iteration, num_iteration)
        return compiled.predict(X, raw_score, threads)
