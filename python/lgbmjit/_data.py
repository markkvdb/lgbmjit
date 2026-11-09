'Conversion of prediction input to the arrays LightGBM would predict on.'

from __future__ import annotations

import os, sys
from typing import Any

import numpy as np


def _pandas(): return sys.modules.get("pandas")


def is_dataframe(data: Any) -> bool:
    pd = _pandas()
    return pd is not None and isinstance(data, pd.DataFrame)


def _allowed(t: type) -> bool:
    return issubclass(t, (np.integer, np.floating, np.bool_)) and (not issubclass(t, (np.timedelta64, np.datetime64)))


def _from_pandas(data: Any, pandas_categorical: list | None) -> np.ndarray:
    pd = _pandas()
    assert pd is not None, "is_dataframe(data) holds, so pandas is imported"
    if len(data.shape) != 2 or data.shape[0] < 1: raise ValueError("Input data must be 2 dimensional and non empty.")
    data = data.copy(deep=False)
    cat_cols = [c for c, t in zip(data.columns, data.dtypes, strict=True) if isinstance(t, pd.CategoricalDtype)]
    if pandas_categorical is not None:
        if len(cat_cols) != len(pandas_categorical):
            raise ValueError(
                f"The data has {len(cat_cols)} categorical columns, "
                f"but the model was trained with {len(pandas_categorical)}.")
        for col, categories in zip(cat_cols, pandas_categorical, strict=True):
            if list(data[col].cat.categories) != list(categories): data[col] = data[col].cat.set_categories(categories)
    if cat_cols: data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
    bad = [f"{c}: {t}" for c, t in data.dtypes.items() if not _allowed(t.type)]
    if bad: raise ValueError("pandas dtypes must be int, float or bool. Bad columns: " + ", ".join(bad))
    target = np.result_type(*[t.type for t in data.dtypes], np.float32)
    try: return data.to_numpy(dtype=target, copy=False)
    except ValueError: return data.to_numpy(dtype=target, na_value=np.nan)


def to_array(data: Any, pandas_categorical: list | None) -> np.ndarray:
    '`data` as a 2-D float32 or float64 array, converted as `lightgbm.Booster.predict` does.'
    if isinstance(data, (str, os.PathLike)): raise TypeError("lgbmjit does not read data files; pass an array, a list or a DataFrame")
    if is_dataframe(data): return _from_pandas(data, pandas_categorical)
    if isinstance(data, list): data = np.array(data)
    if not isinstance(data, np.ndarray): raise TypeError(f"unsupported data type {type(data).__name__}")
    if data.ndim != 2: raise ValueError("Input numpy.ndarray or list must be 2 dimensional")
    if data.dtype not in (np.float32, np.float64): data = data.astype(np.float32)
    if not data.flags.aligned or any(s % data.itemsize for s in data.strides):
        # the compiled code reads whole, aligned elements; np.array always copies to new memory
        data = np.array(data, order="C")
    return data
