import numpy as np, pandas as pd, pytest
from numpy.testing import assert_array_equal

from lgbmjit._data import to_array


def test_array_conversion():
    "Arrays and lists convert as in LightGBM: float32 and float64 are kept without a copy, other dtypes become float32."
    for dtype in (np.int64, np.int8, np.bool_, np.float16, np.uint32): assert to_array(np.ones((3, 2), dtype), None).dtype == np.float32
    for dtype in (np.float32, np.float64):
        X = np.ones((3, 2), dtype)
        assert to_array(X, None) is X
    assert to_array([[1.0, 2.0]], None).dtype == np.float64
    assert to_array([[1, 2]], None).dtype == np.float32
    with pytest.raises(ValueError, match="2 dimensional"): to_array(np.ones(3), None)
    with pytest.raises(TypeError): to_array("data.csv", None)


def test_dataframe_conversion():
    "DataFrames take LightGBM's dtype rule, and categories map to codes in training order."
    df = pd.DataFrame(dict(a=np.ones(2, np.float32), b=np.ones(2, np.int8)))
    assert to_array(df, None).dtype == np.float32
    df["c"] = np.ones(2, np.int64)
    assert to_array(df, None).dtype == np.float64
    cats = pd.DataFrame(dict(a=pd.Categorical(["y", "x", "z", "w"], categories=["z", "y", "x", "w"])))
    assert_array_equal(to_array(cats, [["x", "y", "z"]])[:, 0], [1.0, 0.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="categorical"): to_array(cats, [])
    with pytest.raises(ValueError, match="non empty"): to_array(pd.DataFrame(dict(a=[])), None)
