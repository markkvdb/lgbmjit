from typing import Literal

import numpy as np
import numpy.typing as npt

__version__: str

class Forest:
    """A parsed LightGBM model."""

    def __init__(self, model_str: str) -> None: ...
    @property
    def num_feature(self) -> int: ...
    @property
    def num_class(self) -> int: ...
    @property
    def trees_per_iteration(self) -> int: ...
    @property
    def num_trees(self) -> int: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def pandas_categorical(self) -> str | None: ...
    @property
    def text(self) -> str: ...
    def compile(
        self,
        dtype: str,
        start_iteration: int,
        num_iteration: int,
        block_trees: int,
        tile_rows: int,
        branch_free_leaves: int,
        cold_fraction: float,
    ) -> CompiledForest: ...

class CompiledForest:
    """A model compiled for one input dtype and iteration range."""

    @property
    def dtype(self) -> Literal["float32", "float64"]: ...
    @property
    def tile_rows(self) -> int: ...
    def set_tile_rows(self, rows: int) -> None: ...
    def predict(
        self,
        data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
        raw_score: bool,
        num_threads: int,
    ) -> npt.NDArray[np.float64]: ...

def default_settings() -> dict[str, int | float]: ...
