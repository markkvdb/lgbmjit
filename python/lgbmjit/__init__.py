'Compile LightGBM models to machine code and predict with them.'

from lgbmjit import _lgbmjit
from lgbmjit._model import Model

__version__ = _lgbmjit.__version__

__all__ = ["Model", "__version__"]
