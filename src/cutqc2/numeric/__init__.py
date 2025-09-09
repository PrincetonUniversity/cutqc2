from cutqc2 import config


def numeric_object(which):
    if which == "cupy":
        from .cupy import Cupy as NumericClass
    elif which == "numpy":
        from .numpy import Numpy as NumericClass
    else:
        raise RuntimeError(f"Invalid selection for numeric module: {which}")
    return NumericClass()


try:
    import cupy  # noqa: F401
except ImportError:
    xp = numeric_object("numpy")
else:
    xp = numeric_object(config.core.numeric)
