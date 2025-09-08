from cutqc2 import config


def numeric_object(which):
    if which == "cupy":
        from .cupy import Cupy as NumericClass
    elif which == "numpy":
        from .numpy import Numpy as NumericClass
    else:
        raise RuntimeError(f"Invalid selection for numeric module: {which}")
    return NumericClass()


xp = numeric_object(config.core.numeric)
