import contextlib

import numpy as np

cp = None
with contextlib.suppress(ModuleNotFoundError):
    import cupy as cp


class Numpy:
    name: str = "numpy"

    @staticmethod
    def asnumpy(x):
        """
        Ensure `asnumpy` is always available and returns a numpy array.
        """
        if cp and isinstance(x, cp.ndarray):
            x = x.get()
        return x

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through \
        of any attribute that is not supported above.
        """
        return getattr(np, item)
