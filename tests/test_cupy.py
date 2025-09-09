import numpy as np

from cutqc2.cupy import vector_kron
from cutqc2.numeric import xp


def test_vector_kron():
    a = xp.random.random(1000).astype(xp.float32)
    b = xp.random.random(1000).astype(xp.float32)
    expected = np.kron(a, b)
    result = vector_kron(a, b)
    assert np.allclose(result, expected)
