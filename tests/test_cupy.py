import cupy as cp

from cutqc2.cupy import vector_kron


def test_vector_kron():
    a = cp.random.random(1000, dtype=cp.float32)
    b = cp.random.random(1000, dtype=cp.float32)
    expected = cp.kron(a, b)
    result = vector_kron(a, b)
    assert cp.allclose(result, expected)
