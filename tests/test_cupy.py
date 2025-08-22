from cutqc2.cupy import vector_kron
import cupy as cp


def test_vector_kron():
    a = cp.random.random(1000, dtype=cp.float32)
    b = cp.random.random(1000, dtype=cp.float32)
    expected = cp.kron(a, b)
    result = vector_kron(a, b)
    assert cp.allclose(result, expected)
