from cutqc2.cupy.simple import matrix_add, matrix_subtract, vector_kron
import numpy as np
import cupy as cp
import pytest


@pytest.mark.skip(reason="Cannot run on Github CI currently")
def test_matrix_add():
    a = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32)
    b = np.array([[0, 3, 6, 9], [12, 15, 18, 21]], dtype=np.float32)
    expected = a + b
    result = matrix_add(a, b)
    assert np.allclose(result, expected)


@pytest.mark.skip(reason="Cannot run on Github CI currently")
def test_matrix_subtract():
    a = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32)
    b = np.array([[0, 3, 6, 9], [12, 15, 18, 21]], dtype=np.float32)
    expected = a - b
    result = matrix_subtract(a, b)
    assert np.allclose(result, expected)


@pytest.mark.skip(reason="Cannot run on Github CI currently")
def test_vector_kron():
    a = cp.random.random(1000, dtype=cp.float32)
    b = cp.random.random(1000, dtype=cp.float32)
    expected = cp.kron(a, b)
    result = vector_kron(a, b)
    assert cp.allclose(result, expected)