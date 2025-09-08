import numpy as np

from cutqc2.core.utils import permute_bits, permute_bits_vectorized


def test_permute0():
    n = 0b1101
    # permutation from bit 0 to bit n-1 (right to left)
    permutation = [3, 1, 2, 0]
    expected = 0b1101
    assert permute_bits(n, permutation) == expected


def test_permute1():
    n = 0b1100
    # permutation from bit 0 to bit n-1 (right to left)
    permutation = [3, 1, 2, 0]
    expected = 0b0101
    assert permute_bits(n, permutation) == expected


def test_permute_vectorized():
    n_bits = 8
    n_samples = 1000
    n = np.random.randint(0, 2 ** (n_bits - 1), n_samples)
    permutation = np.arange(n_bits)
    np.random.shuffle(permutation)
    expected = np.array([permute_bits(x, permutation) for x in n])
    result = permute_bits_vectorized(n, permutation, n_bits=n_bits)
    assert np.array_equal(result, expected)
