import numpy as np

from cutqc2.core.utils import permute_bits


def permute_bits_naive(n: int, permutation: list[int]) -> int:
    # A function for permuting bits of an integer n according to a given permutation
    # but implemented without bitshifts for easy understanding.
    n_bits = len(permutation)
    binary_n = f"{n:0{n_bits}b}"
    # Get bit i from position permutation[i]
    binary_n_permuted = "".join(binary_n[permutation[i]] for i in range(n_bits))
    return int(binary_n_permuted, 2)


def test_permute0():
    n = 0b1101
    # permutation from bit 0 to bit n-1 (right to left)
    permutation = [3, 1, 2, 0]
    expected = 0b1101
    assert permute_bits_naive(n, permutation) == expected


def test_permute1():
    n = 0b1100
    # permutation from bit 0 to bit n-1 (right to left)
    permutation = [3, 1, 2, 0]
    expected = 0b0101
    assert permute_bits_naive(n, permutation) == expected


def test_permute_vectorized():
    # Test `cutqc2` version of permute_bits against naive implementation
    n_bits = 8
    n_samples = 1000
    n = np.random.randint(0, 2 ** (n_bits - 1), n_samples)
    permutation = np.arange(n_bits)
    np.random.shuffle(permutation)
    expected = np.array([permute_bits_naive(x, permutation) for x in n])
    result = permute_bits(n, permutation, n_bits=n_bits)
    assert np.array_equal(result, expected)
