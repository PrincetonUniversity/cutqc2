import numpy as np

from cutqc2.core.utils import measure_prob, measure_sign

"""
sigma = measure_sign(<unmeasured_states>, <measurement_bases>)
  unmeasured_state is the n-bit state from cut circuit
  measurement_bases is the n-length list of measurement bases, MSB to LSB
    (bottom-wire to top-wire).
  sigma denotes the sign (+1, -1) in Eq. 3 of the paper.
"""


# If all measurement bases are "comp", then sigma=1.
def test_measure_state0():
    sigma = measure_sign([0b000], ("comp", "comp", "comp"))
    assert sigma == 1


def test_measure_state1():
    sigma = measure_sign([0b001], ("comp", "comp", "comp"))
    assert sigma == 1


def test_measure_state2():
    sigma = measure_sign([0b010], ("comp", "comp", "comp"))
    assert sigma == 1


def test_measure_state3():
    # Eq. 3 in paper:
    #   xx0, xx1 -> +xx  if M_last = I
    # So we have sigma = +1
    sigma = measure_sign([0b100], ("I", "comp", "comp"))
    assert sigma == 1


def test_measure_state4():
    # Eq. 3 in paper:
    #   xx1 -> -xx  if M_last != I
    # So we have sigma = -1
    sigma = measure_sign([0b1110], ("Z", "comp", "comp", "comp"))
    assert sigma == -1


def test_measure_prob0():
    # We go from a 2^n probability vector for a subcircuit
    # to a 2^(n-1) "quasi"-probability vector
    result = measure_prob(
        [
            0.25,  # 000 -> +0.25  for 00
            0,  # 001 -> +0.00  for 01
            0,  # 010 -> +0.00  for 10
            0.25,  # 011 -> +0.25  for 11
            0.125,  # 100 -> -0.125 for 00
            0.125,  # 101 -> -0.125 for 01
            0.125,  # 110 -> -0.125 for 10
            0.125,  # 111 -> -0.125 for 11
        ],
        ["Z", "comp", "comp"],
    )
    assert np.allclose(result, [0.125, -0.125, -0.125, 0.125])
