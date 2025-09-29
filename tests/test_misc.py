import numpy as np

from cutqc2.core.utils import measure_prob, measure_state

"""
sigma, effective_state = measure_state(<unmeasured_state>, <measurement_bases>)
  unmeasured_state is the n-bit state from cut circuit
  measurement_bases is the n-length list of measurement bases, LSB to MSB
    (top-wire to bottom-wire).
  sigma denotes the sign (+1, -1) in Eq. 3 of the paper
  effective_state is the state we wish to attribute the measurement to.
  This is the first n-1 bits of the n-bit unmeasured state.
"""


# If all measurement bases are "comp", then sigma=1 and effective_state=unmeasured state.
def test_measure_state0():
    sigma, effective_state = measure_state(0b000, ["comp", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 0


def test_measure_state1():
    sigma, effective_state = measure_state(0b001, ["comp", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 1


def test_measure_state2():
    sigma, effective_state = measure_state(0b010, ["comp", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 2


def test_measure_state3():
    # Eq. 3 in paper:
    #   xx0, xx1 -> +xx  if M_last = I
    # So we have result = (+1, 0b00) = (1, 0)
    sigma, effective_state = measure_state(0b100, ["comp", "comp", "I"])
    assert sigma == 1
    assert effective_state == 0


def test_measure_state4():
    # Eq. 3 in paper:
    #   xx1 -> -xx  if M_last != I
    # So we have result = (-1, 0b110) = (-1, 6)
    sigma, effective_state = measure_state(0b1110, ["comp", "comp", "comp", "Z"])
    assert sigma == -1
    assert effective_state == 6


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
        ["comp", "comp", "Z"],
    )
    assert np.allclose(result, [0.125, -0.125, -0.125, 0.125])
