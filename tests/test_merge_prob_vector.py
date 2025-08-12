from itertools import product
import numpy as np
from cutqc2.core.utils import merge_prob_vector, unmerge_prob_vector


def test_all_active_qubits_merge():
    # 2-qubit system, all active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = merge_prob_vector(prob_vector, qubit_spec="AA")
    np.testing.assert_array_almost_equal(result, prob_vector)


def test_all_active_qubits_unmerge():
    # 2-qubit system, all active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = np.zeros_like(prob_vector)
    unmerge_prob_vector(prob_vector, qubit_spec="AA", preallocated=result)
    np.testing.assert_array_almost_equal(result, prob_vector)


def test_all_merged_qubits_merge():
    # 2-qubit system, all merged
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = merge_prob_vector(prob_vector, qubit_spec="MM")
    np.testing.assert_array_almost_equal(result, [1.0])


def test_all_merged_qubits_unmerge():
    # 2-qubit system, all merged
    prob_vector = np.array([1.0])
    result = np.zeros(4)
    unmerge_prob_vector(prob_vector, qubit_spec="MM", preallocated=result)
    # probability mass is evenly distributed
    np.testing.assert_array_almost_equal(result, [0.25, 0.25, 0.25, 0.25])


def test_merged_active_qubits_merge():
    # 2-qubit system, 1 merged, 1 active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = merge_prob_vector(prob_vector, qubit_spec="MA")
    # Note that we read the qubit spec from MSB to LSB
    # So qubit 0 is active, qubit 1 is merged
    # (00, 10) merge into 0 as (0.1 + 0.3)
    # (01, 11) merge into 1 as (0.2 + 0.4)
    np.testing.assert_array_almost_equal(result, [0.4, 0.6])


def test_merged_active_qubits_unmerge():
    # 2-qubit system, 1 merged, 1 active
    prob_vector = np.array([0.4, 0.6])
    result = np.zeros(4)
    unmerge_prob_vector(prob_vector, "MA", result)
    # Note that we read the qubit specification from MSB to LSB
    # So qubit 0 is active, qubit 1 is merged
    # 00 (0.4) unmerges into (00, 10) as (0.2, 0.2)
    # 01 (0.6) unmerges into (01, 11) as (0.3, 0.3)
    np.testing.assert_array_almost_equal(result, [0.2, 0.3, 0.2, 0.3])


def test_active_merged_qubits_merge():
    # 2-qubit system, 1 merged, 1 active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = merge_prob_vector(prob_vector, "AM")
    # Note that we read the qubit specification from MSB to LSB
    # So qubit 0 is merged, qubit 1 is active
    # (00, 01) merge into 0 as (0.1 + 0.2)
    # (10, 11) merge into 1 as (0.3 + 0.4)
    np.testing.assert_array_almost_equal(result, [0.3, 0.7])


def test_active_merged_qubits_unmerge():
    # 2-qubit system, 1 merged, 1 active
    prob_vector = np.array([0.3, 0.7])
    result = np.zeros(4)
    unmerge_prob_vector(prob_vector, "AM", result)
    # Note that we read the qubit specification from MSB to LSB
    # So qubit 0 is merged, qubit 1 is active
    # 00 (0.3) unmerges into (00, 01) as (0.15, 0.15)
    # 01 (0.7) unmerges into (10, 11) as (0.35, 0.35)
    np.testing.assert_array_almost_equal(result, [0.15, 0.15, 0.35, 0.35])


def test_mixed_active_merged0_merge():
    # 3-qubit system: 2 merged, 1 active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qubit_spec = "AMM"  # qubit 2 active, qubit 1 and 0 merged

    result = merge_prob_vector(prob_vector, qubit_spec)
    # (000, 001, 010, 011) merge into 0 as (0.1 + 0.2 + 0.3 + 0.4)
    # (100, 101, 110, 111) merge into 1 as (0.5 + 0.6 + 0.7 + 0.8)
    np.testing.assert_array_almost_equal(result, [1.0, 2.6])


def test_mixed_active_merged0_unmerge():
    # 3-qubit system: 2 merged, 1 active
    prob_vector = np.array([1.0, 2.6])
    qubit_spec = "AMM"  # qubit 2 active, qubit 1 and 0 merged
    result = np.zeros(8)

    unmerge_prob_vector(prob_vector, qubit_spec, result)
    # 000 (1.0) unmerges into (000, 001, 010, 011) as (0.25, 0.25, 0.25, 0.25)
    # 100 (2.6) unmerges into (100, 101, 110, 111) as (0.65, 0.65, 0.65, 0.65)
    np.testing.assert_array_almost_equal(
        result, [0.25, 0.25, 0.25, 0.25, 0.65, 0.65, 0.65, 0.65]
    )


def test_mixed_active_merged1_merge():
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qubit_spec = "AAM"  # qubit 2 and 1 active, qubit 0 merged

    result = merge_prob_vector(prob_vector, qubit_spec)
    # (000, 001) merge into 00 as (0.1 + 0.2)
    # (010, 011) merge into 01 as (0.3 + 0.4)
    # (100, 101) merge into 10 as (0.5 + 0.6)
    # (110, 111) merge into 11 as (0.7 + 0.8)
    np.testing.assert_array_almost_equal(result, [0.3, 0.7, 1.1, 1.5])


def test_mixed_active_merged1_unmerge():
    prob_vector = np.array([0.3, 0.7, 1.1, 1.5])
    qubit_spec = "AAM"  # qubit 2 and 1 active, qubit 0 merged
    result = np.zeros(8)

    unmerge_prob_vector(prob_vector, qubit_spec, result)
    # 00 (0.3) unmerges into (000, 001) as (0.15, 0.15)
    # 01 (0.7) unmerges into (010, 011) as (0.35, 0.35)
    # 10 (1.1) unmerges into (100, 101) as (0.55, 0.55)
    # 11 (1.5) unmerges into (110, 111) as (0.75, 0.75)
    np.testing.assert_array_almost_equal(
        result, [0.15, 0.15, 0.35, 0.35, 0.55, 0.55, 0.75, 0.75]
    )


def test_merge_probability_conservation():
    for num_qubits in range(2, 8):
        prob_vector = np.random.random(2**num_qubits)

        for qubit_spec in product("AM", repeat=num_qubits):
            result = merge_prob_vector(prob_vector, qubit_spec)
            assert np.isclose(np.sum(result), np.sum(prob_vector))


def test_unmerge_probability_conservation():
    for num_qubits in range(2, 8):
        for qubit_spec in product("AM", repeat=num_qubits):
            num_active = qubit_spec.count("A")
            prob_vector = np.random.random(2**num_active)
            result = np.zeros(2**num_qubits)

            unmerge_prob_vector(prob_vector, qubit_spec, result)
            assert np.isclose(np.sum(result), np.sum(prob_vector))


def test_conditioning0_merge():
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qubit_spec = "M0M"  # capture indices (0, 1, 4, 5)

    result = merge_prob_vector(prob_vector, qubit_spec)
    # 0.1 + 0.2 + 0.5 + 0.6 = 1.4
    np.testing.assert_array_almost_equal(result, [1.4])


def test_conditioning0_unmerge():
    prob_vector = np.array([1.4])
    qubit_spec = "M0M"  # capture indices (0, 1, 4, 5)
    result = np.zeros(8)

    unmerge_prob_vector(prob_vector, qubit_spec, result)
    # 1.4 / 4 = 0.35 for each of the four indices
    np.testing.assert_array_almost_equal(result, [0.35, 0.35, 0, 0, 0.35, 0.35, 0, 0])


def test_conditioning1_merge():
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qubit_spec = "M1M"  # capture indices (2, 3, 6, 7)

    result = merge_prob_vector(prob_vector, qubit_spec)
    # 0.3 + 0.4 + 0.7 + 0.8 = 2.2
    np.testing.assert_array_almost_equal(result, [2.2])


def test_conditioning1_unmerge():
    prob_vector = np.array([2.2])
    qubit_spec = "M1M"  # capture indices (2, 3, 6, 7)
    result = np.zeros(8)

    unmerge_prob_vector(prob_vector, qubit_spec, result)
    # 2.2 / 4 = 0.55 for each of the four indices
    np.testing.assert_array_almost_equal(result, [0, 0, 0.55, 0.55, 0, 0, 0.55, 0.55])


def test_conditioning2_merge():
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    result = merge_prob_vector(prob_vector, "AM1")

    # (001, 011) merge into 0 as (0.2 + 0.4)
    # (101, 111) merge into 1 as (0.6 + 0.8)
    np.testing.assert_array_almost_equal(result, [0.6, 1.4])


def test_conditioning2_unmerge():
    prob_vector = np.array([0.6, 1.4])
    result = np.zeros(8)
    unmerge_prob_vector(prob_vector, "AM1", result)

    # 0.6 unmerges into (001, 011) as (0.3, 0.3)
    # 1.4 unmerges into (101, 111) as (0.7, 0.7)
    np.testing.assert_array_almost_equal(result, [0, 0.3, 0, 0.3, 0, 0.7, 0, 0.7])
