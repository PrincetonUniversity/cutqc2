import cupy as cp
import numpy as np

from cutqc2.core.dynamic_definition import DynamicDefinition
from cutqc2.core.utils import merge_prob_vector

full_distribution = np.append(np.zeros(15), 1)


def probability_distribution(qubit_spec: str) -> np.ndarray:
    return merge_prob_vector(full_distribution, qubit_spec)


def test_dynamic_definition():
    dynamic_definition = DynamicDefinition(
        num_qubits=4, capacity=1, prob_fn=probability_distribution, epsilon=1e-4
    )

    dynamic_definition.run(max_recursion=10)

    assert len(dynamic_definition.bins) == 1
    assert dynamic_definition.bins[0].qubit_spec == "1111"
    np.testing.assert_array_almost_equal(
        cp.asnumpy(dynamic_definition.bins[0].probabilities), np.array([1.0])
    )
