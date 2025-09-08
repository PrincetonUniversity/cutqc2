"""
This script reproduces Figure 7 of the CutQC paper, showing the effect of
varying recursion levels on reconstruction of a 4-qubit BV Circuit.
"""

import numpy as np

from cutqc2.core.dynamic_definition import DynamicDefinition
from cutqc2.core.utils import merge_prob_vector

# Emulate the probability distribution for a 4-qubit Bernstein-Vazirani problem
four_qubit_BV_probability_distribution = np.append(np.zeros(15), 1)


if __name__ == "__main__":
    dynamic_definition = DynamicDefinition(
        # Bernstein-Vazirani problem with 4 qubits
        num_qubits=4,
        # We choose to have only 1 qubit active at a time
        capacity=1,
        # The probability function is a callable that takes in a "qubit spec"
        # (a string of 0/1/A/M characters), and returns the compressed quantum
        # probability vector by merging/conditioning on specific qubits.
        # The `merge_prob_vector` utility function from CutQC2 is used here.
        prob_fn=lambda qubit_spec, **kwargs: merge_prob_vector(
            four_qubit_BV_probability_distribution, qubit_spec
        ),
    )

    for max_recursion in (1, 2, 3, 4):
        dynamic_definition.run(max_recursion=max_recursion)
        dynamic_definition.plot()

    # For large state vectors, it is not recommended to plot *all* the
    # results (e.g., for 38 qubits), but rather to plot only the
    # largest bins that account for `prob_mass_threshold`.
    dynamic_definition.plot(auto=True, prob_mass_threshold=0.9)
