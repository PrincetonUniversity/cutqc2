import math

from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.cutqc.helper_functions.benchmarks import generate_circ


def test_supremacy_reconstruction_with_increasing_capacity():
    circuit = generate_circ(
        num_qubits=6,
        depth=1,
        circuit_type="supremacy",
        reg_name="q",
        connected_only=True,
        seed=1234,
    )

    cut_circuit = CutCircuit(circuit)
    cut_circuit.cut(
        max_subcircuit_width=math.ceil(circuit.num_qubits / 4 * 3),
        max_cuts=10,
        num_subcircuits=[3],
    )
    cut_circuit.run_subcircuits()

    errors = []
    for capacity in (
        1,
        2,
        3,
        4,
        5,
        6,
    ):
        cut_circuit.postprocess(capacity=capacity)
        probabilities = cut_circuit.get_probabilities()
        error = cut_circuit.verify(probabilities, raise_error=False)
        # error should decrease with increasing capacity
        if len(errors) > 0:
            assert error <= errors[-1]
        errors.append(error)

    # The final error with full capacity should be very small
    assert error < 1e-10
