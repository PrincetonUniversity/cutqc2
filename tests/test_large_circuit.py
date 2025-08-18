"""
Tests for end-end verification of large circuits.
Both correctness and performance can/should be tested here.
"""

from cutqc2.cutqc.helper_functions.benchmarks import construct_random
from cutqc2.core.cut_circuit import CutCircuit


def test_verify():
    qc = construct_random(num_qubits=16, depth=16, seed=5435)
    cut_circuit = CutCircuit(qc)
    cut_circuit.cut(
        max_subcircuit_width=6,
        max_subcircuit_cuts=10,
        subcircuit_size_imbalance=3,
        max_cuts=10,
        num_subcircuits=[5],
    )
    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.verify(probabilities=probabilities)
