"""
Tests for end-end verification of cut circuits.
"""

import math

from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.cutqc.helper_functions.benchmarks import generate_circ


def test_adder_verify():
    circuit = generate_circ(
        num_qubits=4,
        depth=4,
        circuit_type="adder",
        reg_name="q",
        connected_only=True,
        seed=None,
    )

    cut_circuit = CutCircuit(circuit)
    cut_circuit.cut(
        max_subcircuit_width=20,
        max_cuts=20,
        num_subcircuits=[3],
    )

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.verify(probabilities)


def test_figure4_verify(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_circuit.cut(
        max_subcircuit_width=3,
        max_cuts=1,
        num_subcircuits=[2],
    )

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.plot()
    cut_circuit.verify(probabilities)


def test_supremacy_verify(tmp_path):
    circuit = generate_circ(
        num_qubits=6,
        depth=1,
        circuit_type="supremacy",
        reg_name="q",
        connected_only=True,
        seed=None,
    )

    cut_circuit = CutCircuit(circuit)
    cut_circuit.cut(
        max_subcircuit_width=math.ceil(circuit.num_qubits / 4 * 3),
        max_cuts=10,
        num_subcircuits=[3],
    )

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.plot(output_file=tmp_path / "supremacy_verify.png")
    cut_circuit.verify(probabilities)
