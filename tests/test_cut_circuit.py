import textwrap
import pytest
from qiskit import QuantumCircuit
from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.core.dag import DagNode, DAGEdge


@pytest.fixture(scope="module")
def simple_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.reset(0)
    qc.reset(1)
    qc.reset(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    return qc


def test_cut_circuit_add_cut(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_circuit.add_cut("0004", 0, 0)
    got_str = str(cut_circuit)
    expected_str = textwrap.dedent("""
                  ┌───┐     ┌────┐     
        q_0: ─|0>─┤ H ├──■──┤ // ├──■──
                  └───┘┌─┴─┐└────┘  │  
        q_1: ─|0>──────┤ X ├────────┼──
                       └───┘      ┌─┴─┐
        q_2: ─|0>─────────────────┤ X ├
                                  └───┘
    """).strip("\n")
    assert got_str == expected_str
    assert cut_circuit.cuts == [(0, 2)]


def test_cut_circuit_add_cut_at_position(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_circuit.add_cut_at_position(wire_index=0, gate_index=2)
    got_str = str(cut_circuit)
    expected_str = textwrap.dedent("""
                  ┌───┐     ┌────┐     
        q_0: ─|0>─┤ H ├──■──┤ // ├──■──
                  └───┘┌─┴─┐└────┘  │  
        q_1: ─|0>──────┤ X ├────────┼──
                       └───┘      ┌─┴─┐
        q_2: ─|0>─────────────────┤ X ├
                                  └───┘
    """).strip("\n")
    assert got_str == expected_str
    assert cut_circuit.cuts == [(0, 2)]


def test_cut_circuit_generate_subcircuits(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_edge_pairs = [
        (
            DAGEdge(
                DagNode(wire_index=0, gate_index=0, name="0004"),
                DagNode(wire_index=1, gate_index=0, name="0004"),
            ),
            DAGEdge(
                DagNode(wire_index=0, gate_index=1, name="0005"),
                DagNode(wire_index=2, gate_index=0, name="0005"),
            ),
        )
    ]
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0, name="0004"),
                DagNode(wire_index=1, gate_index=0, name="0004"),
            )
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=0, name="0005"),
                DagNode(wire_index=0, gate_index=1, name="0005"),
            )
        ],
    ]

    cut_circuit.add_cuts_and_generate_subcircuits(cut_edge_pairs, subcircuits)

    assert len(cut_circuit) == 2

    first_subcircuit_str = str(cut_circuit[0])
    expected_str = textwrap.dedent("""
                  ┌───┐     
        q_0: ─|0>─┤ H ├──■──
                  └───┘┌─┴─┐
        q_1: ─|0>──────┤ X ├
                       └───┘
    """).strip("\n")
    assert first_subcircuit_str == expected_str

    second_subcircuit_str = str(cut_circuit[1]).strip()
    expected_str = textwrap.dedent("""
        q_0: ───────■──
                  ┌─┴─┐
        q_1: ─|0>─┤ X ├
                  └───┘
    """).strip()
    assert second_subcircuit_str == expected_str


def test_cut_circuit_find_cuts(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_edges_pairs, _ = cut_circuit.find_cuts(
        max_subcircuit_width=2,
        max_cuts=1,
        num_subcircuits=[2],
        max_subcircuit_cuts=1,
        subcircuit_size_imbalance=1,
    )

    assert len(cut_edges_pairs) == 1
    cut_edge0, cut_edge1 = cut_edges_pairs[0]
    assert str(cut_edge0) == "0004[0]0 0004[1]0"
    assert str(cut_edge1) == "0005[0]1 0005[2]0"


def test_cut_circuit_verify(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_edge_pairs = [
        (
            DAGEdge(
                DagNode(wire_index=0, gate_index=0, name="0004"),
                DagNode(wire_index=1, gate_index=0, name="0004"),
            ),
            DAGEdge(
                DagNode(wire_index=0, gate_index=1, name="0005"),
                DagNode(wire_index=2, gate_index=0, name="0005"),
            ),
        )
    ]
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0, name="0004"),
                DagNode(wire_index=1, gate_index=0, name="0004"),
            )
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=0, name="0005"),
                DagNode(wire_index=0, gate_index=1, name="0005"),
            )
        ],
    ]

    cut_circuit.add_cuts_and_generate_subcircuits(cut_edge_pairs, subcircuits)

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.verify(probabilities)


def test_cut_circuit_figure4_cut(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_circuit.cut(
        max_subcircuit_width=3,
        max_subcircuit_cuts=2,
        subcircuit_size_imbalance=3,
        max_cuts=1,
        num_subcircuits=[2],
    )

    assert cut_circuit.cuts == [(2, 0)]


def test_cut_circuit_figure4_reconstruction_order(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_edge_pairs = [
        (
            DAGEdge(
                DagNode(wire_index=0, gate_index=1, name="0014"),
                DagNode(wire_index=2, gate_index=0, name="0014"),
            ),
            DAGEdge(
                DagNode(wire_index=2, gate_index=1, name="0018"),
                DagNode(wire_index=4, gate_index=0, name="0018"),
            ),
        )
    ]
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0, name="0010"),
                DagNode(wire_index=1, gate_index=0, name="0010"),
            ),
            DAGEdge(
                DagNode(wire_index=0, gate_index=1, name="0014"),
                DagNode(wire_index=2, gate_index=0, name="0014"),
            ),
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=1, name="0018"),
                DagNode(wire_index=4, gate_index=0, name="0018"),
            ),
            DAGEdge(
                DagNode(wire_index=3, gate_index=0, name="0021"),
                DagNode(wire_index=2, gate_index=2, name="0021"),
            ),
        ],
    ]
    cut_circuit.add_cuts_and_generate_subcircuits(cut_edge_pairs, subcircuits)
    assert cut_circuit.reconstruction_qubit_order == {0: [1, 0], 1: [3, 4, 2]}


def test_cut_circuit_figure4_verify(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_edge_pairs = [
        (
            DAGEdge(
                DagNode(wire_index=0, gate_index=1, name="0014"),
                DagNode(wire_index=2, gate_index=0, name="0014"),
            ),
            DAGEdge(
                DagNode(wire_index=2, gate_index=1, name="0018"),
                DagNode(wire_index=4, gate_index=0, name="0018"),
            ),
        )
    ]
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0, name="0010"),
                DagNode(wire_index=1, gate_index=0, name="0010"),
            ),
            DAGEdge(
                DagNode(wire_index=0, gate_index=1, name="0014"),
                DagNode(wire_index=2, gate_index=0, name="0014"),
            ),
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=1, name="0018"),
                DagNode(wire_index=4, gate_index=0, name="0018"),
            ),
            DAGEdge(
                DagNode(wire_index=3, gate_index=0, name="0021"),
                DagNode(wire_index=2, gate_index=2, name="0021"),
            ),
        ],
    ]
    cut_circuit.add_cuts_and_generate_subcircuits(cut_edge_pairs, subcircuits)
    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.verify(probabilities)
