import textwrap

import pytest
from qiskit import QuantumCircuit

from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.core.dag import DAGEdge, DagNode


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


def test_add_cut0(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    """
    In the annotated circuit diagram below, the first number in each i,j pair
    indicates the wire index, and the second number indicates the gate index
    on that wire. Single qubit gates (like H) do not increment the gate index.
    
                  ┌───┐ 0,0        0,1    
        q_0: ─|0>─┤ H ├──■──────────■──
                  └───┘┌─┴─┐        │  
        q_1: ─|0>──────┤ X ├────────┼──
                   1,0 └───┘      ┌─┴─┐
        q_2: ─|0>─────────────────┤ X ├
                                  └───┘
                                   2,0
                                   
    Each gate that spans exactly 2 wires (the 2 cx gates in this circuit)
    constitute a "DAGEdge". A DAGEdge is defined by its start and end "DagNode"s.
    A DAGNode is specified by its wire index and gate index on that wire.
    
    The `subcircuits` parameter to <cut_circuit>.cut
    denotes the list of DAGEdges that make up that subcircuit.
    
    Thus, by specifying the subcircuits as:
      subcircuit_0 = [DAGEdge(DagNode(0,0), DagNode(1,0))]
      subcircuit_1 = [DAGEdge(DagNode(0,1), DagNode(2,0))]
    
    we're effectively placing the cut at (note the location of the `//` gate):

                  ┌───┐ 0,0 ┌────┐ 0,1    
        q_0: ─|0>─┤ H ├──■──┤ // ├──■──
                  └───┘┌─┴─┐└────┘  │  
        q_1: ─|0>──────┤ X ├────────┼──
                   1,0 └───┘      ┌─┴─┐
        q_2: ─|0>─────────────────┤ X ├
                                  └───┘
                                   2,0   
    """
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0),
                DagNode(wire_index=1, gate_index=0),
            )
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=0),
                DagNode(wire_index=0, gate_index=1),
            )
        ],
    ]
    cut_circuit.cut(subcircuits=subcircuits)
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
    assert cut_circuit.num_cuts == 1


def test_add_cut1(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0),
                DagNode(wire_index=1, gate_index=0),
            )
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=0),
                DagNode(wire_index=0, gate_index=1),
            )
        ],
    ]
    cut_circuit.cut(subcircuits=subcircuits)

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
    subcircuits = cut_circuit.find_cuts(
        max_subcircuit_width=2,
        max_cuts=1,
        num_subcircuits=[2],
    )

    assert len(subcircuits) == 2
    assert len(subcircuits[0]) == 1
    assert len(subcircuits[1]) == 1

    assert str(subcircuits[0][0]) == "[0]0 [1]0"
    assert str(subcircuits[1][0]) == "[0]1 [2]0"


def test_cut_circuit_verify(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0),
                DagNode(wire_index=1, gate_index=0),
            )
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=0),
                DagNode(wire_index=0, gate_index=1),
            )
        ],
    ]

    cut_circuit.cut(subcircuits=subcircuits)

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.verify(probabilities)


def test_cut_circuit_figure4_cut(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_circuit.cut(
        max_subcircuit_width=3,
        max_cuts=1,
        num_subcircuits=[2],
    )

    assert cut_circuit.num_cuts == 1


def test_cut_circuit_figure4_reconstruction_order(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0),
                DagNode(wire_index=1, gate_index=0),
            ),
            DAGEdge(
                DagNode(wire_index=0, gate_index=1),
                DagNode(wire_index=2, gate_index=0),
            ),
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=1),
                DagNode(wire_index=4, gate_index=0),
            ),
            DAGEdge(
                DagNode(wire_index=3, gate_index=0),
                DagNode(wire_index=2, gate_index=2),
            ),
        ],
    ]
    cut_circuit.cut(subcircuits=subcircuits)
    assert cut_circuit.reconstruction_qubit_order == {0: [1, 0], 1: [3, 4, 2]}


def test_cut_circuit_figure4_verify(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    subcircuits = [
        [
            DAGEdge(
                DagNode(wire_index=0, gate_index=0),
                DagNode(wire_index=1, gate_index=0),
            ),
            DAGEdge(
                DagNode(wire_index=0, gate_index=1),
                DagNode(wire_index=2, gate_index=0),
            ),
        ],
        [
            DAGEdge(
                DagNode(wire_index=2, gate_index=1),
                DagNode(wire_index=4, gate_index=0),
            ),
            DAGEdge(
                DagNode(wire_index=3, gate_index=0),
                DagNode(wire_index=2, gate_index=2),
            ),
        ],
    ]
    cut_circuit.cut(subcircuits=subcircuits)
    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.verify(probabilities)
