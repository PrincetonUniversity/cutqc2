import pytest

from cutqc2.core.dag import DAGEdge, DagNode


def test_dag_node_from_str():
    node = DagNode.from_string("004[5]3")
    assert node.name == "004"
    assert node.wire_index == 5
    assert node.gate_index == 3


def test_dag_node_order1():
    # DagNodes can be ordered by their wire index
    node1 = DagNode(wire_index=0, gate_index=0)
    node2 = DagNode(wire_index=1, gate_index=0)
    assert node1 < node2


def test_dag_node_order2():
    # DagNodes can be ordered by their wire index, or gate_index if wire index is the same
    node1 = DagNode(wire_index=1, gate_index=3)
    node2 = DagNode(wire_index=1, gate_index=0)
    assert node2 < node1


def test_dag_edge_from_str():
    edge = DAGEdge.from_string("0006[5]3 0052[2]4")
    assert edge.source == DagNode(wire_index=2, gate_index=4, name="0052")
    assert edge.dest == DagNode(wire_index=5, gate_index=3, name="0006")


def test_dage_edge():
    node1 = DagNode(wire_index=0, gate_index=0)
    node2 = DagNode(wire_index=1, gate_index=0)
    # regardless of the order of assignment, a DagEdge maintains its node
    # in a consistent order based on wire index
    edge = DAGEdge(node2, node1)
    assert edge.source == node1
    assert edge.dest == node2


def test_dage_edges_common_wire0():
    edge1 = DAGEdge(
        DagNode(wire_index=0, gate_index=0), DagNode(wire_index=1, gate_index=0)
    )
    edge2 = DAGEdge(
        DagNode(wire_index=0, gate_index=1), DagNode(wire_index=2, gate_index=0)
    )

    p, q = edge1 | edge2
    assert p.wire_index == 0


def test_dage_edges_common_wire1():
    edge1 = DAGEdge(
        DagNode(wire_index=0, gate_index=0), DagNode(wire_index=1, gate_index=0)
    )
    edge2 = DAGEdge(
        DagNode(wire_index=3, gate_index=4), DagNode(wire_index=1, gate_index=2)
    )

    p, q = edge1 | edge2
    assert p.wire_index == 1


def test_dage_edges_common_wire_error():
    edge1 = DAGEdge(
        DagNode(wire_index=0, gate_index=0), DagNode(wire_index=1, gate_index=0)
    )
    edge2 = DAGEdge(
        DagNode(wire_index=3, gate_index=4), DagNode(wire_index=2, gate_index=2)
    )

    with pytest.raises(ValueError):  # noqa: PT011
        edge1 | edge2


def test_dage_edges_common_wire_distance():
    edge1 = DAGEdge(
        DagNode(wire_index=0, gate_index=0), DagNode(wire_index=1, gate_index=0)
    )
    edge2 = DAGEdge(
        DagNode(wire_index=3, gate_index=4), DagNode(wire_index=1, gate_index=2)
    )

    p, q = edge1 | edge2
    # returned nodes are ordered
    assert p < q
    # returned nodes can be queried for distance in terms of their gate indices
    assert q - p == 2
