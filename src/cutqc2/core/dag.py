from itertools import product

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit


class DagNode:
    """
    Represents a node in a quantum circuit DAG (Directed Acyclic Graph),
    corresponding to a specific gate on a specific wire (qubit/register).

    Since only inter-wire gates are important for the cut algorithm,
    any mention of `gate_index` refers to the index wrt inter-wire gates only.

    Attributes:
        wire_index (int): The index of the wire (qubit/register).
        gate_index (int): The index of the gate on the wire.
          Note: `gate_index` assumes that only inter-wire gates are considered.
    """

    @classmethod
    def from_string(cls, s: str):
        """
        Create a DagNode from a string representation.

        Args:
            s (str): The string representation of the node in the format '[wire_index]gate_index'.

        Returns:
            DagNode: The created DagNode instance.
        """
        _, rest = s.split("[")
        wire_index, gate_index = map(int, rest.split("]"))
        return cls(wire_index, gate_index)

    def __init__(self, wire_index: int, gate_index: int):
        """
        Initialize a DagNode.

        Args:
            wire_index (int): The index of the wire (qubit/register).
            gate_index (int): The index of the gate on the wire.
        """
        self.wire_index = wire_index
        self.gate_index = gate_index

    def __str__(self):
        """
        Return a string representation of the DagNode.

        Returns:
            str: The string in the format '[wire_index]gate_index'.
        """
        return f"[{self.wire_index}]{self.gate_index}"

    def __eq__(self, other: "DagNode"):
        return (
            self.wire_index == other.wire_index and self.gate_index == other.gate_index
        )

    def __lt__(self, other: "DagNode"):
        """
        Compare two DagNodes for ordering, first by wire_index, then by gate_index.

        Args:
            other (DagNode): The other DagNode to compare to.

        Returns:
            bool: True if this node is less than the other node.
        """
        if self.wire_index < other.wire_index:
            return True
        if self.wire_index == other.wire_index:
            return self.gate_index < other.gate_index
        return False

    def __sub__(self, other):
        """
        Subtract two DagNodes on the same wire to get the difference in gate indices.

        Args:
            other (DagNode): The other DagNode to subtract.

        Returns:
            int: The difference in gate indices.

        Raises:
            ValueError: If the nodes are on different wires.
        """
        if self.wire_index != other.wire_index:
            raise ValueError("Cannot subtract nodes on different wires")
        return self.gate_index - other.gate_index

    def locate(self, dag_circuit: DAGCircuit) -> tuple[Qubit, int]:
        """
        Locate the position of the DagNode in the DAGCircuit.
        Args:
            dag_circuit (DAGCircuit): The DAGCircuit containing the node.

        Returns:
            tuple[Qubit, int]: A tuple containing the Qubit and the index of the gate on that wire.

        Raises:
            ValueError: If the node cannot be found in the DAGCircuit.
        """
        wire = dag_circuit.wires[self.wire_index]
        multi_wire_gate_i = 0

        for i, gate in enumerate(dag_circuit.nodes_on_wire(wire, only_ops=True)):
            if len(gate.qargs) > 1:
                if multi_wire_gate_i == self.gate_index:
                    return wire, i
                multi_wire_gate_i += 1

        raise ValueError("not found")


class DAGEdge:
    """
    Represents an edge in a quantum circuit DAG, connecting two DagNodes.

    Attributes:
        source (DagNode): The source node of the edge.
        dest (DagNode): The destination node of the edge.
    """

    @classmethod
    def from_string(cls, edge_str: str) -> "DAGEdge":
        """
        Create a DAGEdge from a string representation.

        Args:
            edge_str (str): The string representation of the edge in the format 'source dest'.

        Returns:
            DAGEdge: The created DAGEdge instance.
        """
        source_str, dest_str = edge_str.split()
        return cls(DagNode.from_string(source_str), DagNode.from_string(dest_str))

    def __init__(self, first: DagNode, second: DagNode):
        """
        Initialize a DAGEdge between two DagNodes.

        Args:
            first (DagNode): The first node.
            second (DagNode): The second node.
        """
        self.source, self.dest = sorted((first, second))

    def __str__(self):
        """
        Return a string representation of the DAGEdge.

        Returns:
            str: The string in the format 'source dest'.
        """
        return f"{self.source} {self.dest}"

    def __eq__(self, other: "DAGEdge"):
        return self.source == other.source and self.dest == other.dest

    def __or__(self, other: "DAGEdge") -> tuple[DagNode, DagNode]:
        """
        Find the pair of DagNodes (one from each edge) that share the same wire index.

        Args:
            other (DAGEdge): The other DAGEdge to compare with.

        Returns:
            tuple[DagNode, DagNode]: The pair of nodes with the same wire index, sorted.

        Raises:
            ValueError: If there is no common wire between the two edges.
        """
        for a, b in product((self.source, self.dest), (other.source, other.dest)):
            if a.wire_index == b.wire_index:
                return tuple(sorted((a, b)))

        raise ValueError("No common wire")

    def weight(self) -> int:
        return int(self.source.gate_index == 0) + int(self.dest.gate_index == 0)
