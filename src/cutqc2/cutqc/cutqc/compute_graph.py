from collections import defaultdict
from rustworkx import PyDiGraph
from rustworkx.visualization import mpl_draw


class ComputeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    @property
    def effective_qubits(self):
        return sum(node["effective"] for node in self.nodes.values())

    def add_node(self, subcircuit_idx, attributes):
        self.nodes[subcircuit_idx] = attributes

    def remove_node(self, subcircuit_idx):
        """
        Remove a node from the compute graph
        """
        del self.nodes[subcircuit_idx]

    def add_edge(self, u_for_edge, v_for_edge, attributes):
        self.edges.append((u_for_edge, v_for_edge, attributes))

    def get_edges(self, from_node, to_node):
        """
        Get edges in the graph based on some given conditions:
        1. If from_node is given. Only retain edges from the node.
        2. If to_node is given. Only retain edges to the node.
        """
        edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            match_from_node = from_node is None or u_for_edge == from_node
            match_to_node = to_node is None or v_for_edge == to_node
            if match_from_node and match_to_node:
                edges.append(edge)
        return edges

    def assign_bases_to_edges(self, edge_bases, edges):
        """Assign the edge_bases to edges"""
        for edge_basis, edge in zip(edge_bases, edges):
            assert edge in self.edges
            u_for_edge, v_for_edge, attributes = edge
            attributes["basis"] = edge_basis

    def remove_bases_from_edges(self, edges):
        """Remove the edge_bases from edges"""
        for edge in edges:
            u_for_edge, v_for_edge, attributes = edge
            if "basis" in attributes:
                del attributes["basis"]

    def remove_all_bases(self):
        for edge in self.edges:
            u_for_edge, v_for_edge, attributes = edge
            if "basis" in attributes:
                del attributes["basis"]

    def get_init_meas(self, subcircuit_idx):
        """Get the entry_init, entry_meas for a given node"""
        node_attributes = self.nodes[subcircuit_idx]
        bare_subcircuit = node_attributes["subcircuit"]
        entry_init = ["zero"] * bare_subcircuit.num_qubits
        edges_to_node = self.get_edges(from_node=None, to_node=subcircuit_idx)
        for edge in edges_to_node:
            _, v_for_edge, edge_attributes = edge
            assert v_for_edge == subcircuit_idx
            entry_init[bare_subcircuit.qubits.index(edge_attributes["rho_qubit"])] = (
                edge_attributes["basis"]
            )

        entry_meas = ["comp"] * bare_subcircuit.num_qubits
        edges_from_node = self.get_edges(from_node=subcircuit_idx, to_node=None)
        for edge in edges_from_node:
            u_for_edge, _, edge_attributes = edge
            assert u_for_edge == subcircuit_idx
            entry_meas[bare_subcircuit.qubits.index(edge_attributes["O_qubit"])] = (
                edge_attributes["basis"]
            )
        return (tuple(entry_init), tuple(entry_meas))

    def get_contraction_edges(
        self, leading_subcircuit_idx, contracted_subcircuits_indices
    ):
        """
        Edges connecting the leading subcircuit and any one of the contracted subcircuits
        """
        contraction_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge == leading_subcircuit_idx
                and v_for_edge in contracted_subcircuits_indices
            ):
                contraction_edges.append(edge)
            elif (
                v_for_edge == leading_subcircuit_idx
                and u_for_edge in contracted_subcircuits_indices
            ):
                contraction_edges.append(edge)
        return contraction_edges

    def get_leading_edges(self, leading_subcircuit_idx, contracted_subcircuits_indices):
        """
        Edges only connecting the leading subcircuit
        """
        leading_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge == leading_subcircuit_idx
                and v_for_edge not in contracted_subcircuits_indices
            ):
                leading_edges.append(edge)
            elif (
                v_for_edge == leading_subcircuit_idx
                and u_for_edge not in contracted_subcircuits_indices
            ):
                leading_edges.append(edge)
        return leading_edges

    def get_trailing_edges(
        self, leading_subcircuit_idx, contracted_subcircuits_indices
    ):
        """
        Edges only connecting the contracted subcircuits
        """
        trailing_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge != leading_subcircuit_idx
            ):
                trailing_edges.append(edge)
            elif (
                v_for_edge in contracted_subcircuits_indices
                and u_for_edge != leading_subcircuit_idx
            ):
                trailing_edges.append(edge)
        return trailing_edges

    def get_contracted_edges(self, contracted_subcircuits_indices):
        """
        Edges in between the contracted subcircuits
        """
        contracted_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                contracted_edges.append(edge)
        return contracted_edges

    def to_rustworkx(self):
        """
        Convert the compute graph to a PyDiGraph object from rustworkx
        """
        digraph = PyDiGraph()
        node_indices = {}
        for subc_idx, attrs in self.nodes.items():
            attrs |= {"index": subc_idx}
            node_indices |= {subc_idx: digraph.add_node(attrs)}

        for u_for_edge, v_for_edge, attributes in self.edges:
            digraph.add_edge(node_indices[u_for_edge], node_indices[v_for_edge], attributes)

        return digraph

    def draw(self):
        graph = self.to_rustworkx()
        node_list = graph.node_indices()
        node_size = [graph[i]['effective'] * 200 for i in node_list]
        return mpl_draw(graph, node_list=list(node_list), node_size=node_size, with_labels=True, labels=lambda node: node['index'])


    def incoming_to_outgoing_graph(self) -> dict[tuple[int, int], tuple[int, int]]:
        """
        Get a more compact representation of the Compute Graph as a dict of
        2-tuples to 2-tuples:
        (to_subcircuit, to_subcircuit_qubit) => (from_subcircuit, from_subcircuit_qubit)

        Any "holes" in indexing are plugged, so that the indices of both
        the incoming qubits as well as the outgoing qubits are continuous,
        and start at 0.
        """

        compute_graph = {
            (edge[1], edge[2]["rho_qubit"]._index): (
                edge[0],
                edge[2]["O_qubit"]._index,
            )
            for edge in self.edges
        }

        # Remove "holes" in indexing
        to_qubits = defaultdict(set)
        from_qubits = defaultdict(set)

        for (to_sub, to_qubit), (from_sub, from_qubit) in compute_graph.items():
            to_qubits[to_sub].add(to_qubit)
            from_qubits[from_sub].add(from_qubit)

        to_qubits_remap = {}
        from_qubits_remap = {}
        for subcircuit in to_qubits:
            to_qubits_remap[subcircuit] = {
                old: new for new, old in enumerate(sorted(to_qubits[subcircuit]))
            }
        for subcircuit in from_qubits:
            from_qubits_remap[subcircuit] = {
                old: new for new, old in enumerate(sorted(from_qubits[subcircuit]))
            }

        new_compute_graph = {}
        for (to_sub, to_qubit), (from_sub, from_qubit) in compute_graph.items():
            new_to_qubit = to_qubits_remap[to_sub][to_qubit]
            new_from_qubit = from_qubits_remap[from_sub][from_qubit]
            new_compute_graph[(to_sub, new_to_qubit)] = from_sub, new_from_qubit

        # important! - sort keys by (to_subcircuit, to_qubit)
        new_compute_graph = {
            k: new_compute_graph[k] for k in sorted(new_compute_graph)
        }

        return new_compute_graph
