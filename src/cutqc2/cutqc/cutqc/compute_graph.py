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

    def _to_rustworkx(self):
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
        graph = self._to_rustworkx()
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
