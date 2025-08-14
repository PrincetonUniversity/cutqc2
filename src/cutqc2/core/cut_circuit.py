from pathlib import Path
from copy import deepcopy
import itertools
import logging
import numpy as np
from typing import Self
from dataclasses import dataclass
import warnings
from matplotlib import pyplot as plt
import zarr
from mpi4py import MPI

from qiskit import QuantumCircuit
from qiskit.qasm3 import loads
from qiskit.circuit.operation import Operation
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import Qubit, QuantumRegister, CircuitInstruction
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGCircuit

from cutqc2.cutqc.cutqc.evaluator import run_subcircuit_instances, attribute_shots
from cutqc2.cutqc.cutqc.compute_graph import ComputeGraph
from cutqc2.cutqc.helper_functions.non_ibmq_functions import evaluate_circ
from cutqc2.cutqc.helper_functions.conversions import quasi_to_real
from cutqc2.cutqc.helper_functions.metrics import MSE
from cutqc2.core.dag import DagNode, DAGEdge
from cutqc2.core.utils import chunked, merge_prob_vector, permute_bits_vectorized
from cutqc2.core.dynamic_definition import DynamicDefinition


logger = logging.getLogger(__name__)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

MPI_WORK_TAG = 1
MPI_DONE_TAG = 2
MPI_RESULT_TAG = 3


@dataclass
class Instruction:
    op: Operation
    qarg0: int | None = None  # index of the first qubit in the subcircuit
    qarg1: int | None = None  # index of the second qubit in the subcircuit

    def max_qarg(self) -> int:
        """
        Get the maximum qarg index used in this instruction.
        """
        if self.qarg1 is not None:
            return max(self.qarg0, self.qarg1)
        return self.qarg0


class WireCutGate(UnitaryGate):
    """
    Custom gate to represent a wire cut in a quantum circuit.
    """

    def __init__(self):
        super().__init__(data=[[1, 0], [0, 1]], num_qubits=1, label="//")
        # The super constructor initializes name as "unitary" - use our own
        self.name = "cut"


class CutCircuit:
    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        circuit_qasm3: str | None = None,
        add_labels: bool = True,
        preallocate: bool = True,
    ):
        if circuit is None:
            assert circuit_qasm3 is not None
            circuit = loads(circuit_qasm3)
        self.check_valid(circuit)

        self.raw_circuit = circuit.copy()
        self.unlabeled_circuit = circuit.copy()
        if add_labels:
            self.circuit = self.get_labeled_circuit(circuit.copy())
        else:
            self.circuit = circuit.copy()

        self.inter_wire_dag = self.get_inter_wire_dag(self.circuit)
        self.inter_wire_dag_metadata = self.get_dag_metadata(self.inter_wire_dag)

        # location of cuts, as (<wire_index>, <gate_index>) tuples
        self.cuts: list[tuple[int, int]] = []
        self.subcircuits: list[QuantumCircuit] = []

        # DAGEdge pairs that represent the cuts and subcircuits
        self.cut_dagedgepairs: list[tuple[DAGEdge, DAGEdge]] = []
        self.subcircuit_dagedges: list[list[DAGEdge]] = []

        self.complete_path_map: dict[Qubit, list[dict]] = {}
        self._reconstruction_qubit_order = None

        self.dynamic_definition: DynamicDefinition | None = None
        self.probabilities: np.ndarray | None = None

        self.preallocated: np.ndarray | zarr.Array | None = None
        if preallocate:
            self.preallocated = zarr.zeros(2**self.circuit.num_qubits, dtype="float32")

    def __str__(self):
        return str(self.unlabeled_circuit.draw(output="text", fold=-1))

    def __len__(self):
        return len(self.subcircuits)

    def __iter__(self):
        return iter(self.subcircuits)

    def __getitem__(self, item):
        return self.subcircuits[item]

    @staticmethod
    def check_valid(circuit: QuantumCircuit):
        unitary_factors = circuit.num_unitary_factors()
        assert unitary_factors == 1, (
            f"Input circuit is not fully connected thus does not need cutting. Number of unitary factors = {unitary_factors}"
        )

        assert circuit.num_clbits == 0, (
            "Please remove classical bits from the circuit before cutting"
        )
        dag = circuit_to_dag(circuit)
        for op_node in dag.topological_op_nodes():
            assert len(op_node.qargs) <= 2, (
                "CutQC currently does not support >2-qubit gates"
            )
            assert op_node.op.name != "barrier", (
                "Please remove barriers from the circuit before cutting"
            )

    @staticmethod
    def from_file(filepath: str | Path, *args, **kwargs) -> Self:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Keep imports local to this function
        from cutqc2.io.h5 import h5_to_cut_circuit
        from cutqc2.io.zarr import zarr_to_cut_circuit

        supported_formats = {".h5": h5_to_cut_circuit, ".zarr": zarr_to_cut_circuit}
        assert filepath.suffix in supported_formats, "Unsupported format"
        return supported_formats[filepath.suffix](filepath, *args, **kwargs)

    @staticmethod
    def get_labeled_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Get a labeled version of the circuit where each instruction is labeled
        """
        labeled_instructions = []
        for i, instr in enumerate(list(circuit.data)):
            label = f"{i:04d}"
            new_op = instr.operation.copy().to_mutable()
            new_op.label = label
            instr = CircuitInstruction(
                operation=new_op, qubits=instr.qubits, clbits=instr.clbits
            )
            labeled_instructions.append(instr)

        labeled_circuit = QuantumCircuit.from_instructions(
            labeled_instructions, qubits=circuit.qubits, clbits=circuit.clbits
        )
        labeled_circuit.qregs = circuit.qregs

        return labeled_circuit

    @staticmethod
    def get_inter_wire_dag(circuit: QuantumCircuit) -> DAGCircuit:
        """
        Get the dag for the stripped version of a circuit where we only
        preserve gates that span two wires.
        """
        dag = DAGCircuit()
        for qreg in circuit.qregs:
            dag.add_qreg(qreg)

        for vertex in circuit_to_dag(circuit).topological_op_nodes():
            if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
                dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
        return dag

    @staticmethod
    def get_dag_metadata(dag: DAGCircuit) -> dict:
        edges = []
        node_name_ids = {}
        id_node_names = {}
        id_to_dag_edge = {}
        vertex_ids = {}
        curr_node_id = 0
        qubit_gate_counter = {}

        for qubit in dag.qubits:
            qubit_gate_counter[qubit] = 0

        for vertex in dag.topological_op_nodes():
            if len(vertex.qargs) != 2:
                raise Exception("vertex does not have 2 qargs!")

            arg0, arg1 = vertex.qargs

            dag_edge = DAGEdge(
                DagNode(
                    name=vertex.label,
                    wire_index=arg0._index,
                    gate_index=qubit_gate_counter[arg0],
                ),
                DagNode(
                    name=vertex.label,
                    wire_index=arg1._index,
                    gate_index=qubit_gate_counter[arg1],
                ),
            )
            vertex_name = str(dag_edge)

            qubit_gate_counter[arg0] += 1
            qubit_gate_counter[arg1] += 1
            if vertex_name not in node_name_ids and hash(vertex) not in vertex_ids:
                node_name_ids[vertex_name] = curr_node_id
                id_node_names[curr_node_id] = vertex_name
                id_to_dag_edge[curr_node_id] = dag_edge
                vertex_ids[hash(vertex)] = curr_node_id

                curr_node_id += 1

        for u, v, _ in dag.edges():
            if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
                u_id = vertex_ids[hash(u)]
                v_id = vertex_ids[hash(v)]
                edges.append((u_id, v_id))

        n_vertices = dag.size()

        return {
            "n_vertices": n_vertices,
            "edges": edges,
            "vertex_ids": node_name_ids,
            "id_vertices": id_node_names,
            "id_to_dag_edge": id_to_dag_edge,
        }

    @staticmethod
    def get_initializations(
        paulis: list[str], legacy: bool = True
    ) -> list[tuple[int, tuple[str]]]:
        """
        Get coefficients and kets used for each term in the expansion of the
        trace operators for each of the Pauli bases (eq. 2 in paper).

                 0   1   +   i
            0    1
            I    1   1
            X   -1  -1   2
            Y   -1  -1       2
            Z    1  -1

        """

        if legacy:
            from cutqc2.legacy.cutqc.cutqc.post_process_helper import (
                get_instance_init_meas,
                convert_to_physical_init,
            )

            initializations = paulis
            measurements = []  # not used
            results = get_instance_init_meas(initializations, measurements)
            coeffs_kets = [convert_to_physical_init(result[0]) for result in results]
            return coeffs_kets
        terms = {
            "zero": {"zero": 1},
            "I": {"zero": 1, "one": 1},
            "X": {"zero": -1, "one": -1, "plus": 2},
            "Y": {"zero": -1, "one": -1, "plusI": 2},
            "Z": {"zero": 1, "one": -1},
        }

        substitution_lists = [terms[pauli].items() for pauli in paulis]

        result = []
        for combo in itertools.product(*substitution_lists):
            coeff = 1
            labels = []
            for label, c in combo:
                coeff *= c
                labels.append(label)
            result.append((coeff, tuple(labels)))
        return result

    def get_counter(self):
        O_rho_pairs = []
        for input_qubit in self.complete_path_map:
            path = self.complete_path_map[input_qubit]
            if len(path) > 1:
                for path_ctr, item in enumerate(path[:-1]):
                    O_qubit_tuple = item
                    rho_qubit_tuple = path[path_ctr + 1]
                    O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))

        counter = {}
        for subcircuit_idx, subcircuit in enumerate(self.subcircuits):
            counter[subcircuit_idx] = {
                "effective": subcircuit.num_qubits,
                "rho": 0,
                "O": 0,
                "d": subcircuit.num_qubits,
                "depth": subcircuit.depth(),
                "size": subcircuit.size(),
            }
        for pair in O_rho_pairs:
            O_qubit, rho_qubit = pair
            counter[O_qubit["subcircuit_idx"]]["effective"] -= 1
            counter[O_qubit["subcircuit_idx"]]["O"] += 1
            counter[rho_qubit["subcircuit_idx"]]["rho"] += 1

        return counter

    def find_cuts(
        self,
        max_subcircuit_width: int,
        max_cuts: int,
        num_subcircuits: list[int],
        max_subcircuit_cuts: int,
        subcircuit_size_imbalance: int,
    ):
        from cutqc2.cutqc.cutqc.cutter import MIP_Model

        n_vertices, edges, vertex_ids, id_vertices, id_to_dag_edge = (
            self.inter_wire_dag_metadata["n_vertices"],
            self.inter_wire_dag_metadata["edges"],
            self.inter_wire_dag_metadata["vertex_ids"],
            self.inter_wire_dag_metadata["id_vertices"],
            self.inter_wire_dag_metadata["id_to_dag_edge"],
        )

        num_qubits = self.circuit.num_qubits
        for num_subcircuit in num_subcircuits:
            logger.info(f"Trying with {num_subcircuit} subcircuits")
            if (
                num_subcircuit * max_subcircuit_width - (num_subcircuit - 1)
                < num_qubits
                or num_subcircuit > num_qubits
                or max_cuts + 1 < num_subcircuit
            ):
                continue

            mip_model = MIP_Model(
                n_vertices=n_vertices,
                edges=edges,
                vertex_ids=vertex_ids,
                id_vertices=id_vertices,
                id_to_dag_edge=id_to_dag_edge,
                num_subcircuit=num_subcircuit,
                max_subcircuit_width=max_subcircuit_width,
                max_subcircuit_cuts=max_subcircuit_cuts,
                subcircuit_size_imbalance=subcircuit_size_imbalance,
                num_qubits=num_qubits,
                max_cuts=max_cuts,
            )

            if mip_model.solve():
                return mip_model.cut_edges_pairs, mip_model.subcircuits
            else:
                continue
        raise RuntimeError("No viable cuts found")

    def add_cut_at_position(self, wire_index: int, gate_index: int):
        cut_qubit = self.circuit.qubits[wire_index]

        cut_wire_position = 0
        found = False
        for i, instr in enumerate(self.circuit.data):
            if cut_qubit in instr.qubits:  # we're on the right wire
                if cut_wire_position == gate_index:
                    cut_instr = CircuitInstruction(WireCutGate(), qubits=(cut_qubit,))
                    self.unlabeled_circuit.data.insert(i + 1, cut_instr)
                    found = True
                    break
                cut_wire_position += 1

        if found:
            self.cuts.append((wire_index, gate_index))
        else:
            raise ValueError(
                f"Gate index '{gate_index}' or wire {wire_index} not found in the circuit. Cannot add cut."
            )

    def add_cut(self, label: str, wire_index: int, gate_index: int):
        """
        Add a cut to the circuit at the position of the instruction with the specified label.
        Args:
            label: The label of the instruction after which the cut should be made.
            wire_index: The index of the wire where the cut should be made.
        """
        warnings.warn(
            "Method `add_cut` is deprecated. Use `add_cut_at_position` instead."
        )
        cut_qubit = self.circuit.qubits[wire_index]
        gate_counter = {qubit: 0 for qubit in self.circuit.qubits}

        found = False
        for i, instr in enumerate(self.circuit.data):
            if instr.operation.label == label:
                cut_qubit = self.circuit.qubits[wire_index]
                cut_instr = CircuitInstruction(WireCutGate(), qubits=(cut_qubit,))
                # insert the cut instruction right after the current instruction
                self.circuit.data.insert(i + 1, cut_instr)
                # Also add to unlabeled circuit (for visualization purposes)
                self.unlabeled_circuit.data.insert(i + 1, cut_instr)
                found = True
                break
            else:
                for qubit in instr.qubits:
                    gate_counter[qubit] += 1

        if found:
            gate_index = gate_counter[cut_qubit]
            self.cuts.append((wire_index, gate_index))
        else:
            raise ValueError(
                f"Label '{label}' or wire {wire_index} not found in the circuit. Cannot add cut."
            )

    def add_cuts(self, cut_edges: list[tuple[DAGEdge, DAGEdge]]):
        # validate cut_edges
        for cut_edge in cut_edges:
            p, q = cut_edge[0] | cut_edge[1]
            if q - p != 1:
                raise ValueError(
                    "Invalid cut - the cut edge does not connect two adjacent gates."
                )
            self.cut_dagedgepairs.append(cut_edge)
            self.add_cut_at_position(p.wire_index, p.gate_index)

    def add_cuts_and_generate_subcircuits(
        self, cut_edges: list[tuple[DAGEdge, DAGEdge]], subcircuits: list[list[DAGEdge]]
    ):
        self.add_cuts(cut_edges=cut_edges)
        self.subcircuit_dagedges = subcircuits

        node_label_to_subcircuits: dict[str, int] = {}
        for subcircuit_i, dag_edges in enumerate(subcircuits):
            for dag_edge in dag_edges:
                # Both `source` and `dest` of dag_edge have the same `name`.
                # We arbirarily use `source` here.
                node_label_to_subcircuits[dag_edge.source.name] = subcircuit_i

        n_subcircuits = len(subcircuits)

        subcircuit_instructions: dict[int, Instruction] = {
            j: [] for j in range(n_subcircuits)
        }
        next_subcircuit_wire_index: dict[int, int] = {
            j: 0 for j in range(n_subcircuits)
        }

        # subcircuit_i: {wire_index: list of qubit indices}
        subcircuit_map: dict[int, dict[int, list[int]]] = {
            j: {} for j in range(n_subcircuits)
        }

        # wire_index: list of <subcircuit_i, subcircuit_qubit_index> tuples
        complete_path_map: dict[int, list[tuple[int, int]]] = {
            q: [] for q in range(self.circuit.num_qubits)
        }

        # What is the last subcircuit index we saw on a given wire?
        current_subciruit_on_wire: dict[int, int] = {
            q: None for q in range(self.circuit.num_qubits)
        }

        # Instructions on a qubit wire for which we haven't assigned a
        # subcircuit yet.
        pending_instructions_on_wire: dict[int, list[Instruction]] = {
            q: [] for q in range(self.circuit.num_qubits)
        }

        dag = circuit_to_dag(self.circuit)
        for j, op_node in enumerate(dag.topological_op_nodes()):
            op = deepcopy(op_node.op)
            op.label = ""

            if op_node.label in node_label_to_subcircuits:
                # We're looking at a 2-qubit gate, for which we have a subcircuit index
                assert len(op_node.qargs) == 2
                wire_index0, wire_index1 = (
                    op_node.qargs[0]._index,
                    op_node.qargs[1]._index,
                )
                subcircuit_i = node_label_to_subcircuits[op_node.label]

                subcircuit_wire_indices = []
                for wire_index in (wire_index0, wire_index1):
                    prev_subcircuit_i = current_subciruit_on_wire[wire_index]
                    # If the subcircuit on this wire has changed, then we have
                    # a cut.
                    if (
                        prev_subcircuit_i is not None
                        and prev_subcircuit_i != subcircuit_i
                    ):
                        # For the previous subcircuit on this wire,
                        # add a new qubit wire for this wire index.
                        subcircuit_wire_index = next_subcircuit_wire_index[
                            prev_subcircuit_i
                        ]

                        # Find all used subcircuit qubits in the previous subcircuit.
                        prev_subcircuit_used_wires = [
                            j
                            for wires in subcircuit_map[prev_subcircuit_i].values()
                            for j in wires
                        ]
                        # If the previous subcircuit has used this qubit,
                        # make room for more.
                        if subcircuit_wire_index in prev_subcircuit_used_wires:
                            next_subcircuit_wire_index[prev_subcircuit_i] += 1

                        subcircuit_map[prev_subcircuit_i][wire_index].append(
                            subcircuit_wire_index
                        )

                    current_subciruit_on_wire[wire_index] = subcircuit_i

                    subcircuit_wire_index = subcircuit_map[subcircuit_i].get(wire_index)
                    if subcircuit_wire_index is None:
                        subcircuit_wire_index = next_subcircuit_wire_index[subcircuit_i]
                        subcircuit_map[subcircuit_i][wire_index] = [
                            subcircuit_wire_index
                        ]
                        next_subcircuit_wire_index[subcircuit_i] += 1
                    else:
                        subcircuit_wire_index = subcircuit_wire_index[-1]

                    # Flush any pending instructions on this wire to this
                    # subcircuit's instructions
                    while pending_instructions_on_wire[wire_index]:
                        pending = pending_instructions_on_wire[wire_index].pop(0)
                        pending.qarg0 = subcircuit_wire_index
                        subcircuit_instructions[subcircuit_i].append(pending)

                    # If the current entry in the complete path map is not
                    # the same as the last one, append it.
                    if len(complete_path_map[wire_index]) == 0 or complete_path_map[
                        wire_index
                    ][-1] != (subcircuit_i, subcircuit_wire_index):
                        complete_path_map[wire_index].append(
                            (subcircuit_i, subcircuit_wire_index)
                        )

                    subcircuit_wire_indices.append(subcircuit_wire_index)

                # Add the instruction to the subcircuit
                instr = Instruction(
                    op=op,
                    qarg0=subcircuit_wire_indices[0],
                    qarg1=subcircuit_wire_indices[1],
                )
                subcircuit_instructions[subcircuit_i].append(instr)

            else:
                assert len(op_node.qargs) == 1
                wire_index0 = op_node.qargs[0]._index

                # ignore cut nodes
                if op_node.name == "cut":
                    continue

                # We're looking at a regular single-qubit gate
                if (subcircuit_i := current_subciruit_on_wire[wire_index0]) is None:
                    instr = Instruction(
                        op=op,
                        qarg0=None,  # will be filled in later during flushing
                        qarg1=None,
                    )
                    pending_instructions_on_wire[wire_index0].append(instr)
                else:
                    instr = Instruction(
                        op=op,
                        qarg0=subcircuit_map[subcircuit_i][wire_index0][-1],
                        qarg1=None,
                    )
                    subcircuit_instructions[subcircuit_i].append(instr)

        # Create actual subcircuit from `subcircuit_instructions`
        for subcircuit_i, instrs in subcircuit_instructions.items():
            subcircuit_size = max(instr.max_qarg() for instr in instrs) + 1
            subcircuit = QuantumCircuit(subcircuit_size, name="q")
            qreg = QuantumRegister(subcircuit_size, "q")

            for instr in instrs:
                qargs = tuple(
                    Qubit(qreg, q) for q in (instr.qarg0, instr.qarg1) if q is not None
                )

                subcircuit.append(instruction=instr.op, qargs=qargs, cargs=None)

            self.subcircuits.append(subcircuit)

        self.complete_path_map = {qubit: [] for qubit in self.circuit.qubits}
        for wire_index, path in complete_path_map.items():
            qubit = self.circuit.qubits[wire_index]
            for subcircuit_i, qubit_index in path:
                self.complete_path_map[qubit].append(
                    {
                        "subcircuit_idx": subcircuit_i,
                        "subcircuit_qubit": self.subcircuits[subcircuit_i].qubits[
                            qubit_index
                        ],
                    }
                )

        # book-keeping tasks
        self.populate_compute_graph()
        self.populate_subcircuit_entries()

    @property
    def num_cuts(self) -> int:
        return len(self.cuts)

    @property
    def reconstruction_qubit_order(self) -> dict[int, list[int]]:
        subcircuit_out_qubits = {
            subcircuit_idx: [] for subcircuit_idx in range(len(self))
        }
        for input_qubit in self.complete_path_map:
            path = self.complete_path_map[input_qubit]
            output_qubit = path[-1]
            subcircuit_out_qubits[output_qubit["subcircuit_idx"]].append(
                (
                    output_qubit["subcircuit_qubit"],
                    self.circuit.qubits.index(input_qubit),
                )
            )
        for subcircuit_idx in subcircuit_out_qubits:
            subcircuit_out_qubits[subcircuit_idx] = sorted(
                subcircuit_out_qubits[subcircuit_idx],
                key=lambda x: self[subcircuit_idx].qubits.index(x[0]),
                reverse=True,
            )
            subcircuit_out_qubits[subcircuit_idx] = [
                x[1] for x in subcircuit_out_qubits[subcircuit_idx]
            ]
        return subcircuit_out_qubits

    @reconstruction_qubit_order.setter
    def reconstruction_qubit_order(self, value: dict[int, list[int]]):
        self._reconstruction_qubit_order = deepcopy(value)

    def reconstruction_flat_qubit_order(self) -> np.array:
        reconstruction_qubit_order = self.reconstruction_qubit_order
        result = []
        for subcircuit in self.smart_order:
            _result = reconstruction_qubit_order[subcircuit]
            result.extend(_result)
        perm = np.argsort(result)[::-1]
        return perm

    def cut(
        self,
        max_subcircuit_width: int,
        max_cuts: int,
        num_subcircuits: list[int],
        max_subcircuit_cuts: int,
        subcircuit_size_imbalance: int,
        generate_subcircuits: bool = True,
    ):
        cut_edges_pairs, subcircuits = self.find_cuts(
            max_subcircuit_width=max_subcircuit_width,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            max_subcircuit_cuts=max_subcircuit_cuts,
            subcircuit_size_imbalance=subcircuit_size_imbalance,
        )

        self.add_cuts_and_generate_subcircuits(
            cut_edges=cut_edges_pairs, subcircuits=subcircuits
        )

    def run_subcircuits(
        self,
        subcircuits: list[int] | None = None,
        backend: str = "statevector_simulator",
    ):
        subcircuits = subcircuits or range(len(self))
        for subcircuit in subcircuits:
            logger.info(f"Running subcircuit {subcircuit} on backend: {backend}")
            subcircuit_measured_probs = run_subcircuit_instances(
                subcircuit=self[subcircuit],
                subcircuit_instance_init_meas=self.subcircuit_instances[subcircuit],
                backend=backend,
            )
            self.subcircuit_entry_probs[subcircuit] = attribute_shots(
                subcircuit_measured_probs=subcircuit_measured_probs,
                subcircuit_entries=self.subcircuit_entries[subcircuit],
            )
            self.subcircuit_packed_probs[subcircuit] = self.get_packed_probabilities(
                subcircuit
            )

    def get_packed_probabilities(
        self, subcircuit_i: int, qubit_spec: str | None = None
    ) -> np.ndarray:
        n_prob_vecs: int = sum(
            [subcircuit_i in (e[0], e[1]) for e in self.compute_graph.edges]
        )
        prob_vec_length: int = (
            qubit_spec.count("A")
            if qubit_spec is not None
            else self.compute_graph.nodes[subcircuit_i]["effective"]
        )
        probs = np.zeros(((4,) * n_prob_vecs + (2**prob_vec_length,)), dtype="float32")

        for k, v in self.subcircuit_entry_probs[subcircuit_i].items():
            # we store probabilities as the flat value of init/meas, without the unused locations,
            # with I=0, X=1, Y=2, Z=3.
            # So, for example, index (0, 1, 2, 0) might correspond to any of:
            #    ('zero', 'I', 'X'), ('comp', 'Y', 'comp', 'I')
            #              0    1              2            0
            #    ('zero', 'I', 'zero', 'X', 'comp'), ('Y', 'comp', 'I', 'comp')
            #              0            1              2            0
            # etc.
            # The exact form can be determined by the number of in-degrees and out-degrees
            # of the subcircuit 'node' in the computation graph, as well as the O- and rho- qubits
            # in the 'edges' of the computation graph.
            index = tuple(
                [
                    "IXYZ".index(x)
                    for x in list(k[0]) + list(k[1])
                    if x not in ("zero", "comp")
                ]
            ) + (Ellipsis,)

            if qubit_spec is None:
                probs[index] = v
            else:
                probs[index] = merge_prob_vector(v, qubit_spec)
        return probs

    def get_all_subcircuit_packed_probs(
        self, qubit_specs: dict[int, str] | None = None
    ) -> dict[int, np.ndarray]:
        result = {}
        for i in range(len(self)):
            result[i] = self.get_packed_probabilities(
                i, qubit_spec=qubit_specs.get(i) if qubit_specs else None
            )
        return result

    def get_subcircuit_effective_qubits(self, qubit_spec: str | None = None):
        effective_qubits = []
        for node in self.smart_order:
            effective_qubits.append(self.compute_graph.nodes[node]["effective"])

        if qubit_spec is None:
            active_qubits = np.sum(effective_qubits)
            qubit_spec = "A" * active_qubits
        else:
            active_qubits = qubit_spec.count("A")

        starts = [0]
        for length in effective_qubits[:-1]:
            starts.append(starts[-1] + length)
        ends = [start + length for start, length in zip(starts, effective_qubits)]

        effective_qubits_dict = {}
        for j, start, end in zip(self.smart_order, starts, ends):
            effective_qubits_dict[j] = qubit_spec[start:end]
        return effective_qubits_dict, active_qubits

    def _compute_probabilities(
        self,
        active_qubits,
        subcircuit_packed_probs,
        initializations_list,
        total_initializations: int,
    ) -> np.array:
        logger.info(f"Processing {total_initializations} initializations")
        logger.info(f"{active_qubits=}")

        result = np.zeros(2**active_qubits, dtype=np.float32)

        for j, initializations in enumerate(initializations_list):
            if (j + 1) % 10_000 == 0:
                logger.info(
                    f"Processed {j + 1}/{total_initializations} initializations"
                )

            # `itertools.product` causes the rightmost element to advance on
            # every iteration, to maintain lexical ordering. (00, 01, 10 ...)
            # We wish to 'count up', with the 0th index advancing fastest,
            # so we reverse the obtained tuple from `itertools.product`.
            initializations = initializations[::-1]
            measurements = np.array(initializations)[self.in_to_out_mask]

            initialization_probabilities = None
            for subcircuit in self.smart_order:
                subcircuit_initializations = tuple(
                    initializations[
                        self.in_starts[subcircuit] : self.in_starts[subcircuit + 1]
                    ]
                )
                subcircuit_measurements = tuple(
                    measurements[
                        self.out_starts[subcircuit] : self.out_starts[subcircuit + 1]
                    ]
                )
                subcircuit_index = (
                    subcircuit_initializations + subcircuit_measurements
                ) + (Ellipsis,)
                subcircuit_probabilities = subcircuit_packed_probs[subcircuit][
                    subcircuit_index
                ]

                initialization_probabilities = (
                    np.kron(initialization_probabilities, subcircuit_probabilities)
                    if initialization_probabilities is not None
                    else subcircuit_probabilities
                )
            result += initialization_probabilities

        return result

    def compute_probabilities(self, qubit_spec: str | None = None) -> np.array:
        logger.info(f"Computing probabilities for qubit spec {qubit_spec}")

        effective_qubits_dict, active_qubits = self.get_subcircuit_effective_qubits(
            qubit_spec
        )
        subcircuit_packed_probs = self.get_all_subcircuit_packed_probs(
            qubit_specs=effective_qubits_dict
        )

        total_work = self.n_basis ** sum(self.in_degrees)
        gen = itertools.product(range(self.n_basis), repeat=sum(self.in_degrees))
        num_workers = mpi_size - 1
        active_workers = 0

        if mpi_rank == 0:
            if num_workers == 0:
                # No workers, just do the work locally
                result = self._compute_probabilities(
                    active_qubits,
                    subcircuit_packed_probs,
                    gen,
                    total_initializations=total_work,
                )
            else:
                logger.info(f"Using {num_workers} workers")
                gen = chunked(gen, chunk_size=total_work // mpi_size)
                result = np.zeros(2**active_qubits, dtype=np.float32)

                # Initially send one work item to each worker
                for worker_rank in range(1, mpi_size):
                    try:
                        work = next(gen)
                        mpi_comm.send(
                            [active_qubits, subcircuit_packed_probs, work, len(work)],
                            dest=worker_rank,
                            tag=MPI_WORK_TAG,
                        )
                        active_workers += 1
                    except StopIteration:
                        break

                while active_workers > 0:
                    # Receive results from any worker
                    status = MPI.Status()
                    _result = mpi_comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI_RESULT_TAG, status=status
                    )
                    result += _result

                    worker_rank = status.Get_source()
                    try:
                        work = next(gen)
                        mpi_comm.send(
                            [active_qubits, subcircuit_packed_probs, work, len(work)],
                            dest=worker_rank,
                            tag=MPI_WORK_TAG,
                        )
                    except StopIteration:
                        # No more work; tell this worker to stop
                        mpi_comm.send(None, dest=worker_rank, tag=MPI_DONE_TAG)
                        active_workers -= 1
        else:
            # Worker process
            while True:
                status = MPI.Status()
                work = mpi_comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                if status.Get_tag() == MPI_DONE_TAG:
                    break
                result = self._compute_probabilities(*work)
                mpi_comm.send(result, dest=0, tag=MPI_RESULT_TAG)

        result /= 2**self.num_cuts
        return result

    def postprocess(
        self,
        capacity: int | None = None,
        max_recursion: int = 1,
        quasi: bool = False,
        compute: bool = True,
    ) -> np.ndarray:
        logger.info("Postprocessing the cut circuit")
        if capacity is None:
            capacity = self.compute_graph.effective_qubits
        else:
            capacity = min(capacity, self.compute_graph.effective_qubits)

        self.n_basis: int = 4  # I/X/Y/Z
        n_subcircuits: int = len(self)

        self.smart_order = np.argsort(
            [node["effective"] for node in self.compute_graph.nodes.values()]
        )

        incoming_to_outgoing_graph = self.compute_graph.incoming_to_outgoing_graph()
        self.in_degrees = [
            len([k for k in incoming_to_outgoing_graph if k[0] == subcircuit])
            for subcircuit in range(n_subcircuits)
        ]
        out_degrees = [
            len([v for v in incoming_to_outgoing_graph.values() if v[0] == subcircuit])
            for subcircuit in range(n_subcircuits)
        ]

        self.in_starts = np.insert(np.cumsum(self.in_degrees), 0, 0)
        self.out_starts = np.insert(np.cumsum(out_degrees), 0, 0)

        in_to_out_permutation = []
        out_indices = {}
        counter = 0
        for subcircuit in range(n_subcircuits):
            out_indices[subcircuit] = list(
                range(counter, counter + out_degrees[subcircuit])
            )
            counter += out_degrees[subcircuit]

        for subcircuit in range(n_subcircuits):
            from_subcircuits = [
                v for k, v in incoming_to_outgoing_graph.items() if k[0] == subcircuit
            ]
            for from_subcircuit, from_qubit in from_subcircuits:
                in_to_out_permutation.append(out_indices[from_subcircuit][from_qubit])
        self.in_to_out_mask = np.argsort(in_to_out_permutation)

        # reset old computations before proceeding
        if self.preallocated is not None:
            self.preallocated[:] = 0

        self.dynamic_definition = DynamicDefinition(
            num_qubits=self.compute_graph.effective_qubits,
            capacity=capacity,
            prob_fn=self.compute_probabilities,
            preallocated=self.preallocated,
        )
        logger.info("Starting dynamic definition run")
        unmerged_probabilities = self.dynamic_definition.run(
            max_recursion=max_recursion
        )

        logger.info("Permuting bits to match original circuit order")
        reconstructed_probabilities = np.zeros_like(unmerged_probabilities)
        permuted_indices = permute_bits_vectorized(
            arr=np.arange(unmerged_probabilities.size),
            permutation=self.reconstruction_flat_qubit_order(),
            n_bits=self.circuit.num_qubits,
        )
        reconstructed_probabilities[permuted_indices] = unmerged_probabilities

        if not quasi:
            logger.info("Converting quasi to real probabilities")
            reconstructed_probabilities = quasi_to_real(
                quasiprobability=reconstructed_probabilities, mode="nearest"
            )
        self.probabilities = reconstructed_probabilities

        if compute and isinstance(self.probabilities, zarr.Array):
            return self.probabilities[:]
        else:
            return self.probabilities

    def get_ground_truth(self, backend: str) -> np.ndarray:
        logger.info(f"Evaluating ground truth using {backend}")
        return evaluate_circ(circuit=self.raw_circuit, backend=backend)

    def verify(
        self,
        probabilities: np.ndarray,
        backend: str = "statevector_simulator",
        atol: float = 1e-10,
        raise_error: bool = True,
    ) -> float:
        logger.info("Verifying cut circuit against original circuit")
        ground_truth = self.get_ground_truth(backend)

        approximation_error = (
            MSE(target=ground_truth, obs=probabilities)
            * 2**self.circuit.num_qubits
            / np.linalg.norm(ground_truth) ** 2
        )

        if approximation_error > atol:
            msg = "Difference in cut circuit and uncut circuit is outside of floating point error tolerance"
            if raise_error:
                raise RuntimeError(msg)
            else:
                logger.error(msg)

        return approximation_error

    def populate_compute_graph(self):
        """
        Generate the connection graph among subcircuits
        """
        subcircuits = self.subcircuits

        self.compute_graph = ComputeGraph()
        counter = self.get_counter()
        for subcircuit_idx, subcircuit_attributes in counter.items():
            subcircuit_attributes = deepcopy(subcircuit_attributes)
            subcircuit_attributes["subcircuit"] = subcircuits[subcircuit_idx]
            self.compute_graph.add_node(
                subcircuit_idx=subcircuit_idx, attributes=subcircuit_attributes
            )

        for circuit_qubit in self.complete_path_map:
            path = self.complete_path_map[circuit_qubit]
            for counter in range(len(path) - 1):
                upstream_subcircuit_idx = path[counter]["subcircuit_idx"]
                downstream_subcircuit_idx = path[counter + 1]["subcircuit_idx"]
                self.compute_graph.add_edge(
                    u_for_edge=upstream_subcircuit_idx,
                    v_for_edge=downstream_subcircuit_idx,
                    attributes={
                        "O_qubit": path[counter]["subcircuit_qubit"],
                        "rho_qubit": path[counter + 1]["subcircuit_qubit"],
                    },
                )

    def populate_subcircuit_entries(self):
        compute_graph = self.compute_graph

        subcircuit_entries = {}
        subcircuit_instances = {}

        for subcircuit_idx in compute_graph.nodes:
            bare_subcircuit = compute_graph.nodes[subcircuit_idx]["subcircuit"]
            subcircuit_entries[subcircuit_idx] = {}
            subcircuit_instances[subcircuit_idx] = []
            from_edges = compute_graph.get_edges(from_node=subcircuit_idx, to_node=None)
            to_edges = compute_graph.get_edges(from_node=None, to_node=subcircuit_idx)
            subcircuit_edges = from_edges + to_edges
            for subcircuit_edge_bases in itertools.product(
                "IXYZ", repeat=len(subcircuit_edges)
            ):
                subcircuit_entry_init = ["zero"] * bare_subcircuit.num_qubits
                subcircuit_entry_meas = ["comp"] * bare_subcircuit.num_qubits
                for edge_basis, edge in zip(subcircuit_edge_bases, subcircuit_edges):
                    (
                        upstream_subcircuit_idx,
                        downstream_subcircuit_idx,
                        edge_attributes,
                    ) = edge
                    if subcircuit_idx == upstream_subcircuit_idx:
                        O_qubit = edge_attributes["O_qubit"]
                        subcircuit_entry_meas[bare_subcircuit.qubits.index(O_qubit)] = (
                            edge_basis
                        )
                    elif subcircuit_idx == downstream_subcircuit_idx:
                        rho_qubit = edge_attributes["rho_qubit"]
                        subcircuit_entry_init[
                            bare_subcircuit.qubits.index(rho_qubit)
                        ] = edge_basis
                    else:
                        raise IndexError(
                            "Generating entries for a subcircuit. subcircuit_idx should be either upstream or downstream"
                        )

                subcircuit_entry_term = []
                for coeff, paulis in self.get_initializations(subcircuit_entry_init):
                    initializations_and_measurements = (
                        paulis,
                        tuple(subcircuit_entry_meas),
                    )
                    if (
                        initializations_and_measurements
                        not in subcircuit_instances[subcircuit_idx]
                    ):
                        subcircuit_instances[subcircuit_idx].append(
                            initializations_and_measurements
                        )
                    subcircuit_entry_term.append(
                        (coeff, initializations_and_measurements)
                    )

                subcircuit_entries[subcircuit_idx][
                    (tuple(subcircuit_entry_init), tuple(subcircuit_entry_meas))
                ] = subcircuit_entry_term

        self.subcircuit_entries, self.subcircuit_instances = (
            subcircuit_entries,
            subcircuit_instances,
        )
        self.subcircuit_entry_probs = {}
        self.subcircuit_packed_probs = {}

    def to_file(self, filepath: str | Path, *args, **kwargs) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Keep imports local to this function
        from cutqc2.io.h5 import cut_circuit_to_h5
        from cutqc2.io.zarr import cut_circuit_to_zarr

        supported_formats = {".h5": cut_circuit_to_h5, ".zarr": cut_circuit_to_zarr}
        assert filepath.suffix in supported_formats, "Unsupported format"
        return supported_formats[filepath.suffix](self, filepath, *args, **kwargs)

    def plot(
        self, plot_ground_truth: bool = True, xlim: tuple[int, int] | None = None
    ) -> None:
        fig, ax = plt.subplots()
        if plot_ground_truth:
            ground_truth = self.get_ground_truth(backend="statevector_simulator")
            ax.plot(range(len(ground_truth)), ground_truth, linestyle="--", color="r")

        probabilities = self.probabilities[:]
        ax.bar(np.arange(len(probabilities)), probabilities)
        if xlim is not None:
            ax.set_xlim(xlim)

        if self.dynamic_definition is not None:
            ax.set_title(
                f"Capacity {self.dynamic_definition.capacity}, Recursion {self.dynamic_definition.recursion_level}"
            )

        plt.tight_layout()
        plt.show()
