import itertools
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, QuantumRegister, Qubit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.qasm3 import loads

from cutqc2.core.compute_graph import ComputeGraph
from cutqc2.core.dag import DAGEdge, DagNode
from cutqc2.core.dynamic_definition import DynamicDefinition
from cutqc2.core.utils import (
    attribute_shots,
    chunked,
    merge_prob_vector,
    permute_bits,
    run_subcircuit_instances,
)
from cutqc2.cupy import vector_kron
from cutqc2.cutqc.helper_functions.conversions import quasi_to_real
from cutqc2.cutqc.helper_functions.metrics import MSE
from cutqc2.cutqc.helper_functions.non_ibmq_functions import evaluate_circ
from cutqc2.numeric import xp

logger = logging.getLogger(__name__)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


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
        self, circuit: QuantumCircuit | None = None, circuit_qasm3: str | None = None
    ):
        """
        Initialize a cuttable circuit from a `QuantumCircuit` or a QASM3 string.

        Parameters
        ----------
        circuit
            An existing `QuantumCircuit`. If None, `circuit_qasm3` must be provided.
        circuit_qasm3
            QASM3 string for creating a circuit when `circuit` is None.
        """
        if circuit is None:
            assert circuit_qasm3 is not None
            circuit = loads(circuit_qasm3)
        self.check_valid(circuit)

        self.circuit = circuit.copy()

        # A QuantumCircuit object that contains the original circuit with cut gates inserted
        # Here we initialize it, but will add instructions in the `cut` method.
        self.circuit_with_cut_gates = QuantumCircuit(*circuit.qregs)

        self.inter_wire_dag = self.get_inter_wire_dag(self.circuit)
        self.inter_wire_dag_metadata = self.get_dag_metadata(self.inter_wire_dag)

        self.num_cuts = 0
        self.subcircuits: list[QuantumCircuit] = []

        self.subcircuit_dagedges: list[list[DAGEdge]] = []

        self.complete_path_map: dict[int, list[tuple[int, int]]] = {}
        self._reconstruction_qubit_order = None

        self.dynamic_definition: DynamicDefinition | None = None

    def __str__(self):
        """
        Get a text diagram of the circuit with cut gates inserted.

        Returns
        -------
        str
            ASCII drawing of the circuit with cuts.
        """
        return str(self.circuit_with_cut_gates.draw(output="text", fold=-1))

    def __len__(self):
        """Number of generated subcircuits after cutting."""
        return len(self.subcircuits)

    def __iter__(self):
        """Iterate over generated subcircuits."""
        return iter(self.subcircuits)

    def __getitem__(self, item):
        """
        Get a subcircuit by index.

        Parameters
        ----------
        item
            Index of the subcircuit.

        Returns
        -------
        QuantumCircuit
            The requested subcircuit.
        """
        return self.subcircuits[item]

    @property
    def default_filepath(self) -> str:
        """
        Generate a default filepath for saving the cut circuit.
        By default, the filename is of the form:
        {num_qubits}q-{num_cuts}cuts-{num_subcircuits}subcircuits_{timestamp}.zarr
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"{self.circuit.num_qubits}q-{self.num_cuts}c-{len(self)}s_{timestamp}.zarr"
        )

    @staticmethod
    def check_valid(circuit: QuantumCircuit):
        """
        Validate that the input circuit is supported for cutting.

        - Circuit must be fully connected (one unitary factor)
        - No classical bits
        - No barriers
        - Only 1- or 2-qubit gates
        """
        unitary_factors = circuit.num_unitary_factors()
        assert unitary_factors == 1, (
            f"Input circuit is not fully connected thus does not need cutting. Number of unitary factors = {unitary_factors}"
        )

        assert circuit.num_clbits == 0, (
            "Please remove classical bits from the circuit before cutting"
        )
        dag = circuit_to_dag(circuit)
        for op_node in dag.topological_op_nodes():
            assert len(op_node.qargs) <= 2, (  # noqa: PLR2004
                "CutQC currently does not support >2-qubit gates"
            )
            assert op_node.op.name != "barrier", (
                "Please remove barriers from the circuit before cutting"
            )

    @staticmethod
    def from_file(filepath: str | Path, *args, **kwargs) -> Self:
        """
        Load a `CutCircuit` from a file on disk.

        Parameters
        ----------
        filepath
            Path to a saved cut circuit (currently `.zarr`).

        Returns
        -------
        CutCircuit
            A reconstructed `CutCircuit` instance.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Keep imports local to this function
        from cutqc2.io.zarr import zarr_to_cut_circuit

        supported_formats = {".zarr": zarr_to_cut_circuit}
        assert filepath.suffix in supported_formats, "Unsupported format"
        return supported_formats[filepath.suffix](filepath, *args, **kwargs)

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
            if len(vertex.qargs) == 2 and vertex.op.name != "barrier":  # noqa: PLR2004
                dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
        return dag

    @staticmethod
    def get_dag_metadata(dag: DAGCircuit) -> dict:
        """
        Extract metadata required by the cut searcher from a 2-wire-only DAG.

        Returns
        -------
        dict
            Dictionary containing number of vertices, edges, and a mapping from
            vertex id to `DAGEdge` objects.
        """
        edges = []
        node_name_ids = {}
        id_to_dag_edge = {}
        vertex_ids = {}
        curr_node_id = 0
        qubit_gate_counter = {}

        for qubit in dag.qubits:
            qubit_gate_counter[qubit] = 0

        for vertex in dag.topological_op_nodes():
            if len(vertex.qargs) != 2:  # noqa: PLR2004
                raise Exception("vertex does not have 2 qargs!")

            arg0, arg1 = vertex.qargs

            dag_edge = DAGEdge(
                DagNode(
                    wire_index=arg0._index,
                    gate_index=qubit_gate_counter[arg0],
                ),
                DagNode(
                    wire_index=arg1._index,
                    gate_index=qubit_gate_counter[arg1],
                ),
            )
            vertex_name = str(dag_edge)

            qubit_gate_counter[arg0] += 1
            qubit_gate_counter[arg1] += 1
            if vertex_name not in node_name_ids and hash(vertex) not in vertex_ids:
                node_name_ids[vertex_name] = curr_node_id
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
                convert_to_physical_init,
                get_instance_init_meas,
            )

            initializations = paulis
            measurements = []  # not used
            results = get_instance_init_meas(initializations, measurements)
            return [convert_to_physical_init(result[0]) for result in results]
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

    def find_cuts(
        self, max_subcircuit_width: int, max_cuts: int, num_subcircuits: list[int]
    ):
        """
        Search for a feasible partitioning of the circuit into subcircuits.

        Parameters
        ----------
        max_subcircuit_width
            Maximum number of qubits per subcircuit.
        max_cuts
            Maximum number of wire cuts allowed.
        num_subcircuits
            Candidate numbers of subcircuits to try, in order.

        Returns
        -------
        list[list[DAGEdge]]
            A list of subcircuits; each subcircuit is a list of 2-qubit `DAGEdge`s.

        Raises
        ------
        RuntimeError
            If no viable cuts are found.
        """
        from cutqc2.core.mip import MIPCutSearcher

        n_vertices, edges, id_to_dag_edge = (
            self.inter_wire_dag_metadata["n_vertices"],
            self.inter_wire_dag_metadata["edges"],
            self.inter_wire_dag_metadata["id_to_dag_edge"],
        )

        num_qubits = self.circuit.num_qubits
        for num_subcircuit in num_subcircuits:
            logger.info(f"Trying with {num_subcircuit} subcircuits")

            if num_subcircuit > num_qubits:
                logger.info(
                    f"Skipping as more subcircuits ({num_subcircuit}) than qubits ({num_qubits})"
                )
                continue

            if max_cuts + 1 < num_subcircuit:
                logger.info(
                    f"Skipping as not enough cuts ({max_cuts}) to create {num_subcircuit} subcircuits"
                )
                continue

            """
            Each subcircuit can use up to max_subcircuit_width qubits, so in total they
            could cover at most num_subcircuit * max_subcircuit_width qubits. However,
            adjacent subcircuits must overlap by at least 1 qubit to reconnect properly,
            which reduces the distinct qubit coverage by (num_subcircuit - 1).
            If even after accounting for this overlap the total coverage is still less
            than num_qubits, then it's impossible to fit the circuit into this partition.
            """
            if (
                num_subcircuit * max_subcircuit_width - (num_subcircuit - 1)
                < num_qubits
            ):
                logger.info(
                    f"Skipping as cannot fit all qubits ({num_qubits}) into {num_subcircuit} subcircuits with max width {max_subcircuit_width}"
                )
                continue

            mip_model = MIPCutSearcher(
                n_vertices=n_vertices,
                edges=edges,
                id_to_dag_edge=id_to_dag_edge,
                num_subcircuit=num_subcircuit,
                max_subcircuit_width=max_subcircuit_width,
                num_qubits=num_qubits,
                max_cuts=max_cuts,
            )

            if mip_model.solve():
                return mip_model.subcircuits
            continue
        raise RuntimeError("No viable cuts found")

    @property
    def greedy_subcircuit_order(self):
        """
        Order subcircuits by ascending effective width.

        The order of subcircuits in which the reconstructor computes
        the Kronecker products incurs different sizes of carryover vectors
        and affects the total number of floating-point multiplications. This
        ordering approach places the smallest subcircuits first in order to
        minimize the carryover in the size of the vectors.
        Returns
        -------
        np.array
            Indices of subcircuits in ascending order of effective width.
        """
        return np.argsort(
            [node["effective"] for node in self.compute_graph.nodes.values()]
        )

    @property
    def reconstruction_qubit_order(self) -> dict[int, list[int]]:
        """
        Map each subcircuit to the list of original qubit indices it outputs.

        Returns
        -------
        dict[int, list[int]]
            For each subcircuit index, a list of original circuit qubit indices
            in descending subcircuit order used during reconstruction.
        """
        subcircuit_out_qubits = {
            subcircuit_idx: [] for subcircuit_idx in range(len(self))
        }
        for input_qubit, path in self.complete_path_map.items():
            output_subcircuit_i, output_qubit = path[-1]
            subcircuit_out_qubits[output_subcircuit_i].append(
                (
                    output_qubit,
                    input_qubit,
                )
            )
        for subcircuit_idx in subcircuit_out_qubits:  # noqa: PLC0206
            subcircuit_out_qubits[subcircuit_idx] = sorted(
                subcircuit_out_qubits[subcircuit_idx],
                key=lambda x: x[0],
                reverse=True,
            )
            subcircuit_out_qubits[subcircuit_idx] = [
                x[1] for x in subcircuit_out_qubits[subcircuit_idx]
            ]
        return subcircuit_out_qubits

    @reconstruction_qubit_order.setter
    def reconstruction_qubit_order(self, value: dict[int, list[int]]):
        """Override the reconstruction qubit mapping used for output assembly."""
        self._reconstruction_qubit_order = deepcopy(value)

    def reconstruction_flat_qubit_order(self) -> np.array:
        """
        Get the flat permutation of original qubit indices for reconstruction.

        Returns
        -------
        np.array
            A permutation vector mapping positions in the reconstructed state
            to the original circuit qubit order (descending ranks).
        """
        reconstruction_qubit_order = self.reconstruction_qubit_order
        result = []
        for subcircuit in self.greedy_subcircuit_order:
            _result = reconstruction_qubit_order[subcircuit]
            result.extend(_result)

        result = np.array(result)
        # Return the descending ranks of the qubits in the reconstruction order
        return (-result).argsort().argsort()

    def cut(  # noqa: PLR0912, PLR0915
        self,
        max_subcircuit_width: int | None = None,
        max_cuts: int | None = None,
        num_subcircuits: list[int] | None = None,
        subcircuits: list[list[DAGEdge]] | list[list[str]] | None = None,
    ):
        """
        Partition the circuit into subcircuits and build their `QuantumCircuit`s.

        Parameters
        ----------
        max_subcircuit_width
            Maximum qubits per subcircuit (when searching).
        max_cuts
            Maximum number of wire cuts allowed (when searching).
        num_subcircuits
            Candidate counts to try (when searching).
        subcircuits
            Optional precomputed subcircuits (as `DAGEdge` objects or strings).
        """
        if subcircuits is None:
            subcircuits = self.find_cuts(
                max_subcircuit_width=max_subcircuit_width,
                max_cuts=max_cuts,
                num_subcircuits=num_subcircuits,
            )
        elif isinstance(subcircuits[0][0], str):
            subcircuits = [
                [DAGEdge.from_string(s) for s in sublist] for sublist in subcircuits
            ]

        self.subcircuit_dagedges = subcircuits

        wire_and_gate_to_subcircuit: dict[tuple[int, int], int] = {}
        for subcircuit_i, dag_edges in enumerate(subcircuits):
            for dag_edge in dag_edges:
                wire_and_gate_to_subcircuit[
                    dag_edge.source.wire_index, dag_edge.source.gate_index
                ] = subcircuit_i
                wire_and_gate_to_subcircuit[
                    dag_edge.dest.wire_index, dag_edge.dest.gate_index
                ] = subcircuit_i

        n_subcircuits = len(subcircuits)

        # --------------------------------------------------
        # Useful data structures for parsing the circuit
        # --------------------------------------------------

        # mapping from subcircuit index to list of Instructions
        subcircuit_instructions: dict[int, list[Instruction]] = {
            j: [] for j in range(n_subcircuits)
        }
        # next available wire index for each subcircuit
        next_subcircuit_wire_index: dict[int, int] = {
            j: 0 for j in range(n_subcircuits)
        }
        # mapping from subcircuit index to {uncut circuit wire index: subcircuit wire index} mapping
        subcircuit_map: dict[int, dict[int, list[int]]] = {
            j: {} for j in range(n_subcircuits)
        }
        # mapping from uncut circuit wire index to a list of (subcircuit index, subcircuit wire index) tuples
        complete_path_map: dict[int, list[tuple[int, int]]] = {
            q: [] for q in range(self.circuit.num_qubits)
        }
        # mapping from uncut circuit wire index to subcircuit index we last saw on that wire
        current_subciruit_on_wire: dict[int, int | None] = {
            q: None for q in range(self.circuit.num_qubits)
        }
        # mapping from uncut circuit wire index to list of Instructions that
        # haven't been assigned to a subcircuit (yet)
        pending_instructions_on_wire: dict[int, list[Instruction]] = {
            q: [] for q in range(self.circuit.num_qubits)
        }
        # mapping from uncut circuit wire index to number of 2-qubit gates
        # we've seen on that wire so far
        two_qubit_gate_index_on_wire: dict[int, int] = {
            q: 0 for q in range(self.circuit.num_qubits)
        }
        # --------------------------------------------------

        dag = circuit_to_dag(self.circuit)
        for op_node in dag.topological_op_nodes():
            # The new operation that we're constructing
            op = deepcopy(op_node.op)

            if len(op_node.qargs) == 2:  # noqa: PLR2004
                wire_index0 = op_node.qargs[0]._index
                gate_index0 = two_qubit_gate_index_on_wire[wire_index0]
                wire_index1 = op_node.qargs[1]._index
                gate_index1 = two_qubit_gate_index_on_wire[wire_index1]
                if (wire_index0, gate_index0) in wire_and_gate_to_subcircuit:
                    subcircuit_i = wire_and_gate_to_subcircuit[wire_index0, gate_index0]
                elif (wire_index1, gate_index1) in wire_and_gate_to_subcircuit:
                    subcircuit_i = wire_and_gate_to_subcircuit[wire_index1, gate_index1]
                else:
                    raise ValueError("2-qubit gate is not part of any subcircuit")

                subcircuit_wire_indices = []
                for wire_index in (wire_index0, wire_index1):
                    prev_subcircuit_i = current_subciruit_on_wire[wire_index]
                    # If the subcircuit on this wire has changed, then we have
                    # a cut.
                    if (
                        prev_subcircuit_i is not None
                        and prev_subcircuit_i != subcircuit_i
                    ):
                        self.circuit_with_cut_gates.append(
                            instruction=CircuitInstruction(
                                WireCutGate(),
                                qubits=(
                                    self.circuit_with_cut_gates.qubits[wire_index],
                                ),
                            )
                        )
                        self.num_cuts += 1

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

            self.circuit_with_cut_gates.append(
                instruction=CircuitInstruction(op, qubits=op_node.qargs),
                qargs=op_node.qargs,
                cargs=None,
            )

            if len(op_node.qargs) == 2:  # noqa: PLR2004
                two_qubit_gate_index_on_wire[op_node.qargs[0]._index] += 1
                two_qubit_gate_index_on_wire[op_node.qargs[1]._index] += 1

        # We're done parsing all Instructions
        # Create actual subcircuit from `subcircuit_instructions`
        for instrs in subcircuit_instructions.values():
            subcircuit_size = max(instr.max_qarg() for instr in instrs) + 1
            subcircuit = QuantumCircuit(subcircuit_size, name="q")
            qreg = QuantumRegister(subcircuit_size, "q")

            for instr in instrs:
                qargs = tuple(
                    Qubit(qreg, q) for q in (instr.qarg0, instr.qarg1) if q is not None
                )

                subcircuit.append(instruction=instr.op, qargs=qargs, cargs=None)

            self.subcircuits.append(subcircuit)

        self.complete_path_map = complete_path_map

        # book-keeping tasks
        self.populate_compute_graph()
        self.populate_subcircuit_entries()

    def run_subcircuits(
        self,
        subcircuits: list[int] | None = None,
        backend: str = "statevector_simulator",
    ):
        """
        Execute all subcircuits on a backend and collect probability vectors.

        Parameters
        ----------
        subcircuits
            Subcircuit indices to run; defaults to all.
        backend
            Backend name (e.g., "statevector_simulator").
        """
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
        """
        Pack entry probability vectors for a subcircuit into a dense tensor.

        The packed tensor has one 4-sized axis per incident edge (I/X/Y/Z),
        and a final axis of length 2^k for k effective qubits. If `qubit_spec`
        is provided, probabilities are merged accordingly.

        Parameters
        ----------
        subcircuit_i
            Index of the subcircuit.
        qubit_spec
            Optional spec over {"A","M","0","1"} for effective qubits.

        Returns
        -------
        np.ndarray
            Packed probability tensor for the subcircuit.
        """
        # Find the in-degree + out-degree of the subcircuit in the compute graph.
        # This tells us how many probability vector dimensions we need.
        n_prob_vecs: int = sum(
            [subcircuit_i in (e[0], e[1]) for e in self.compute_graph.edges]
        )
        prob_vec_length: int = (
            qubit_spec.count("A")
            if qubit_spec is not None
            else self.compute_graph.nodes[subcircuit_i]["effective"]
        )
        probs = xp.zeros(((4,) * n_prob_vecs + (2**prob_vec_length,)), dtype="float32")

        for k, value in self.subcircuit_entry_probs[subcircuit_i].items():
            value_cp = xp.asarray(value)
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
            index = (
                *("IXYZ".index(x) for x in [*k[0], *k[1]] if x not in ("zero", "comp")),
                Ellipsis,
            )

            if qubit_spec is None:
                probs[index] = value_cp
            else:
                probs[index] = merge_prob_vector(value_cp, qubit_spec)
        return probs

    def _get_subcircuit_effective_qubits(self, qubit_spec: str | None = None):
        """
        Partition a global qubit specification string into per-subcircuit slices.

        Each subcircuit has an 'effective' qubit count stored in the compute graph.
        This function assigns to each subcircuit the portion of the qubit_spec
        that corresponds to its effective qubits.

        Parameters
        ----------
        qubit_spec : str or None
            A string over {"A", "M", "0", "1"} representing the role of each active qubit,
            concatenated across *all* subcircuits in greedy_subcircuit_order.
            If None, a default spec of "A" for each active qubit is used.

        Returns
        -------
        effective_qubits_dict : dict
            Mapping {subcircuit_node: qubit_spec_slice} where the slice is the part
            of the global qubit_spec corresponding to that subcircuit's effective qubits.
        active_qubits : int
            Total number of active ("A") qubits across all subcircuits.
        """
        # Get the effective qubit counts for each subcircuit
        effective_counts = [
            self.compute_graph.nodes[node]["effective"]
            for node in self.greedy_subcircuit_order
        ]

        total_effective = sum(effective_counts)

        # If no spec is provided, default to all active ("A")
        if qubit_spec is None:
            qubit_spec = "A" * total_effective

        # Count how many "A"s there are in the global spec
        active_qubits = qubit_spec.count("A")

        # Compute starting indices for each subcircuit's slice
        starts = [0] + list(itertools.accumulate(effective_counts))[:-1]

        # Assign each subcircuit its slice of the qubit_spec
        effective_qubits_dict = {
            node: qubit_spec[start : start + length]
            for node, start, length in zip(
                self.greedy_subcircuit_order, starts, effective_counts, strict=False
            )
        }

        return effective_qubits_dict, active_qubits

    def _compute_probabilities(
        self,
        active_qubits,
        subcircuit_packed_probs,
        initializations_list,
        log_every: int | None = None,
        total_work: int | None = None,
    ) -> np.array:
        """
        Core routine to accumulate probabilities over all initialization tuples.

        Parameters
        ----------
        active_qubits
            Total number of active qubits across subcircuits.
        subcircuit_packed_probs
            Mapping from subcircuit index to its packed probability tensor.
        initializations_list
            Iterable of initialization tuples over input Pauli bases.
        log_every
            Log progress every this many iterations (optional).
        total_work
            Total number of iterations for progress reporting (optional).

        Returns
        -------
        np.ndarray
            Flat probability vector of size 2^active_qubits.
        """
        result = xp.zeros(2**active_qubits, dtype="float32")

        for j, initializations in enumerate(initializations_list):
            if log_every is not None and j % log_every == 0:
                logger.info(f"  Processed {j}/{total_work} initializations")

            # `itertools.product` causes the rightmost element to advance on
            # every iteration, to maintain lexical ordering. (00, 01, 10 ...)
            # We wish to 'count up', with the 0th index advancing fastest,
            # so we reverse the obtained tuple from `itertools.product`.
            initializations = np.array(initializations)[::-1]  # noqa: PLW2901
            measurements = initializations[self.in_to_out_mask]

            initialization_probabilities = None
            for subcircuit in self.greedy_subcircuit_order:
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

                if initialization_probabilities is not None:
                    initialization_probabilities = vector_kron(
                        initialization_probabilities, subcircuit_probabilities
                    )
                else:
                    initialization_probabilities = subcircuit_probabilities

            result += initialization_probabilities

        return result

    def compute_probabilities(self, qubit_spec: str | None = None) -> np.array:  # noqa: PLR0912, PLR0915
        logger.info(f"Computing probabilities for qubit spec {qubit_spec}")
        """
        Compute the reconstructed probability vector for a given qubit spec.

        This function parallelizes across MPI ranks by distributing chunks of
        input-initialization tuples and reducing the partial results.

        Parameters
        ----------
        qubit_spec
            A string over {"A","M","0","1"} for all effective qubits.

        Returns
        -------
        np.ndarray
            Flat probability vector of size 2^active_qubits.
        """

        effective_qubits_dict, active_qubits = self._get_subcircuit_effective_qubits(
            qubit_spec
        )
        subcircuit_packed_probs = {}
        for i in range(len(self)):
            subcircuit_packed_probs[i] = self.get_packed_probabilities(
                i, qubit_spec=effective_qubits_dict[i]
            )

        total_work = self.n_basis ** sum(self.in_degrees)
        gen = itertools.product(range(self.n_basis), repeat=sum(self.in_degrees))
        num_workers = mpi_size - 1
        active_workers = 0

        MPI_WORK_TAG, MPI_DONE_TAG, MPI_RESULT_TAG = 1, 2, 3
        if mpi_rank == 0:
            if num_workers == 0:
                # No workers, just do the work locally
                result = self._compute_probabilities(
                    active_qubits,
                    subcircuit_packed_probs,
                    gen,
                    log_every=10_000,
                    total_work=total_work,
                )
            else:
                chunk_size = min(8192, total_work // num_workers)
                logger.info(f"{num_workers=}, {chunk_size=}")

                gen = chunked(gen, chunk_size=chunk_size)
                result = xp.zeros(2**active_qubits, "float32")
                processed_work = 0

                # Initially send one work item to each worker
                for worker_rank in range(1, mpi_size):
                    try:
                        work = next(gen)
                        mpi_comm.send(
                            [active_qubits, subcircuit_packed_probs, work],
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
                    processed_work += chunk_size
                    logger.info(
                        f"  Processed {processed_work}/{total_work} initializations"
                    )
                    result += _result

                    worker_rank = status.Get_source()
                    try:
                        work = next(gen)
                        mpi_comm.send(
                            [active_qubits, subcircuit_packed_probs, work],
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

        if mpi_rank == 0:
            result /= 2**self.num_cuts
        else:
            result = None

        result = mpi_comm.bcast(result, root=0)
        mpi_comm.Barrier()
        return result

    def postprocess(
        self, capacity: int | None = None, max_recursion: int = 1
    ) -> np.ndarray:
        """
        Run dynamic definition to reconstruct the full probability vector.

        Parameters
        ----------
        capacity
            Target capacity (#active qubits) for dynamic definition.
        max_recursion
            Maximum recursion depth for bin refinement.

        Returns
        -------
        np.ndarray
            The reconstructed probability vector for the full circuit.
        """
        logger.info("Postprocessing the cut circuit")
        if capacity is None:
            capacity = self.compute_graph.effective_qubits
        else:
            capacity = min(capacity, self.compute_graph.effective_qubits)

        self.n_basis: int = 4  # I/X/Y/Z
        n_subcircuits: int = len(self)

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

        self.dynamic_definition = DynamicDefinition(
            num_qubits=self.compute_graph.effective_qubits,
            capacity=capacity,
            prob_fn=self.compute_probabilities,
        )
        logger.info("Starting dynamic definition run")
        self.dynamic_definition.run(max_recursion=max_recursion)

    def get_probabilities(
        self, full_states: np.ndarray | None = None, quasi: bool = False
    ) -> np.ndarray:
        """
        Get reconstructed probabilities for specific output basis states.

        Parameters
        ----------
        full_states
            Optional list of basis-state indices to query. If None, compute for all
            2^n states (may be memory intensive).
        quasi
            If True, return quasi-probabilities; otherwise project to the nearest
            probability vector.

        Returns
        -------
        np.ndarray
            Probability vector for the requested states.
        """
        if full_states is None:
            warnings.warn(
                "Generating all 2^num_qubits states. This may be memory intensive.",
                stacklevel=2,
            )
            full_states = np.arange(2**self.circuit.num_qubits, dtype="int64")

        perm = self.reconstruction_flat_qubit_order()
        permuted_indices = permute_bits(
            arr=full_states,
            permutation=perm,
            n_bits=self.circuit.num_qubits,
        )
        reconstructed_probabilities = self.dynamic_definition.probabilities(
            full_states=permuted_indices
        )

        if not quasi:
            logger.info("Converting quasi to real probabilities")
            reconstructed_probabilities = quasi_to_real(
                quasiprobability=reconstructed_probabilities, mode="nearest"
            )
        return reconstructed_probabilities

    def get_ground_truth(self, backend: str) -> np.ndarray:
        """
        Evaluate the original circuit (without cuts) on a backend.

        Parameters
        ----------
        backend
            Backend name (e.g., "statevector_simulator").

        Returns
        -------
        np.ndarray
            Exact probability vector.
        """
        logger.info(f"Evaluating ground truth using {backend}")
        return evaluate_circ(circuit=self.circuit, backend=backend)

    def verify(
        self,
        probabilities: np.ndarray,
        backend: str = "statevector_simulator",
        atol: float = 1e-10,
        raise_error: bool = True,
    ) -> float:
        """
        Compare reconstructed probabilities with exact ground truth.

        Parameters
        ----------
        probabilities
            Reconstructed probability vector to verify.
        backend
            Backend to use for obtaining ground truth.
        atol
            Allowed tolerance on relative MSE.
        raise_error
            If True, raise if error exceeds tolerance; else log.

        Returns
        -------
        float
            Relative mean squared error (normalized as in tests).
        """
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
            logger.error(msg)

        return approximation_error

    def populate_compute_graph(self):
        """Generate the computation graph among subcircuits.

        Nodes contain per-subcircuit metadata (effective qubits, depth, size, etc.).
        Directed edges describe qubit flow between subcircuits and annotate O/rho
        qubit indices used during reconstruction.
        """
        subcircuits = self.subcircuits

        self.compute_graph = ComputeGraph()

        counter = {}
        for j, subcircuit in enumerate(self.subcircuits):
            counter[j] = {
                "effective": subcircuit.num_qubits,
                "rho": 0,
                "O": 0,
                "d": subcircuit.num_qubits,
                "depth": subcircuit.depth(),
                "size": subcircuit.size(),
            }

        for path in self.complete_path_map.values():
            if len(path) > 1:
                for j, (subcircuit_i, _) in enumerate(path[:-1]):
                    next_subcircuit_i = path[j + 1][0]
                    counter[subcircuit_i]["effective"] -= 1
                    counter[subcircuit_i]["O"] += 1
                    counter[next_subcircuit_i]["rho"] += 1

        for subcircuit_idx, subcircuit_attributes in counter.items():
            subcircuit_attributes_copy = deepcopy(subcircuit_attributes)
            subcircuit_attributes_copy["subcircuit"] = subcircuits[subcircuit_idx]
            self.compute_graph.add_node(
                subcircuit_idx=subcircuit_idx, attributes=subcircuit_attributes_copy
            )

        for path in self.complete_path_map.values():
            for j in range(len(path) - 1):
                upstream_subcircuit_idx = path[j][0]
                downstream_subcircuit_idx = path[j + 1][0]
                self.compute_graph.add_edge(
                    u_for_edge=upstream_subcircuit_idx,
                    v_for_edge=downstream_subcircuit_idx,
                    attributes={
                        "O_qubit": path[j][1],
                        "rho_qubit": path[j + 1][1],
                    },
                )

    def populate_subcircuit_entries(self):
        """
        Build subcircuit-entry tables and instances for all subcircuits.

        For each subcircuit, enumerate valid initialization/measurement label
        combinations along incident edges and record the corresponding
        coefficient-weighted terms. Also initialize storage for measured
        probability vectors.
        """
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
                for edge_basis, edge in zip(
                    subcircuit_edge_bases, subcircuit_edges, strict=False
                ):
                    (
                        upstream_subcircuit_idx,
                        downstream_subcircuit_idx,
                        edge_attributes,
                    ) = edge
                    if subcircuit_idx == upstream_subcircuit_idx:
                        O_qubit = edge_attributes["O_qubit"]
                        subcircuit_entry_meas[O_qubit] = edge_basis
                    elif subcircuit_idx == downstream_subcircuit_idx:
                        rho_qubit = edge_attributes["rho_qubit"]
                        subcircuit_entry_init[rho_qubit] = edge_basis
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

    def to_file(self, filepath: str | Path | None, *args, **kwargs) -> None:
        """
        Save the cut circuit to a file.
        Parameters
        ----------
        filepath: str | Path | None
            File path to save the cut circuit. If None, uses the default filepath.
        args: Any
            Any additional positional arguments to pass to the save function.
        kwargs: Any
            Any additional keyword arguments to pass to the save function.
        """
        if filepath is None:
            filepath = self.default_filepath
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Keep imports local to this function
        from cutqc2.io.zarr import cut_circuit_to_zarr

        supported_formats = {".zarr": cut_circuit_to_zarr}
        assert filepath.suffix in supported_formats, "Unsupported format"
        return supported_formats[filepath.suffix](self, filepath, *args, **kwargs)

    def plot(
        self,
        plot_ground_truth: bool = False,
        full_states: np.ndarray | None = None,
        output_file: str | Path | None = None,
    ) -> None:
        """
        Plot reconstructed probabilities (and optional ground truth).

        Parameters
        ----------
        plot_ground_truth
            Whether to overlay ground truth.
        full_states
            Optional list of states to plot; if None, plots all.
        output_file
            If provided, save the plot to this path; else show interactively.
        """
        if full_states is None:
            warnings.warn(
                "Generating all 2^num_qubits states. This may be memory intensive.",
                stacklevel=2,
            )
            full_states = np.arange(2**self.circuit.num_qubits, dtype="int64")

        fig, ax = plt.subplots()
        if plot_ground_truth:
            ground_truth = self.get_ground_truth(backend="statevector_simulator")[
                full_states
            ]
            ax.plot(range(len(ground_truth)), ground_truth, linestyle="--", color="r")

        probabilities = self.get_probabilities(full_states=full_states)
        ax.bar(np.arange(len(full_states)), probabilities)

        if self.dynamic_definition is not None:
            ax.set_title(
                f"Capacity {self.dynamic_definition.capacity}, Recursion {self.dynamic_definition.recursion_level}"
            )

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(str(output_file))
        else:
            plt.show(block=False)
