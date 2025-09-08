import heapq
from pathlib import Path

import cupy as cp
import numpy as np
import zarr
from qiskit.qasm3 import dumps, loads

from cutqc2 import __version__
from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.core.dag import DAGEdge
from cutqc2.core.dynamic_definition import Bin, DynamicDefinition


def cut_circuit_to_zarr(cut_circuit, filepath: str | Path) -> None:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    store = zarr.storage.LocalStore(str(filepath))
    root = zarr.group(store=store, overwrite=True)

    meta_group = root.create_group("metadata", overwrite=True)

    meta_group.attrs.put(
        {
            "version": __version__,
            "circuit_qasm": dumps(cut_circuit.raw_circuit),
        }
    )

    if cut_circuit.num_cuts > 0:
        cuts_arr = np.array(
            [(str(src), str(dest)) for src, dest in cut_circuit.cut_dagedgepairs],
            dtype=[("src", "U20"), ("dest", "U20")],
        )
        root.create_array("cuts", data=cuts_arr)

        if cut_circuit.complete_path_map:
            complete_path_map = [{}] * cut_circuit.circuit.num_qubits
            for qubit, path in cut_circuit.complete_path_map.items():
                value = [
                    (e["subcircuit_idx"], e["subcircuit_qubit"]._index) for e in path
                ]
                complete_path_map[qubit._index] = value

            dtype = np.dtype([("subcircuit", "i4"), ("qubit", "i4")])
            for j, path in enumerate(complete_path_map):
                root.create_array(
                    f"complete_path_map/{j}",
                    data=np.array(path, dtype=dtype),
                )

        # Get expensive properties once
        reconstruction_qubit_order = cut_circuit.reconstruction_qubit_order

        for subcircuit_i in range(len(cut_circuit)):
            subcircuit_group = root.create_group(f"subcircuits/{subcircuit_i}")

            subcircuit_group.attrs.put(
                {
                    "qasm": dumps(cut_circuit[subcircuit_i]),
                }
            )

            subcircuit_group.create_array(
                "nodes",
                data=np.array(
                    [
                        str(edge)
                        for edge in cut_circuit.subcircuit_dagedges[subcircuit_i]
                    ],
                    dtype="U20",
                ),
            )

            value = reconstruction_qubit_order[subcircuit_i]
            subcircuit_group.create_array(
                "qubit_order", data=np.array(value, dtype="int")
            )

            if cut_circuit.subcircuit_entry_probs:
                prob_group = subcircuit_group.create_group("probabilities")
                for k, v in cut_circuit.subcircuit_entry_probs[subcircuit_i].items():
                    key = "_".join(["-".join(k[0]), "-".join(k[1])])
                    prob_group.create_array(key, data=np.array(v, dtype="float64"))

                subcircuit_group.create_array(
                    "packed_probabilities",
                    data=cp.asnumpy(cut_circuit.get_packed_probabilities(subcircuit_i)),
                )

        # Dynamic Definition
        if (dd := cut_circuit.dynamic_definition) is not None:
            dd_group = root.create_group("dynamic_definition")
            dd_group.attrs.put(
                {
                    "num_qubits": dd.num_qubits,
                    "capacity": dd.capacity,
                    "epsilon": dd.epsilon,
                }
            )
            for j, bin in enumerate(heapq.nsmallest(len(dd.bins), dd.bins)):
                bin_group = dd_group.create_group(str(j))
                bin_group.attrs.put(
                    {
                        "qubit_spec": bin.qubit_spec,
                        "probability_mass": float(bin.probability_mass),
                    }
                )
                bin_group.create_array(
                    "probabilities", data=cp.asnumpy(bin.probabilities)
                )


def zarr_to_cut_circuit(filepath: str | Path) -> CutCircuit:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    root = zarr.open(str(filepath))

    qasm_str = root["metadata"].attrs["circuit_qasm"]
    cut_circuit = CutCircuit(loads(qasm_str))

    # Load cuts and subcircuits
    if "cuts" in root and "subcircuits" in root:
        cuts = root["cuts"][()]
        cut_edge_pairs = [
            (
                DAGEdge.from_string(src),
                DAGEdge.from_string(dest),
            )
            for (src, dest) in cuts
        ]

        subcircuit_dagedges = [None] * len(root["subcircuits"])
        for subcircuit_i in root["subcircuits"]:
            subcircuit_group = root[f"subcircuits/{subcircuit_i}"]
            subcircuit_idx = int(subcircuit_i)

            subcircuit_n_dagedges = [
                DAGEdge.from_string(edge) for edge in subcircuit_group["nodes"][()]
            ]
            subcircuit_dagedges[subcircuit_idx] = subcircuit_n_dagedges

        cut_circuit.add_cuts_and_generate_subcircuits(
            cut_edge_pairs, subcircuit_dagedges
        )

    # Reconstruction qubit order & subcircuit probabilities
    reconstruction_qubit_order = {}
    entry_probs = {}

    if "subcircuits" in root:
        for subcircuit_i in root["subcircuits"]:
            subcircuit_group = root[f"subcircuits/{subcircuit_i}"]
            subcircuit_idx = int(subcircuit_i)

            if "qubit_order" in subcircuit_group:
                reconstruction_qubit_order[subcircuit_idx] = (
                    subcircuit_group["qubit_order"][()].astype(int).tolist()
                )

            if "probabilities" in subcircuit_group:
                prob_group = subcircuit_group["probabilities"]
                prob_dict = {}
                for key in prob_group:
                    str_a, str_b = key.split("_")
                    tuple_key = (tuple(str_a.split("-")), tuple(str_b.split("-")))
                    prob_dict[tuple_key] = prob_group[key][()].astype(float)
                entry_probs[subcircuit_idx] = prob_dict

            if "packed_probabilities" in subcircuit_group:
                packed_probs = subcircuit_group["packed_probabilities"][()]
                cut_circuit.subcircuit_packed_probs[subcircuit_idx] = packed_probs

    if reconstruction_qubit_order:
        cut_circuit.reconstruction_qubit_order = reconstruction_qubit_order
    if entry_probs:
        cut_circuit.subcircuit_entry_probs = entry_probs

    # Dynamic Definition
    if "dynamic_definition" in root:
        dd_group = root["dynamic_definition"]
        dd = DynamicDefinition(
            num_qubits=dd_group.attrs["num_qubits"],
            capacity=dd_group.attrs["capacity"],
            prob_fn=cut_circuit.compute_probabilities,
            epsilon=dd_group.attrs.get("epsilon", 1e-4),
        )

        for bin_str in dd_group:
            bin_group = dd_group[bin_str]
            qubit_spec = bin_group.attrs["qubit_spec"]
            probabilities = bin_group["probabilities"][()]
            bin = Bin(qubit_spec=qubit_spec, probabilities=probabilities)
            dd.push(bin)

        cut_circuit.dynamic_definition = dd

    return cut_circuit
