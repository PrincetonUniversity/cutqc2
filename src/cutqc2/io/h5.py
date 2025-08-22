from pathlib import Path
import h5py
import numpy as np
import cupy as cp
from qiskit.qasm3 import dumps, loads
from cutqc2 import __version__
from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.core.dag import DAGEdge


def cut_circuit_to_h5(
    cut_circuit: CutCircuit, filepath: str | Path, *args, **kwargs
) -> None:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    with h5py.File(filepath, "w") as f:
        str_type = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("version", data=__version__, dtype=str_type)
        f.create_dataset(
            "circuit_qasm", data=dumps(cut_circuit.raw_circuit), dtype=str_type
        )

        if cut_circuit.num_cuts > 0:
            f.create_dataset(
                "cuts",
                data=np.array(
                    [
                        (str(src), str(dest))
                        for src, dest in cut_circuit.cut_dagedgepairs
                    ],
                    dtype=np.dtype([("src", str_type), ("dest", str_type)]),
                ),
            )

            if cut_circuit.complete_path_map:
                complete_path_map = [{}] * cut_circuit.circuit.num_qubits
                for qubit, path in cut_circuit.complete_path_map.items():
                    value = [
                        (e["subcircuit_idx"], e["subcircuit_qubit"]._index)
                        for e in path
                    ]
                    complete_path_map[qubit._index] = value

                dtype = np.dtype([("subcircuit", "i4"), ("qubit", "i4")])
                for j, path in enumerate(complete_path_map):
                    f.create_dataset(
                        f"complete_path_map/{j}",
                        data=np.array(path, dtype=dtype),
                    )

            # Get expensive properties once
            reconstruction_qubit_order = cut_circuit.reconstruction_qubit_order

            for subcircuit_i in range(len(cut_circuit)):
                subcircuit_group = f.create_group(f"subcircuits/{subcircuit_i}")

                subcircuit_group.create_dataset(
                    "qasm", data=dumps(cut_circuit[subcircuit_i]), dtype=str_type
                )
                subcircuit_group.create_dataset(
                    "nodes",
                    data=np.array(
                        [
                            str(edge)
                            for edge in cut_circuit.subcircuit_dagedges[subcircuit_i]
                        ],
                        dtype=np.dtype(str_type),
                    ),
                )

                value = reconstruction_qubit_order[subcircuit_i]
                subcircuit_group.create_dataset(
                    "qubit_order", data=np.array(value, dtype="int")
                )

                if cut_circuit.subcircuit_entry_probs:
                    group = f.create_group(f"subcircuits/{subcircuit_i}/probabilities")
                    for k, v in cut_circuit.subcircuit_entry_probs[
                        subcircuit_i
                    ].items():
                        key = "_".join(["-".join(k[0]), "-".join(k[1])])
                        group.create_dataset(key, data=np.array(v, dtype="float64"))

                    subcircuit_group.create_dataset(
                        "packed_probabilities",
                        data=cp.asnumpy(
                            cut_circuit.get_packed_probabilities(subcircuit_i)
                        ),
                    )


def h5_to_cut_circuit(filepath: str | Path, *args, **kwargs) -> CutCircuit:
    with h5py.File(filepath, "r") as f:
        qasm_str = f["circuit_qasm"][()].decode("utf-8")
        cut_circuit = CutCircuit(loads(qasm_str))

        if "cuts" in f and "subcircuits" in f:
            cuts = f["cuts"][()]
            cut_edge_pairs = [
                (
                    DAGEdge.from_string(src.decode("utf-8")),
                    DAGEdge.from_string(dest.decode("utf-8")),
                )
                for (src, dest) in cuts
            ]

            subcircuit_dagedges = [None] * len(f["subcircuits"])

            for subcircuit_i in f["subcircuits"]:
                subcircuit_group = f["subcircuits"][subcircuit_i]
                subcircuit_i = int(subcircuit_i)

                subcircuit_n_dagedges = [
                    DAGEdge.from_string(edge.decode("utf-8"))
                    for edge in subcircuit_group["nodes"][()]
                ]
                subcircuit_dagedges[subcircuit_i] = subcircuit_n_dagedges

            cut_circuit.add_cuts_and_generate_subcircuits(
                cut_edge_pairs, subcircuit_dagedges
            )

        reconstruction_qubit_order = {}
        entry_probs = {}
        if "subcircuits" in f:
            for subcircuit_i in f["subcircuits"]:
                subcircuit_group = f["subcircuits"][subcircuit_i]
                subcircuit_i = int(subcircuit_i)
                if "qubit_order" in subcircuit_group:
                    reconstruction_qubit_order[subcircuit_i] = (
                        subcircuit_group["qubit_order"][()].astype(int).tolist()
                    )

                if "probabilities" in subcircuit_group:
                    prob_group = subcircuit_group["probabilities"]
                    prob_dict = {}
                    for key, ds in prob_group.items():
                        str_a, str_b = key.split("_")
                        tuple_key = (tuple(str_a.split("-")), tuple(str_b.split("-")))
                        prob_dict[tuple_key] = ds[()].astype(float)
                    entry_probs[subcircuit_i] = prob_dict

                if "packed_probabilities" in subcircuit_group:
                    packed_probs = subcircuit_group["packed_probabilities"][()]
                    cut_circuit.subcircuit_packed_probs[subcircuit_i] = packed_probs

        if reconstruction_qubit_order:
            cut_circuit.reconstruction_qubit_order = reconstruction_qubit_order
        if entry_probs:
            cut_circuit.subcircuit_entry_probs = entry_probs

    return cut_circuit
