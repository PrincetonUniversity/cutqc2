import copy
import itertools
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from qiskit.circuit.library.standard_gates import HGate, SdgGate, SGate, XGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

from cutqc2.cutqc.helper_functions.non_ibmq_functions import evaluate_circ
from cutqc2.numeric import xp

logger = logging.getLogger(__name__)


def chunked(gen, chunk_size):
    """Yield lists of length chunk_size from generator gen."""
    gen = iter(gen)
    while True:
        chunk = list(itertools.islice(gen, chunk_size))
        if not chunk:
            break
        yield chunk


def permute_bits(n: int, permutation: list[int]) -> int:
    n_bits = len(permutation)
    binary_n = f"{n:0{n_bits}b}"
    # Get bit i from position permutation[i]
    binary_n_permuted = "".join(binary_n[permutation[i]] for i in range(n_bits))
    return int(binary_n_permuted, 2)


def permute_bits_vectorized(
    arr: np.ndarray, permutation: np.ndarray, n_bits: int
) -> np.ndarray:
    n_bits = len(permutation)
    result = np.zeros_like(arr)

    for dest_bit_msb, src_bit_msb in enumerate(permutation):
        # Convert MSB index to LSB index for bit operations
        src_bit_lsb = n_bits - 1 - src_bit_msb
        dest_bit_lsb = n_bits - 1 - dest_bit_msb

        # Extract source bit and place in destination position
        result |= ((arr >> src_bit_lsb) & 1) << dest_bit_lsb

    return result


def merge_prob_vector(
    unmerged_prob_vector: np.ndarray,
    qubit_spec: str,
    qubit_spec_lsb_first: bool = False,
) -> np.ndarray:
    """
    Compress quantum probability vector by merging specified qubits
    and conditioning on fixed qubit values.

    Parameters
    ----------
    unmerged_prob_vector : np.ndarray
        Original probability vector (2^num_qubits,)
    qubit_spec : str
        String of length `num_qubits`, MSB to LSB, with each character
        indicating:
        - "A": qubit is preserved in output
        - "M": qubit is summed over
        - "0"/"1": qubit is fixed to that value
    qubit_spec_lsb_first : bool
        If True, qubit_spec is given LSB to MSB instead of MSB to LSB.

    Returns
    -------
    np.ndarray
        Compressed probability vector (2^num_active,) with marginalization and conditioning applied.
    """
    if not qubit_spec_lsb_first:
        qubit_spec = qubit_spec[::-1]  # LSB to MSB

    num_qubits = len(qubit_spec)
    assert len(unmerged_prob_vector) == 2**num_qubits, (
        "Mismatch in qubit count and vector length."
    )

    active_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "A"]
    num_active = len(active_qubit_indices)

    if num_active == num_qubits:
        return xp.copy(xp.asarray(unmerged_prob_vector))

    merged_prob_vector = xp.zeros(2**num_active, dtype="float32")

    for state in range(len(unmerged_prob_vector)):
        match = True
        for i, spec in enumerate(qubit_spec):
            bit_val = (state >> i) & 1
            if spec == "0" and bit_val != 0:
                match = False
                break
            if spec == "1" and bit_val != 1:
                match = False
                break
        if not match:
            continue

        # Construct index for active qubits
        active_state = 0
        for out_pos, i in enumerate(active_qubit_indices):
            bit_val = (state >> i) & 1
            if bit_val:
                active_state |= 1 << out_pos  # LSB-first output

        merged_prob_vector[active_state] += unmerged_prob_vector[state]

    return merged_prob_vector


def unmerge_prob_vector(
    merged_prob_vector: np.ndarray,
    qubit_spec: str,
    full_states: np.ndarray | None = None,
    qubit_spec_lsb_first: bool = False,
) -> None:
    """
    Expand a merged quantum probability vector back to a full vector
    by evenly distributing over merged qubits and conditioning on fixed ones.

    Parameters
    ----------
    merged_prob_vector : np.ndarray
        Compressed probability vector (2^num_active,)
    qubit_spec : str
        String of length num_qubits, MSB to LSB, with each character
        indicating:
        - "A": active (preserved)
        - "M": merged (marginalized out)
        - "0"/"1": fixed bits
    full_states : np.ndarray or None
        Array of full states to fill in.
        If None, all 2**|num_qubits| states are filled-in.
    qubit_spec_lsb_first : bool
        If True, qubit_spec is given LSB to MSB instead of MSB to LSB.
    """
    if not qubit_spec_lsb_first:
        qubit_spec = qubit_spec[::-1]

    num_qubits = len(qubit_spec)
    if full_states is None:
        if num_qubits > 20:  # noqa: PLR2004
            warnings.warn(
                "Generating all 2^num_qubits states. This may be memory intensive.",
                stacklevel=2,
            )
        full_states = np.arange(2**num_qubits, dtype=np.int64)

    active_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "A"]
    merged_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "M"]
    fixed_qubit_conditions = {
        i: int(q) for i, q in enumerate(qubit_spec) if q in ("0", "1")
    }

    num_merged = len(merged_qubit_indices)

    unmerged = np.zeros_like(full_states, dtype="float32")
    for j, full_state in enumerate(full_states):
        match = True
        for i, val in fixed_qubit_conditions.items():
            bit_val = (full_state >> i) & 1
            if bit_val != val:
                match = False
                break
        if not match:
            continue

        # Build index into merged vector
        active_index = 0
        for out_pos, i in enumerate(active_qubit_indices):
            bit_val = (full_state >> i) & 1
            if bit_val:
                active_index |= 1 << out_pos  # LSB-first output

        num_merge_combinations = 2**num_merged

        # Uniformly distribute merged prob
        unmerged[j] += merged_prob_vector[active_index] / num_merge_combinations

    return unmerged


def run_subcircuit_instances(
    subcircuit, subcircuit_instance_init_meas, backend: str = "statevector_simulator"
):
    total = len(subcircuit_instance_init_meas)
    subcircuit_measured_probs = {}

    def process_instance(i, instance_init_meas):
        logger.info(f"Running subcircuit instance {i + 1}/{total}")
        results = {}
        if "Z" in instance_init_meas[1]:
            return results

        subcircuit_instance = modify_subcircuit_instance(
            subcircuit=subcircuit,
            init=instance_init_meas[0],
            meas=instance_init_meas[1],
        )
        subcircuit_inst_prob = evaluate_circ(
            circuit=subcircuit_instance, backend=backend
        )

        mutated_meas = mutate_measurement_basis(meas=instance_init_meas[1])
        total_mutations = len(mutated_meas)
        for j, meas in enumerate(mutated_meas):
            logger.info(f"{j + 1}/{total_mutations}")
            measured_prob = measure_prob(
                unmeasured_prob=subcircuit_inst_prob, meas=meas
            )
            results[(instance_init_meas[0], meas)] = measured_prob
        return results

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_instance, i, instance_init_meas)
            for i, instance_init_meas in enumerate(subcircuit_instance_init_meas)
        ]
        for future in as_completed(futures):
            subcircuit_measured_probs.update(future.result())

    return subcircuit_measured_probs


def mutate_measurement_basis(meas):
    """
    I and Z measurement basis correspond to the same logical circuit
    """
    if all(x != "I" for x in meas):
        return [meas]
    mutated_meas = []
    for x in meas:
        if x != "I":
            mutated_meas.append([x])
        else:
            mutated_meas.append(["I", "Z"])
    return list(itertools.product(*mutated_meas))


def modify_subcircuit_instance(subcircuit, init, meas):  # noqa: PLR0912
    """
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        if x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x in ("I", "comp"):
            continue
        if x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    return dag_to_circuit(subcircuit_instance_dag)


def measure_prob(unmeasured_prob, meas):
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob
    measured_prob = np.zeros(int(2 ** meas.count("comp")))

    for full_state, p in enumerate(unmeasured_prob):
        sigma, effective_state = measure_state(full_state=full_state, meas=meas)
        measured_prob[effective_state] += sigma * p
    return measured_prob


def measure_state(full_state, meas):
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1], strict=False):
        if meas_bit == "1" and meas_basis not in ("I", "comp"):
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state


def attribute_shots(subcircuit_measured_probs, subcircuit_entries):
    """
    Attribute the subcircuit_instance shots into respective subcircuit entries
    subcircuit_entry_probs[entry_init, entry_meas] = entry_prob
    """
    subcircuit_entry_probs = {}
    for subcircuit_entry_init_meas in subcircuit_entries:
        subcircuit_entry_term = subcircuit_entries[subcircuit_entry_init_meas]
        subcircuit_entry_prob = None
        for term in subcircuit_entry_term:
            coefficient, subcircuit_instance_init_meas = term
            subcircuit_instance_prob = subcircuit_measured_probs[
                subcircuit_instance_init_meas
            ]
            if subcircuit_entry_prob is None:
                subcircuit_entry_prob = coefficient * subcircuit_instance_prob
            else:
                subcircuit_entry_prob += coefficient * subcircuit_instance_prob
        subcircuit_entry_probs[subcircuit_entry_init_meas] = subcircuit_entry_prob
    return subcircuit_entry_probs
