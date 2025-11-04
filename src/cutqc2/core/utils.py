import copy
import itertools
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from qiskit import QuantumCircuit
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


def permute_bits(arr: np.ndarray, permutation: np.ndarray, n_bits: int) -> np.ndarray:
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
    subcircuit: QuantumCircuit,
    subcircuit_instance_init_meas: list[tuple[tuple[str], tuple[str]]],
    backend: str = "statevector_simulator",
    max_workers: int = 1,
) -> dict[tuple[tuple[str], tuple[str]], np.ndarray | float]:
    """
    Evaluate a set of subcircuit instances under different initializations and
    measurement bases, returning their measured probability distributions.

    The function iterates over provided `(init, meas)` specifications, creates a
    runnable subcircuit instance via `modify_subcircuit_instance`, simulates it
    using `evaluate_circ`, and then projects the resulting state/probabilities
    into the requested measurement bases. If a measurement specification
    contains "Z", that instance is skipped. Any identity basis entries "I" are
    expanded into both "I" and "Z" via `mutate_measurement_basis` and each
    mutation is measured separately.

    Parameters
    ----------
    subcircuit : QuantumCircuit
        Base subcircuit to instantiate and simulate.
    subcircuit_instance_init_meas : list[tuple[tuple[str], tuple[str]]]
        A list of `(init, meas)` pairs, where:
        - `init` is a tuple of state labels per qubit (e.g., "zero", "one",
          "plus", "minus", "plusI", "minusI").
        - `meas` is a tuple of measurement basis labels per qubit (e.g.,
          "comp", "X", "Y", "I"). If any entry is "Z", the instance is
          skipped.
    backend : str, optional
        Backend identifier passed to `evaluate_circ` (default is
        "statevector_simulator").
    max_workers : int, optional
        Maximum number of parallel threads to use (default is 1, i.e., no
        parallelism).

    Returns
    -------
    dict[tuple[tuple[str], tuple[str]], np.ndarray | float]
        A mapping from `(init, meas)` to the measured probability vector (or a
        scalar if the circuit evaluates to a single probability). The `meas`
        key in the mapping reflects any mutated basis produced by
        `mutate_measurement_basis`.
    """
    total = len(subcircuit_instance_init_meas)
    subcircuit_measured_probs = {}

    def process_instance(i, instance_init_meas):
        logger.info(f"Running subcircuit instance {i + 1}/{total}")
        initialization, measurement = instance_init_meas

        results = {}
        # `mutate_measurement_basis` expands all occurrences of I bases into I
        # and Z bases (".. measuring a qubit in either the I or Z basis
        # corresponds physically to the same quantum circuit ..")
        # So we can ignore any incoming bases that have "Z" in the measurement
        # bases. Returning a {} results in no effective updates in the caller.
        if "Z" in measurement:
            return {}

        subcircuit_instance = modify_subcircuit_instance(
            subcircuit=subcircuit,
            init=initialization,
            meas=measurement,
        )

        subcircuit_inst_prob = evaluate_circ(
            circuit=subcircuit_instance, backend=backend
        )
        subcircuit_inst_prob = xp.asarray(subcircuit_inst_prob)

        mutated_measurement = mutate_measurement_basis(bases=measurement)
        for _measurement in mutated_measurement:
            # bases in `_measurement` are LSB to MSB. Reverse them since
            # `measure_prob` is expecting MSB to LSB.
            measured_prob = measure_prob(
                unmeasured_prob=subcircuit_inst_prob, meas=_measurement[::-1]
            )
            results[(initialization, _measurement)] = measured_prob
        return results

    max_workers = min(max_workers, cpu_count(), total)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_instance, i, instance_init_meas)
            for i, instance_init_meas in enumerate(subcircuit_instance_init_meas)
        ]
        for future in as_completed(futures):
            subcircuit_measured_probs.update(future.result())

    return subcircuit_measured_probs


def mutate_measurement_basis(bases: tuple[str]) -> list[tuple[str]]:
    """
    Expand a measurement-basis specification by replacing identity entries
    with both identity and Z bases.

    If all entries are non-identity (no "I" present), the function returns a
    singleton list containing the original `meas`. Otherwise, for each qubit
    position with basis "I", it generates two alternatives: "I" and "Z". The
    Cartesian product across positions yields all mutated basis tuples.

    Parameters
    ----------
    bases : Sequence[str]
        Per-qubit measurement bases (e.g., "comp", "X", "Y", "I").

    Returns
    -------
    list[tuple[str, ...]]
        All mutated measurement-basis tuples. If no mutation is needed, this is
        `[meas]`.
    """
    if all(x != "I" for x in bases):
        return [bases]
    mutated_bases = []
    for x in bases:
        if x != "I":
            mutated_bases.append([x])
        else:
            mutated_bases.append(["I", "Z"])
    return list(itertools.product(*mutated_bases))


def modify_subcircuit_instance(  # noqa: PLR0912
    subcircuit: QuantumCircuit, init: tuple[str], meas: tuple[str]
) -> QuantumCircuit:
    """
    Create a runnable instance of `subcircuit` with specified initialization and
    measurement-basis rotations applied.

    For each qubit i, the corresponding entry in `init` determines the state
    preparation inserted at the front of the circuit DAG.

    For each qubit i, the corresponding entry in `meas` determines a basis
    rotation appended to the back of the circuit so that subsequent computational
    basis measurement is equivalent to measuring in that basis.

    Parameters
    ----------
    subcircuit : qiskit.QuantumCircuit
        Base subcircuit to modify.
    init : Sequence[str]
        Per-qubit initialization labels. Length must equal the number of qubits
        in `subcircuit`.
    meas : Sequence[str]
        Per-qubit measurement basis labels. Length must equal the number of
        qubits in `subcircuit`.

    Returns
    -------
    qiskit.QuantumCircuit
        A new circuit with the requested preparations and basis rotations
        applied.
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        match x:
            case "zero":
                continue
            case "one":
                subcircuit_instance_dag.apply_operation_front(
                    op=XGate(), qargs=[q], cargs=[]
                )
            case "plus":
                subcircuit_instance_dag.apply_operation_front(
                    op=HGate(), qargs=[q], cargs=[]
                )
            case "minus":
                subcircuit_instance_dag.apply_operation_front(
                    op=HGate(), qargs=[q], cargs=[]
                )
                subcircuit_instance_dag.apply_operation_front(
                    op=XGate(), qargs=[q], cargs=[]
                )
            case "plusI":
                subcircuit_instance_dag.apply_operation_front(
                    op=SGate(), qargs=[q], cargs=[]
                )
                subcircuit_instance_dag.apply_operation_front(
                    op=HGate(), qargs=[q], cargs=[]
                )
            case "minusI":
                subcircuit_instance_dag.apply_operation_front(
                    op=SGate(), qargs=[q], cargs=[]
                )
                subcircuit_instance_dag.apply_operation_front(
                    op=HGate(), qargs=[q], cargs=[]
                )
                subcircuit_instance_dag.apply_operation_front(
                    op=XGate(), qargs=[q], cargs=[]
                )
            case _:
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


def bases_to_bitmask(bases: tuple[str]) -> np.array:
    map = {"comp": 1, "I": 0, "X": 0, "Y": 0, "Z": 0}
    mask = [map[b] for b in bases]
    return xp.array(mask)


def compress_bits(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a bit mask that drops bits.
    mask is an array/list of 0/1, where bit 0 = LSB, bit[-1] = MSB.
    """
    kept_positions = xp.nonzero(mask)[0]
    # Create mapping from old bit positions to new packed bit positions
    shifts = xp.arange(len(kept_positions))
    # Compute new number by summing selected bits shifted to compact positions
    result = xp.zeros_like(arr)
    for src, dst in zip(kept_positions, shifts, strict=False):
        result |= ((arr >> src) & 1) << dst
    return result


def measure_prob(unmeasured_prob: np.ndarray, meas: tuple[str]) -> np.ndarray:
    """
    Project a probability vector from mixed measurement bases onto the
    computational subspace defined by entries equal to "comp".

    If all bases are computational (or `unmeasured_prob` is a scalar), the input
    is returned unchanged. Otherwise, for each full state index, the state is
    mapped to an effective computational-basis index using `measure_state` and
    accumulated with the appropriate sign.

    Parameters
    ----------
    unmeasured_prob : np.ndarray | float
        Probability vector over all 2^n basis states (MSB-to-LSB convention) or
        a scalar probability.
    meas : Sequence[str]
        Per-qubit measurement basis labels, MSB to LSB (e.g., "comp", "X", "Y", "I").

    Returns
    -------
    np.ndarray | float
        Measured probability vector over 2^k states, where k is the number of
        entries equal to "comp" in `meas`. If the input is a scalar or `meas`
        is entirely computational, the input is returned.
    """
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob

    measured_prob = xp.zeros(int(2 ** meas.count("comp")))
    meas_bitmask = bases_to_bitmask(meas)[::-1]
    _n = int(xp.round(xp.log2(unmeasured_prob.size)))
    effective_states = compress_bits(xp.arange(1 << _n), meas_bitmask)

    sigmas = measure_sign(xp.arange(len(unmeasured_prob)), meas)
    values = sigmas * unmeasured_prob

    xp.add.at(measured_prob, xp.array(effective_states), xp.array(values))
    return measured_prob


def measure_sign(
    full_states: np.ndarray | list[int], meas: tuple[str, ...]
) -> np.ndarray:
    """
    Vectorized version of sign measurement.
    full_states: array of ints (shape (...,))
    meas: tuple/list of strings, e.g. ("X", "I", "Y")
    Returns array of ±1 with same shape as full_states.
    """
    n = len(meas)
    full_states = xp.asarray(full_states, dtype=xp.int64)

    # Bits to consider: from MSB→LSB, but we'll index LSB-first for efficiency
    # Compute the bits matrix: shape (len(full_states), n)
    bits = ((full_states[:, None] >> xp.arange(n - 1, -1, -1)) & 1).astype(bool)

    # Mask of which bases trigger a sign flip (True = flip)
    flip_mask = xp.array([b not in ("I", "comp") for b in meas])

    # Count flips: bits that are 1 *and* basis requires flipping
    flips = xp.sum(bits & flip_mask, axis=1)

    # (-1) ** (# of flips)
    return xp.where(flips % 2 == 0, 1, -1)


def attribute_shots(
    subcircuit_measured_probs: dict[tuple[tuple[str], tuple[str]], np.ndarray | float],
    subcircuit_entries,
) -> dict[tuple[tuple[str], tuple[str]], np.ndarray | float]:
    """
    Aggregate measured probabilities for symbolic subcircuit entries by linear
    combination of their contributing instances.

    Each entry key maps to a list of terms `(coefficient, instance_key)`. The
    function forms `entry_prob = sum_i coefficient_i *
    subcircuit_measured_probs[instance_key_i]` for every entry.

    Parameters
    ----------
    subcircuit_measured_probs : dict
        Mapping from `(init, meas)` instance keys to measured probability vectors
        (or scalars).
    subcircuit_entries : dict
        Mapping from entry keys to a list of `(coefficient, instance_key)` terms.

    Returns
    -------
    dict
        Mapping from entry keys to aggregated probability vectors (or scalars).
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
