import numpy as np


def permute_bits(n: int, permutation: list[int]) -> int:
    n_bits = len(permutation)
    binary_n = f"{n:0{n_bits}b}"
    # Get bit i from position permutation[i]
    binary_n_permuted = "".join(binary_n[permutation[i]] for i in range(n_bits))
    return int(binary_n_permuted, 2)


def distribute(n_qubits: int, probabilities: dict[str, np.array], capacity: int) -> str:
    largest_bin_value = 0
    largest_bin_index = 0
    largest_bin_qubit_spec = ""
    for _qubit_spec, prob in probabilities.items():
        largest_bin_value_candidate = np.max(prob)
        if largest_bin_value_candidate > largest_bin_value:
            largest_bin_value = largest_bin_value_candidate
            largest_bin_index = np.argmax(prob)
            largest_bin_qubit_spec = _qubit_spec

    largest_bin_str = f"{largest_bin_index:0{n_qubits}b}"
    for j in range(n_qubits):
        largest_bin_qubit_spec = largest_bin_qubit_spec.replace(
            "A", largest_bin_str[j], 1
        )
    # qubit_spec = qubit_spec.replace("M", "A", capacity)

    return largest_bin_qubit_spec


def merge_prob_vector(unmerged_prob_vector: np.ndarray, qubit_spec: str) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Compressed probability vector (2^num_active,) with marginalization and conditioning applied.
    """
    num_qubits = len(qubit_spec)
    assert len(unmerged_prob_vector) == 2**num_qubits, (
        "Mismatch in qubit count and vector length."
    )

    active_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "A"]
    num_active = len(active_qubit_indices)

    if num_active == num_qubits:
        return np.copy(unmerged_prob_vector)

    merged_prob_vector = np.zeros(2**num_active, dtype="float32")

    for state in range(len(unmerged_prob_vector)):
        match = True
        for i, spec in enumerate(qubit_spec):
            bit_index = num_qubits - 1 - i  # MSB-first mapping
            bit_val = (state >> bit_index) & 1
            if spec == "0" and bit_val != 0:
                match = False
                break
            elif spec == "1" and bit_val != 1:
                match = False
                break
        if not match:
            continue

        # Construct index for active qubits
        active_state = 0
        for out_pos, i in enumerate(active_qubit_indices):
            bit_index = num_qubits - 1 - i
            bit_val = (state >> bit_index) & 1
            if bit_val:
                active_state |= 1 << (num_active - 1 - out_pos)  # MSB-first output

        merged_prob_vector[active_state] += unmerged_prob_vector[state]

    return merged_prob_vector


def unmerge_prob_vector(merged_prob_vector: np.ndarray, qubit_spec: str) -> np.ndarray:
    """
    Expand a merged quantum probability vector back to a full vector
    by evenly distributing over merged qubits and conditioning on fixed ones.

    Parameters
    ----------
    merged_prob_vector : np.ndarray
        Compressed probability vector (2^num_active,)
    qubit_spec : str
        String of length num_qubits with characters:
        - "A": active (preserved)
        - "M": merged (marginalized out)
        - "0"/"1": fixed bits

    Returns
    -------
    np.ndarray
        Expanded full probability vector of shape (2^num_qubits,)
    """
    num_qubits = len(qubit_spec)
    active_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "A"]
    merged_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "M"]
    fixed_qubit_conditions = {
        i: int(q) for i, q in enumerate(qubit_spec) if q in ("0", "1")
    }

    num_active = len(active_qubit_indices)
    num_merged = len(merged_qubit_indices)

    expanded = np.zeros(2**num_qubits, dtype=np.float32)

    for full_state in range(2**num_qubits):
        match = True
        for i, val in fixed_qubit_conditions.items():
            bit_index = num_qubits - 1 - i  # MSB to LSB
            bit_val = (full_state >> bit_index) & 1
            if bit_val != val:
                match = False
                break
        if not match:
            continue

        # Build index into merged vector
        active_index = 0
        for out_pos, i in enumerate(active_qubit_indices):
            bit_index = num_qubits - 1 - i
            bit_val = (full_state >> bit_index) & 1
            if bit_val:
                active_index |= 1 << (num_active - 1 - out_pos)

        num_merge_combinations = 2**num_merged

        # Uniformly distribute merged prob
        expanded[full_state] = merged_prob_vector[active_index] / num_merge_combinations

    return expanded
