## The `CutCircuit` class - Walkthrough

This walkthrough traces the path through the key methods of `CutCircuit`, and explains the internal attributes and data structures as they are populated and consumed. It follows the example below.

```python
from cutqc2.library.sample import sample_circuit
from cutqc2.core.cut_circuit import CutCircuit


if __name__ == "__main__":
    cut_circuit = CutCircuit(sample_circuit)

    cut_circuit.cut(
        max_subcircuit_width=3,  # Max qubits per subcircuit
        max_cuts=1,  # Max total cuts in the circuit
        num_subcircuits=[2],  # Desired number of subcircuits to try
    )
    print(cut_circuit)

    for subcircuit in cut_circuit:
        print(subcircuit)
        print()

    cut_circuit.run_subcircuits()
    
    cut_circuit.to_file("tutorial.zarr")
    cut_circuit = CutCircuit.from_file("tutorial.zarr")

    cut_circuit.postprocess(capacity=3, max_recursion=9)

    probabilities = cut_circuit.get_probabilities()

    error = cut_circuit.verify(probabilities, raise_error=False)
    print(f"Verification error: {error}")

    cut_circuit.plot(plot_ground_truth=True)

```

### 1) Initialization

Constructor: `CutCircuit(circuit | circuit_qasm3)`

- Inputs:
  - `circuit`: a `qiskit.QuantumCircuit`; if `None`, a qasm3 `str` representation of the circuit using the the `circuit_qasm3` argument must be provided.


- `cut_circuit = CutCircuit(circuit)` validates the circuit (single component, no classical bits, no barriers, ≤ 2-qubit ops). and initializes the following attributes:

  - `self.circuit`: a copy of the input circuit.
  - `self.circuit_with_cut_gates`: the input circuit with placeholder cut gates, useful for visualization. This is initially a blank `QuantumCircuit` with the same quantum registers, and is later populated when `cut()` is called.
  - `self.num_cuts`: 0 (number of cuts inserted - initially zero)
  - `self.subcircuits`: []  (list of `QuantumCircuit` subcircuits - initially empty)
  - `self.subcircuit_dagedges`: []  (list of lists of `DAGEdge` that define subcircuits). `DAGEdge` objects are abstractions that represent a two-qubit gate and its associated wire and gate indices.
  - `self.complete_path_map`: {} (wire_index → [(subcircuit_i, subcircuit_wire_index), ...]). This maps each original wire to the sequence of subcircuits and local qubits it traverses.
  - `self.dynamic_definition`: None. This will later hold a `DynamicDefinition` object that manages the recursive reconstruction process.


Some other internal attributes that are initialized are:
  - `self.inter_wire_dag`: a DAG containing only two-qubit gates (for cut search).
  - `self.inter_wire_dag_metadata`: metadata for `self.inter_wire_dag`:
    - `n_vertices`: number of two-qubit vertices
    - `edges`: edges among those vertices
    - `id_to_dag_edge`: mapping from integer vertex ids to `DAGEdge` objects. 


### 2) Cutting the circuit

Method: `cut(max_subcircuit_width, max_cuts, num_subcircuits, subcircuits=None)`

- This step calls `find_cuts()` to search a feasible partition:
  - Internally, it uses `self.inter_wire_dag_metadata` and the `MIPCutSearcher` class, which is a MIP solver to find a partition of the two-qubit gates into subcircuits that respect certain constraints.
  - Returns `list[list[DAGEdge]]` where each inner list is the list of `DAGEdge` objects (two-qubit gates) assigned to that subcircuit.


- Once the gates are assigned to subcircuits, the `cut()` method:
  - Traverses the original circuit topologically and constructs, for each subcircuit:
    - `subcircuit_instructions[subcircuit_i]`: list of Instructions mapped to that subcircuit and local wire indices.
    - When the active subcircuit on a wire changes, inserts a `WireCutGate()` into `self.circuit_with_cut_gates` and increments `self.num_cuts`.
  - Once parsed, it builds actual `QuantumCircuit` objects for each subcircuit from `subcircuit_instructions`.
  - `complete_path_map[wire_index] → [(subcircuit_i, subcircuit_wire_index), ...]`: the path a qubit takes across subcircuits.


- Towards the end of `cut()`:
  - `populate_compute_graph()`:
    - This step creates `self.compute_graph` (nodes and edges):
      - For each subcircuit `j`: node attributes include
        - `effective`: the number of effective qubits (reduced by outgoing cuts)
        - `rho` and `O` counts (incoming/outgoing cut roles)
      - For each wire path in `complete_path_map`, adds edges between adjacent (upstream, downstream) subcircuits with attributes:
        - `O_qubit`: local outgoing qubit index in upstream subcircuit
        - `rho_qubit`: local incoming qubit index in downstream subcircuit
  - `populate_subcircuit_entries()`:
    - For each subcircuit node, this step enumerates incident edges and I/X/Y/Z choices, and builds:
      - `self.subcircuit_entries[subcircuit_idx]`: mapping of (initialization_labels, measurement_labels) to list of coefficient-weighted terms;
      - `self.subcircuit_instances[subcircuit_idx]`: deduplicated list of all (initialization, measurement) tuples needed to run for that subcircuit.

    - Why this matters:
       - This defines what experiments must be run *per subcircuit* to gather probability vectors needed for reconstruction.

```{admonition} Compute Graph
:class: note

See the [Circuit-Cutting Implementation Details](notebooks/04_circuit_cutting_internals.ipynb) tutorial for more details on the Computation Graph.
```

Result:
- `self.subcircuits`: list of `QuantumCircuit` subcircuits.
- `self.circuit_with_cut_gates`: the original circuit with `//` markers showing cuts.
- `self.complete_path_map`: how original wires traverse subcircuits and local qubits.

What you’ll see:
- `print(cut_circuit)` now prints `self.circuit_with_cut_gates` and shows `//` where cuts were inserted.
- Iterating over `cut_circuit` using `for subcircuit in cut_circuit:` iterates through each subcircuit, which can then be printed out.


### 3) Executing subcircuits and packing probabilities

Method: `run_subcircuits(subcircuits=None, backend="statevector_simulator")`

- Runs each subcircuit's instances:
  - Uses utility functions `run_subcircuit_instances` to execute and `attribute_shots` to map measured results to entries. This step fills:
    - `self.subcircuit_entry_probs[subcircuit_idx][(init, meas)] = 2^k-length probability vector`
    - `self.subcircuit_packed_probs[subcircuit_idx] = get_packed_probabilities(subcircuit_idx)`
  - The method: `get_packed_probabilities(subcircuit_i, qubit_spec=None)`, for each subcircuit:
    - Builds a dense tensor with one axis of length 4 per incident edge (I/X/Y/Z), and a final axis of length `2^k` where `k` is the effective qubits for that subcircuit.
    - If `qubit_spec` is provided for that subcircuit, applies `merge_prob_vector` to reduce the last axis. (This functionality is used in `compute_probabilities()` later).

After `run_subcircuits()`:
- `self.subcircuit_entry_probs` and `self.subcircuit_packed_probs` are populated and ready for the reconstruction phase.


### 4) Save and load

Method: `to_file(filepath | None, ...)`
- Saves the cut circuit and reconstruction artifacts to `.zarr`. If `filepath` is `None`, uses a default scheme including qubit count, cut count, subcircuit count, and a timestamp.

Method: `from_file(filepath)`
- Loads a saved `CutCircuit` from `.zarr`, rehydrating its state for postprocessing or plotting.


### 5) Reconstruction setup

When you call `postprocess(capacity, max_recursion)`, where `capacity` is the max active qubits allowed at any step, and `max_recursion` is the max depth of recursive refinement:
- It computes the flow structure across all subcircuits, and maps incoming positions to outgoing positions in a flattened layout used during aggregation.
- Builds `self.dynamic_definition = DynamicDefinition(num_qubits, capacity, prob_fn=self.compute_probabilities)`
- Calls `self.dynamic_definition.run(max_recursion=...)`
  - This triggers the first call to `compute_probabilities(qubit_spec=None)` for an all-active (up to the desired capacity) initial qubit spec, and then recursive bin refinement, calling back to `compute_probabilities` with refined qubit specs in each recursion.

```{admonition} What *qubit_spec* means here
:class: note

The term *qubit_spec* is used throughout the codebase. This is a string over `{"A","M","0","1"}` across all effective qubits (in an internal, consistent order). `"A"` keeps a qubit active (resolved), `"M"` allows merging (marginalizing), and "0"/"1" fixes it to that state.

```

### 6) Computing probabilities (MPI-aware core)

Method: `compute_probabilities(qubit_spec=None)`

- This method breaks down the problem:
  - Partitions the global `qubit_spec` into per-subcircuit qubit spec.
  - Builds a local `subcircuit_packed_probs` dictionary for all subcircuits, applying any merging implied by the per-subcircuit qubit spec.
  - Defines the total "work" as the Cartesian product across all input cut-edge Pauli choices: `total_work = n_basis ** sum(self.in_degrees)` with `n_basis=4`.
- Parallel execution:
  - Rank 0 operates as a scheduler; workers (ranks > 0) receive chunks of initialization tuples and return partial probability vectors.
  - Scheduler (Rank 0):
    - Slices the iterator of work into balanced chunks.
    - Sends `[active_qubits, subcircuit_packed_probs, work_chunk]` to each worker with `MPI_WORK_TAG`.
    - Receives partial results with `MPI_RESULT_TAG` and accumulates them.
    - Sends `MPI_DONE_TAG` to each worker when no more chunks remain.
  - Worker (Rank > 0):
    - Receives a message; if `MPI_DONE_TAG`, breaks; otherwise, calls `_compute_probabilities(...)` on the chunk and replies with the partial result.

```{admonition} _compute_probabilities
:class: important

Method: `_compute_probabilities(active_qubits, subcircuit_packed_probs, initializations_list, ...)`
  - **This is the core GPU-aware computational kernel**.
  - This step computes the partial probability vector for a given list of initialization tuples.
  - For each initialization tuple, it constructs the per-subcircuit index `(init_slice, meas_slice, Ellipsis)` to pull the probability vector from `subcircuit_packed_probs`.
  - Finally, it combines subcircuits left-to-right using `vector_kron` (a Python wrapper over a CUDA kernel) to form a single `2^active_qubits` vector.
```

At the end:
- Rank 0 divides by `2**self.num_cuts` (normalization of the stitched distribution), broadcasts to all ranks, and returns the vector.

```{admonition} Why this design
:class: note

- A single call to `compute_probabilities` can be very expensive; distributing embarrassingly parallel initialization work scales linearly across ranks.
- Since each rank calls a CUDA kernel, this step can utilize multiple GPUs across nodes.
```


### 7) Plotting and verifying probabilities

Method: `get_probabilities(full_states=None, quasi=False)`
- If `full_states` is `None`, creates all integers `[0, ..., 2^n - 1]`.
- Computes the reconstruction bit permutation:
  - `perm = self.reconstruction_flat_qubit_order()`
    - Internally:
      - Builds a flat list (by greedy subcircuit order) of original qubit indices that the downstream reconstruction yields.
      - Converts it into a permutation vector so we can permute bits correctly. The permuted indices are stored as `permuted_indices`.
- Calls `self.dynamic_definition.probabilities(full_states=permuted_indices)` to sum all bins into requested output indices; optionally converts quasi to real probabilities.

Method: `verify(probabilities, backend="statevector_simulator", atol=1e-10, raise_error=True)`
- Computes ground truth via `evaluate_circ(self.circuit, backend)` and returns a normalized MSE error. Raises an Exception (optional) if error is above tolerance.

Method: `plot(plot_ground_truth=False, full_states=None, output_file=None)`
- Plots reconstructed probabilities, optionally overlaying ground truth.

```{admonition} Note
:class: note

Verifying probabilities is only possible for small circuits where the full statevector can be simulated. Plotting is possible for larger circuits, as long as `full_states` is judiciously selected and not too large.
```
