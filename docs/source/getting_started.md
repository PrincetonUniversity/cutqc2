# Getting started

`cutqc2` has 3 main functionalities:

1. **Circuit cutting**: Cut large quantum circuits into smaller sub-circuits that can be executed on smaller quantum devices.
2. **Execution**: Execute the sub-circuits on quantum hardware or simulators, and collect the results.
3. **Postprocessing**: Reconstruct the results of the original circuit from the results of the sub-circuits.
4. **(Optional) Verify**: Verify the results of the original circuit using classical simulation.
5. **(Optional) Visualize**: Plot the probability distribution of the results.

## Circuit cutting and execution

Currently, steps 1 and 2 are performed using the `cutqc2 cut` command.

Say you have a qasm3 file representing a quantum circuit. Some examples are provided in the `examples/scripts` folder in the codebase. You can cut and execute the circuit as follows:

```
cutqc2 cut \
  --file supremacy_6qubit.qasm3 \
  --max-subcircuit-width 5 \
  --max-subcircuit-cuts 10 \
  --subcircuit-size-imbalance 2 \
  --max-cuts 10 \
  --num-subcircuits 3 \
  --output-file supremacy_6qubit.zarr
```

This will:
 - Cut the circuit into sub-circuits with a maximum width of 5 qubits, and a maximum of 10 cuts.
 - The `subcircuit-size-imbalance` parameter controls how much larger one sub-circuit can be compared to another.
 - The `num-subcircuits` parameter specifies how many sub-circuits to create.
 - Run each sub-circuit on a simulator (by default, `qiskit`'s statevector simulator is used).
 - Save results (that can be used for reconstruction) in a `zarr` file named `supremacy_6qubit.zarr`.

## Postprocessing

Once you have the `zarr` file with the results of the sub-circuits, you can reconstruct the results of the original circuit using the `cutqc2 postprocess` command:

```
cutqc2 postprocess \
  --file supremacy_6qubit.zarr \
  --save
```

This will:
 - Read the `zarr` file with the results of the sub-circuits.
 - Combine (reconstrcut) the results so they are identical to what you would have obtained by executing the original circuit. *This is a computationally intensive step!* that can benefit from GPU and MPI support.
 - Save the reconstructed results back into the same `zarr` file.

## Verification

Finally, you can verify the results of the original circuit using classical simulation with the `cutqc2 verify` command:

```
cutqc2 verify \
  --file supremacy_6qubit.zarr
```

This will:
 - Read the `zarr` file with the results of the original circuit (reconstructed in the previous step).
 - Classically simulate the original circuit using `qiskit`'s statevector simulator to obtain reference results. 
 - Compare the reconstructed results with the reference results.

Obviously, running the original circuit using classical simulation is only feasible for small circuits (up to about 20 qubits).

## Visualization

This step may not work yet. Coming soon!

You can see some complete examples in the `examples/scripts` folder.
