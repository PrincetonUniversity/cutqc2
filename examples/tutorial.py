from math import pi
import qiskit
from cutqc2.core.cut_circuit import CutCircuit


def example_circuit():
    qc = qiskit.QuantumCircuit(5)
    qc.reset(0)
    qc.reset(1)
    qc.reset(2)
    qc.reset(3)
    qc.reset(4)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)

    qc.cz(0, 1)
    qc.t(2)
    qc.t(3)
    qc.t(4)

    qc.cz(0, 2)
    qc.rx(pi / 2, 4)

    qc.rx(pi / 2, 0)
    qc.rx(pi / 2, 1)
    qc.cz(2, 4)

    qc.t(0)
    qc.t(1)
    qc.cz(2, 3)
    qc.rx(pi / 2, 4)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)

    return qc


if __name__ == "__main__":
    circuit = example_circuit()
    cut_circuit = CutCircuit(circuit)
    print(cut_circuit)

    # Cut the circuit automatically with specified parameters.
    cut_circuit.cut(
        max_subcircuit_width=3,
        max_subcircuit_cuts=2,
        subcircuit_size_imbalance=3,
        max_cuts=1,
        num_subcircuits=[2],
    )

    # Notice the location of the cut as the '//' marker gate.
    print(cut_circuit)

    print("----- Subcircuits -----")
    for subcircuit in cut_circuit:
        print(subcircuit)
        print()

    # Run all the subcircuits - by default we use the `statevector_simulator`
    # backend from qiskit.
    cut_circuit.run_subcircuits()

    # Save the cut circuit to a file
    # We could have done this at any point after creating the `CutCircuit` object.
    cut_circuit.to_file("tutorial.h5")

    # Load the cut circuit from the file
    cut_circuit = CutCircuit.from_file("tutorial.h5")

    # We perform the postprocessing step on the reloaded cut-circuit,
    # though we could have done this on the original cut-circuit as well.

    # Modify the `capacity` and `max_recursion` parameters to control
    # the accuracy/runtime of the reconstruction process.
    probabilities = cut_circuit.postprocess(capacity=3, max_recursion=9)

    # Verification involves comparing the generated probabilities
    # with the expected probabilities from a simulation of the uncut circuit.
    # By default we use the `statevector_simulator` backend from qiskit.
    error = cut_circuit.verify(probabilities, raise_error=False)
    print(f"Verification error: {error}")

    # Plot the results of the reconstruction
    cut_circuit.plot()
