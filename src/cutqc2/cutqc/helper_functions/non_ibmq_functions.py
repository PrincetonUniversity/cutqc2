import random, pickle, os, copy, random
from qiskit import QuantumCircuit
import qiskit_aer as aer
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import psutil

from cutqc2.cutqc.helper_functions.conversions import dict_to_array


def evaluate_circ(circuit, backend, options=None):
    circuit = copy.deepcopy(circuit)
    max_memory_mb = psutil.virtual_memory().total >> 20
    max_memory_mb = int(max_memory_mb / 4 * 3)
    if backend == "statevector_simulator":
        simulator = aer.Aer.get_backend("statevector_simulator")
        result = simulator.run(circuit).result()
        statevector = result.get_statevector(circuit)
        prob_vector = Statevector(statevector).probabilities()
        return prob_vector
    elif backend == "noiseless_qasm_simulator":
        simulator = aer.Aer.get_backend("aer_simulator", max_memory_mb=max_memory_mb)
        if isinstance(options, dict) and "num_shots" in options:
            num_shots = options["num_shots"]
        else:
            num_shots = max(1024, 2**circuit.num_qubits)

        if isinstance(options, dict) and "memory" in options:
            memory = options["memory"]
        else:
            memory = False
        if circuit.num_clbits == 0:
            circuit.measure_all()
        result = simulator.run(circuit, shots=num_shots, memory=memory).result()

        if memory:
            qasm_memory = np.array(result.get_memory(circuit))
            assert len(qasm_memory) == num_shots
            return qasm_memory
        else:
            noiseless_counts = result.get_counts(circuit)
            assert sum(noiseless_counts.values()) == num_shots
            noiseless_counts = dict_to_array(
                distribution_dict=noiseless_counts, force_prob=True
            )
            return noiseless_counts
    else:
        raise NotImplementedError
