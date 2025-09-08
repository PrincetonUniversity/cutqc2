import textwrap

import cudaq
from qiskit.primitives import StatevectorSampler

from cutqc2.qiskit_to_cudaq import Kernel


def normalize(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def total_variation_distance(counts1, counts2):
    keys = set(counts1) | set(counts2)
    p = normalize({k: counts1.get(k, 0) for k in keys})
    q = normalize({k: counts2.get(k, 0) for k in keys})
    return 0.5 * sum(abs(p[k] - q[k]) for k in keys)


def test_kernel_str(figure_4_qiskit_circuit):
    cudaq_kernel = Kernel("my_cudaq_kernel", figure_4_qiskit_circuit)
    got_str = str(cudaq_kernel)
    expected_str = textwrap.dedent("""
             ╭───╮                  ╭───────────╮    ╭───╮    ╭───╮
        q0 : ┤ h ├──●────────●──────┤ rx(1.571) ├────┤ t ├────┤ h ├
             ├───┤╭─┴─╮      │      ├───────────┤    ├───┤    ├───┤
        q1 : ┤ h ├┤ z ├──────┼──────┤ rx(1.571) ├────┤ t ├────┤ h ├
             ├───┤├───┤    ╭─┴─╮    ╰───────────╯    ╰───╯    ├───┤
        q2 : ┤ h ├┤ t ├────┤ z ├──────────●────────────●──────┤ h ├
             ├───┤├───┤    ╰───╯          │          ╭─┴─╮    ├───┤
        q3 : ┤ h ├┤ t ├───────────────────┼──────────┤ z ├────┤ h ├
             ├───┤├───┤╭───────────╮    ╭─┴─╮    ╭───┴───┴───╮├───┤
        q4 : ┤ h ├┤ t ├┤ rx(1.571) ├────┤ z ├────┤ rx(1.571) ├┤ h ├
             ╰───╯╰───╯╰───────────╯    ╰───╯    ╰───────────╯╰───╯
    """).lstrip("\n")
    assert got_str == expected_str


def test_kernel_src(figure_4_qiskit_circuit):
    cudaq_kernel = Kernel("my_cudaq_kernel", figure_4_qiskit_circuit)
    got_src = cudaq_kernel.src.strip()
    expected_src = textwrap.dedent("""
        def my_cudaq_kernel():
            qubits = cudaq.qvector(5)
            h(qubits[0])
            h(qubits[1])
            h(qubits[2])
            h(qubits[3])
            h(qubits[4])
            cz(qubits[0], qubits[1])
            t(qubits[2])
            t(qubits[3])
            t(qubits[4])
            cz(qubits[0], qubits[2])
            rx(1.5707963267948966, qubits[4])
            rx(1.5707963267948966, qubits[0])
            rx(1.5707963267948966, qubits[1])
            cz(qubits[2], qubits[4])
            t(qubits[0])
            t(qubits[1])
            cz(qubits[2], qubits[3])
            rx(1.5707963267948966, qubits[4])
            h(qubits[0])
            h(qubits[1])
            h(qubits[2])
            h(qubits[3])
            h(qubits[4])
    """).strip()

    assert got_src == expected_src


def test_kernel_shot_distribution(figure_4_qiskit_circuit):
    figure_4_qiskit_circuit.measure_all()

    shots = 1_000_000
    qiskit_result = (
        StatevectorSampler().run([figure_4_qiskit_circuit], shots=shots).result()
    )
    # Qiskit count bitstrings are MSB-first: the leftmost bit corresponds to
    # the highest qubit index (the 'bottom' wire)
    qiskit_counts = qiskit_result[0].data.meas.get_counts()

    cudaq_kernel = Kernel("cudaq_kernel", figure_4_qiskit_circuit)
    cudaq_results = cudaq.sample(cudaq_kernel, shots_count=shots)
    cudaq_counts = dict(cudaq_results.items())
    # CudaQ count bitstrings are LSB-first: the leftmost bit corresponds to
    # the lowest qubit index (the 'top' wire)
    # We reverse the order to match the qiskit convention
    cudaq_counts = {k[::-1]: v for k, v in cudaq_counts.items()}

    tvd = total_variation_distance(qiskit_counts, cudaq_counts)
    assert tvd < 0.01, f"Total variation distance is too high: {tvd:.4f}"
