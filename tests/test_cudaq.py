import cudaq


@cudaq.kernel
def initial_state():
    qubits = cudaq.qvector(3)
    h(qubits[0])  # noqa: F821
    for i in range(len(qubits) - 1):
        x.ctrl(qubits[0], qubits[i + 1])  # noqa: F821


def test_cudaq():
    results = cudaq.sample(initial_state, shots_count=100)
    results = dict(results.items())
    # We should only have '000' and '111' in the results
    assert all(key in ["000", "111"] for key in results)
