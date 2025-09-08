from math import pi

import pytest
from qiskit import QuantumCircuit


@pytest.fixture(scope="function")
def figure_4_qiskit_circuit():
    qc = QuantumCircuit(5)
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
