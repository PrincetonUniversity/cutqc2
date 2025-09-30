import sys

import pytest

from cutqc2.library.sample import sample_circuit


@pytest.fixture
def figure_4_qiskit_circuit():
    return sample_circuit


if sys.platform == "darwin":
    collect_ignore = ["test_cudaq.py", "test_qiskit_to_cudaq.py"]
