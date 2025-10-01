from pathlib import Path
import subprocess
import sys


THIS_DIR = Path(__file__).parent


def test_mpi():
    cmd = ["mpirun", "-n", "2", sys.executable, "--version"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert "Python" in result.stdout


def test_cuda_aware_mpi():
    cmd = ["mpirun", "-n", "2", sys.executable, f"{THIS_DIR}/_cuda_aware_mpi_check.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert "CUDA-aware broadcast OK" in result.stdout
