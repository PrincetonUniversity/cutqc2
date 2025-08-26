"""
This file is not meant to picked up directly by `pytest`, but indirectly
via `tests/test_cuda_aware_mpi.py`, which runs it (as intended) via `mpirun`.
"""

import cupy as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    arr = cp.arange(10, dtype=cp.float32)
else:
    arr = cp.empty(10, dtype=cp.float32)

comm.Bcast(arr, root=0)

expected = cp.arange(10, dtype=cp.float32)
if cp.allclose(arr, expected):
    print(f"Rank {rank}: CUDA-aware broadcast OK")
else:
    print(f"Rank {rank}: FAILED")
