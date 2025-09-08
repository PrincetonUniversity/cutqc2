from pathlib import Path

import cupy as cp


def get_module():
    # This operation is expensive - call it as few times as possible.
    # We declare a global variable `cp_module` to store the compiled module.
    with Path.open(Path(__file__).parent / "cutqc.cu") as f:
        code = f.read()
    return cp.RawModule(code=code, options=("-std=c++11",))


cp_module = get_module()


def vector_kron(a: cp.array, b: cp.array):
    assert a.ndim == 1
    assert b.ndim == 1
    p, q = a.shape[0], b.shape[0]

    vector_kron_kernel = cp_module.get_function("vectorKron")

    threads_per_block = 256
    blocks_per_grid = (p * q + threads_per_block - 1) // threads_per_block

    result = cp.empty((p * q,), dtype=cp.float32)

    vector_kron_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            a,
            b,
            result,
            p,
            q,
        ),
    )
    return result
