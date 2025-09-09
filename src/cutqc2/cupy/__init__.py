from pathlib import Path

from cutqc2.numeric import xp


def get_module():
    # This operation is expensive - call it as few times as possible.
    # We declare a global variable `cp_module` to store the compiled module.
    with Path.open(Path(__file__).parent / "cutqc.cu") as f:
        code = f.read()
    return xp.RawModule(code=code, options=("-std=c++11",))


cp_module = get_module() if xp.name == "cupy" else None


def vector_kron(a, b):
    assert a.ndim == 1
    assert b.ndim == 1
    if cp_module is None:
        return xp.kron(a, b)

    p, q = a.shape[0], b.shape[0]

    vector_kron_kernel = cp_module.get_function("vectorKron")

    threads_per_block = 256
    blocks_per_grid = (p * q + threads_per_block - 1) // threads_per_block

    result = xp.empty((p * q,), dtype=xp.float32)

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
