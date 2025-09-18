# FAQs

## Installation

### How do I use cutqc2 with GPU support?
Install the package with the `gpu` extra, in addition to any others you want (you will typically want the `dev` extra too):
```
pip install -e .[dev,gpu]
```

### I don't have a GPU. Can I still use cutqc2?
Yes. `cutqc2` can run on systems without a GPU. However, performance will be slower, so it should only be used for testing and development purposes.
Install the package without the `gpu` extra:
```
pip install -e .[dev]
```

If you have already installed the package but want to run it without a GPU, you can set the `CUTQC2_CORE_NUMERIC` environment variable to `numpy`:
```
export CUTQC2_CORE_NUMERIC=numpy
```
This will make `cutqc2` use `numpy` instead of `cupy` for numerical computations.

### How do I run cutqc2 with MPI?

Currently the `cutqc2 postprocess` step can benefit from MPI support. If you've read the [Command line Usage](notebooks/merge_unmerge.ipynb) tutorial, you will know that normally you run this command as follows:
```
cutqc2 postprocess \
  --file supremacy_6qubit.zarr \
  --save
```

To run it with MPI support, use `mpirun` or `mpiexec` to launch multiple processes. For example:
```
mpirun -n 4 cutqc2 postprocess \
  --file supremacy_6qubit.zarr \
  --save
```

Note that a main worker process coordinates work among the rest of the MPI ranks, so if you want to parallelize work across 2 nodes, you should run the command using `mpirun -np 3 ..`.
You will typically want to submit this command as a job to your cluster's job scheduler (e.g., SLURM, PBS, etc.). See a sample `.sbatch` file in the `examples/scripts` folder.

## Development

### I don't know much about MPI (or mpi4py). Any pointers?

For some helpful links on MPI, see [Learning resources: MPI](https://researchcomputing.princeton.edu/education/external-online-resources/mpi). For a quick introduction to `mpi4py`, especially its correct usage on the Princeton clusters, visit [this page](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py).