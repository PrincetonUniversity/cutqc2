# Cluster deployment

The following notes detail deployment-related steps for `cutqc2` and `cuQuantum` on the [Della](https://researchcomputing.princeton.edu/systems/della) cluster at Princeton. Other clusters at Princeton should work with more or less the same steps.

## Installing and deploying cutqc2

The cluster has several `anaconda3` modules to choose from, which help us create a new python environment to install `cutqc2`. At the time of this writing, the following steps were used to install `cutqc2`:

```
module load anaconda3/2024.10
conda create --name cutqc2 python=3.12 pip
conda activate cutqc2
git clone https://github.com/PrincetonUniversity/cutqc2.git
cd cutqc2
pip install -e .[dev,docs,gpu]
```

We now run a sample script in the `examples/scripts` folder to verify that everything works.

Since we're running this on the head node, we'll choose the simplest example, `supremacy_6qubit.sh`:
Before running, we'll load the modules that give us access to MPI. We'll also run this with the environment variable `CUTQC2_CORE_NUMERIC` set to `numpy`, since the head node does not have a GPU.
See the [FAQ](faqs.md) page for more details.

```
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
export CUTQC2_CORE_NUMERIC=numpy
cd examples/scripts
bash supremacy_6qubit.sh
```

The example should take ~10 seconds to run.

When submitting jobs to the cluster's job scheduler (SLURM), use a `.sbatch` file similar to `examples/scripts/job.sbatch`.

## Installing and deploying cuQuantum

`cuquantum` (specifically, `cutensornet`) is a dependency of `cutqc2`, though it is not currently utilized for tensor contraction. For future work, it may be useful to experiment with `cutensornet` on Della.
The following writeup details the steps to get `cuQuantum` working on a multi-GPU cluster with cuda-aware MPI. The steps were tested on the Della cluster @ Princeton on 09/16/2025, and are mostly a tweaked variation on the official install [instructions](https://docs.nvidia.com/cuda/cuquantum/22.05.0/cutensornet/getting_started.html).

### cuquantum

```
wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-25.06.0.10_cuda12-archive.tar.xz
```
export `CUQUANTUM_ROOT` environment variable to whereever you `tar xf` the file above, and add it to your `~/.bash_profile` or equivalent.

### cutensor

```
wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-2.3.0.6_cuda12-archive.tar.xz
```

export `CUTENSOR_ROOT` environment variable to whereever you `tar xf` the file above, and add it to your `~/.bash_profile` or equivalent.

In subsequent steps, we will assume (modify any `job.sbatch` files accordingly if this path is different):
```
CUQUANTUM_ROOT=/scratch/gpfs/vineetb/ext/cuquantum-linux-x86_64-25.06.0.10_cuda12-archive
CUTENSOR_ROOT=/scratch/gpfs/vineetb/ext/libcutensor-linux-x86_64-2.3.0.6_cuda12-archive
```

### Clone cuQuantum repo

```
cd ~
git clone git@github.com:NVIDIA/cuQuantum.git
cd cuQuantum
git checkout v25.06.0
```

`v25.06.0` is the last tag that was tested with CUDA 12.9. Later tags *might* work, but we likely need `module load cudatoolkit/13.0` etc in the steps below.

### Compile cuda-aware MPI library

```
export CUDA_PATH=/usr/local/cuda-12.9
export MPI_PATH=/usr/local/openmpi/cuda-12.9/4.1.8/nvhpc255/
cd $CUQUANTUM_ROOT/distributed_interfaces
source activate_mpi_cutn.sh
echo $CUTENSORNET_COMM_LIB
```

This compiles `libcutensornet_distributed_interface_mpi.so`. Note down the location of `$CUTENSORNET_COMM_LIB`. This is used in `job.sbatch` files that follow. If you specify it relative to `$CUQUANTUM_ROOT`, you shouldn't need to change it anyway.

## cuQuantum CUDA examples

### Single GPU example

#### Compile

```
module load cudatoolkit/12.9
cd ~/cuQuantum/samples/cutensornet
nvcc tensornet_example.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -o tensornet_example
```

#### Run

We can run this interactively by requesting a GPU node using `srun`:

```
srun -t 00:05:00 --mem=64G --gres=gpu:1 --pty /bin/bash
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${CUTENSOR_ROOT}/lib:${LD_LIBRARY_PATH}
./tensornet_example
```

`CUQUANTUM_ROOT` and `CUTENSOR_ROOT` should be set on the compute node where you try this, of course (add them to your `~/.bash_profile` or redefine them once on the compute node).

Ensure you're back on the head node before proceeding!

### cuda-aware MPI example

A successful run of this verifies that CUDA-aware MPI is working fine with cuQuantum.

#### Compile

```
module load cudatoolkit/12.9
cd ~/cuQuantum/samples/cutensornet
export MPI_PATH=/usr/local/openmpi/cuda-12.9/4.1.8/nvhpc255/
nvcc tensornet_example_mpi_auto.cu -I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -I${MPI_PATH}/include -L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -L${MPI_PATH}/lib64 -lmpi -o tensornet_example_mpi_auto
```

#### job.sbatch

Submit using `sbatch job.sbatch`:

```
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH -t 00:03:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --output=log.txt

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8

export CUQUANTUM_ROOT=/scratch/gpfs/vineetb/ext/cuquantum-linux-x86_64-25.06.0.10_cuda12-archive
export CUTENSOR_ROOT=/scratch/gpfs/vineetb/ext/libcutensor-linux-x86_64-2.3.0.6_cuda12-archive
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${CUTENSOR_ROOT}/lib:${LD_LIBRARY_PATH}
export CUTENSORNET_COMM_LIB=${CUQUANTUM_ROOT}/distributed_interfaces/libcutensornet_distributed_interface_mpi.so

srun ~/cuQuantum/samples/cutensornet/tensornet_example_mpi_auto
```

## CuQuantum Python examples

### Create conda environment with cuquantum

We create the conda environment at `/scratch/gpfs/vineetb/envs/cuquantum` simply to save space in the home folder. Modify as needed.

```
module load anaconda3/2024.10
conda create --prefix /scratch/gpfs/vineetb/envs/cuquantum1 python=3.12 pip
conda activate /scratch/gpfs/vineetb/envs/cuquantum
conda install conda-forge::cuquantum-python
pip install mpi4py
```

### Run python cuda-aware MPI example

The example `example22_mpi_auto.py` is packaged in the `cuQuantum` repo as `python/samples/tensornet/contraction/coarse/example22_mpi_auto.py`.
A successful run of this verifies that CUDA-aware MPI is working fine with cuQuantum using Python.

#### job.sbatch

Submit using `sbatch job.sbatch`:

```
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G
#SBATCH -t 00:01:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig"
#SBATCH --output=log.txt

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load cudatoolkit/12.9
module load anaconda3/2024.10

export CUQUANTUM_ROOT=/scratch/gpfs/vineetb/ext/cuquantum-linux-x86_64-25.06.0.10_cuda12-archive
export CUTENSOR_ROOT=/scratch/gpfs/vineetb/ext/libcutensor-linux-x86_64-2.3.0.6_cuda12-archive
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${CUTENSOR_ROOT}/lib:${LD_LIBRARY_PATH}
export CUTENSORNET_COMM_LIB=${CUQUANTUM_ROOT}/distributed_interfaces/libcutensornet_distributed_interface_mpi.so

conda activate /scratch/gpfs/vineetb/envs/cuquantum

mpirun -np 2 python ~/cuQuantum/python/samples/tensornet/contraction/coarse/example22_mpi_auto.py
```

### Run distributed Kronecker product calculation using cuda-aware MPI

The following example does a distributed Kronecker product calculation using cuda-aware MPI.

```
import os
import sys 

import cupy as cp
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI  # this line initializes MPI

from cuquantum.bindings import cutensornet as cutn
from cuquantum.tensornet import contract, get_mpi_comm_pointer


root = 0 
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

if not "CUTENSORNET_COMM_LIB" in os.environ:
    raise RuntimeError("need to set CUTENSORNET_COMM_LIB to the path of the MPI wrapper library")

if not os.path.isfile(os.environ["CUTENSORNET_COMM_LIB"]):
    raise RuntimeError("CUTENSORNET_COMM_LIB does not point to the path of the MPI wrapper library")

device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()

expr = 'i,j->ij'
shapes = [(2**int(sys.argv[1]),), (2**int(sys.argv[2]),)]

if rank == root:
    operands = [cp.arange(*shape).astype('float32') for shape in shapes]
else:
    operands = [cp.empty(shape) for shape in shapes]

for operand in operands:
   comm.Bcast(operand, root)

handle = cutn.create()
cutn.distributed_reset_configuration(
    handle, *get_mpi_comm_pointer(comm)
)

result = contract(expr, *operands, options={'device_id' : device_id, 'handle': handle})

# Check correctness - not always possible
if rank == root and len(sys.argv) > 3 and sys.argv[3] == 'check':
   result_cp = cp.einsum(expr, *operands, optimize=True)
   print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result, result_cp))
```

When saved as `kron.py` and run using the `job.sbatch` sbatch script above, changing the last line to:
```
mpirun -np 2 python kron.py 10 21
```

Things should work fine till around a combined size of 33 elements (2^33 = about 32G). Beyond that, we get an `out-of-memory` error, as expected.
Use `python kron.py 10 21 check` for small values of the arguments to verify correctness.