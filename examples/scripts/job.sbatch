#!/bin/bash

# This sbatch script is provided as an example to run CutQC2 on a SLURM cluster.
# Since compute nodes may not have internet access, we recommend downloading the file
# on the head node using:
#   cutqc2 download --file random_16q_5s.zarr --path .

# This examples roughly illustrates what runs on our clusters - adjust as needed.
# We assume that a `random_16q_5s.zarr` folder exists in the same directory as this script.
# To submit the job, run: sbatch job.sbatch

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="nomig"
#SBATCH --output=log.txt

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load anaconda3/2024.10

conda activate cutqc2

export CUTQC2_CORE_NUMERIC=cupy
srun cutqc2 postprocess --file random_16qubit.zarr.unzip --save