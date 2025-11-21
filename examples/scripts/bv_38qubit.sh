#!/bin/bash

cutqc2 cut \
  --file bv_38qubit.qasm3 \
  --max-subcircuit-width 6 \
  --max-cuts 100 \
  --num-subcircuits 10 \
  --output-file bv_38qubit.zarr

# Notice the `--save` to save the subcircuit probabilities back to the
# `.zarr` "file".
cutqc2 run \
  --file bv_38qubit.zarr \
  --save

# Notice the `--save` to save the post-processed results back to the
# `.zarr` "file".
cutqc2 postprocess \
  --file bv_38qubit.zarr \
  --capacity 20 \
  --max-recursion 10
  --save

cutqc2 plot \
  --file bv_38qubit.zarr \
  --output-file bv_38qubit.png
