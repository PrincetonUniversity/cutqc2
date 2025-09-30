#!/bin/bash

cutqc2 cut \
  --file bv_38qubit.qasm3 \
  --max-subcircuit-width 6 \
  --max-cuts 100 \
  --num-subcircuits 10 \
  --output-file bv_38qubit.zarr

cutqc2 postprocess \
  --file bv_38qubit.zarr \
  --capacity 20 \
  --max-recursion 10
  --save

cutqc2 plot \
  --file bv_38qubit.zarr \
  --output-file bv_38qubit.png
