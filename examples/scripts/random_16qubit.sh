#!/bin/bash

cutqc2 cut \
  --file random_16qubit.qasm3 \
  --max-subcircuit-width 6 \
  --max-subcircuit-cuts 10 \
  --subcircuit-size-imbalance 3 \
  --max-cuts 10 \
  --num-subcircuits 5 \
  --output-file random_16qubit.zarr

cutqc2 postprocess \
  --file random_16qubit.zarr \
  --save

# Do this only for small circuits!
cutqc2 verify \
  --file supremacy_6qubit.zarr