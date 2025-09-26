#!/bin/bash

cutqc2 cut \
  --file supremacy_6qubit.qasm3 \
  --max-subcircuit-width 5 \
  --max-cuts 10 \
  --num-subcircuits 3 \
  --output-file supremacy_6qubit.zarr

cutqc2 postprocess \
  --file supremacy_6qubit.zarr \
  --save

# Do this only for small circuits!
cutqc2 verify \
  --file supremacy_6qubit.zarr

cutqc2 plot \
  --file supremacy_6qubit.zarr \
  --output-file supremacy_6qubit.png