#!/bin/bash

# The following command downloads and unzips a pre-cut and pre-run 16 qubit
# random circuit with 5 subcircuits. It is saved as `random_16q_5s.zarr.unzip`
# in the current folder.

# You can see the list of available circuits to download by running
# `cutqc2 download --list`.
cutqc2 download \
  --file random_16q_5s.zarr \
  --path .

# To actually generate the file locally, you can run
# cutqc2 cut \
#   --file random_16qubit.qasm3 \
#   --max-subcircuit-width 6 \
#   --max-cuts 10 \
#   --num-subcircuits 5 \
#   --output-file random_16qubit.zarr
# and adjust the path the subsequent steps accordingly.

# Notice the `--save` to save the post-processed results back to the
# `.zarr` "file".
cutqc2 postprocess \
  --file ./random_16q_5s.zarr.unzip \
  --save

## Do this only for small circuits!
cutqc2 verify \
  --file random_16q_5s.zarr.unzip

cutqc2 plot \
  --file random_16q_5s.zarr.unzip \
  --output-file random_16qubit.png