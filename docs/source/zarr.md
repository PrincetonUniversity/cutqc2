# CutQC2 zarr file format

All relevant information during `Cutqc2` processing - the original circuit specification, cut locations, subcircuit probability values, and the final reconstructed probabilities - are stored in a single [zarr](https://zarr.readthedocs.io/en/stable/quickstart.html) file.
A zarr file is a directory containing multiple json files and binary data files, can be stored on disk or in cloud storage, allows for efficient access to subsets of the data without loading everything into memory,
as well as parallel read and write operations. It can be accessed from Python, C++ and Rust, among other languages.

See a detailed description of the Zarr format [here](https://zarr.dev/)

## Creating the zarr file

If we look at the `supremacy_6qubit.sh` script in the `examples/scripts` folder, it has a command to generate and cut a 6 qubit supremacy circuit.
```shell
    cutqc2 cut \
      --file supremacy_6qubit.qasm3 \
      --max-subcircuit-width 5 \
      --max-subcircuit-cuts 10 \
      --subcircuit-size-imbalance 2 \
      --max-cuts 10 \
      --num-subcircuits 3 \
      --output-file supremacy_6qubit.zarr
```

Run the above command to generate the `supremacy_6qubit.zarr` "file".

## Inspecting the zarr file

In a `.zarr` file (which is actually a folder), metadata is stored in json files (in `zarr.json` files) at various levels of the directory structure.
```shell
  $ cat supremacy_6qubit.zarr/zarr.json
  {
    "attributes": {
      "version": "0.0.6",
      "circuit_qasm": "OPENQASM 3.0;\ninclude
  ...
```

`.zarr` files are easily accessed and manipulated in Python, but to keep this discussion language-agnostic, here we will use the [jq](https://stedolan.github.io/jq/) tool to inspect and explain the contents of the json file.
`jq` is a command line tool commonly available on Linux and MacOS systems, and allows us to extract arbitrary fields from json data.

### Top-level metadata

The root `zarr.json` file contains metadata about the zarr file itself, including the Cutqc2 version number used to create it.

```shell
$ jq -r ".attributes.version" supremacy_6qubit.zarr/zarr.json
0.0.6
```

It is important to realize that the `.zarr` file created by one version of Cutqc2 may not be readable by another version of Cutqc2 (at least till we reach the `1.x` version).

Similarly, the original circuit in QASM format is stored in the `circuit_qasm` attribute.
```shell
$ jq -r ".attributes.circuit_qasm" supremacy_6qubit.zarr/zarr.json
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[0];
...
```

### Subcircuits

#### Number of subcircuits
```shell
$ jq -r ".attributes.n" supremacy_6qubit.zarr/subcircuits/zarr.json
3
```
Subcircuits are numbered from 0 to n-1, where n is the number of subcircuits.

#### Subcircuit metadata

The qasm3 representation of each subcircuit is stored as the `qasm` attribute in its numbered folder.

```shell
$ jq -r ".attributes.qasm" supremacy_6qubit.zarr/subcircuits/0/zarr.json
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
...
```

#### Subcircuit probabilities

The probabilities for each subcircuit are stored in the `packed_probabilities` group in its numbered folder. Let's inspect it's shape.

```shell
$ jq -r ".shape" supremacy_6qubit.zarr/subcircuits/0/packed_probabilities/zarr.json
[
  4,
  4,
  4,
  8
]
```

> [!NOTE]
> Why is the size (4, 4, 4, 8)? TODO: Computation Graph