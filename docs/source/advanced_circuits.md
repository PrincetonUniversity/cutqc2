# Generating and cutting advanced circuits

`cutqc2` has a library of advanced quantum circuits in the `cutqc2.library` module. The functions in this module generate `qiskit.QuantumCircuit` objects that can be used as input to the `CutCircuit` class.

Circuits can be created using the `generate_circ` function, which takes the following parameters:
- `num_qubits`: The number of qubits in the circuit.
- `depth`: The depth of the circuit. This parameter may not apply to all circuit types, and can be set `None` if not needed for that circuit type.
- `circuit_type`: The type of circuit to generate. Options include `"supremacy"`, `"sycamore"`, `"bv"`, `"qft"`, `"aqft"`, and others.
- `reg_name`: The name of the quantum register. Default is `"q"`.
- `connected_only`: If `True` (the default), ensures that the circuit only contains gates between connected qubits. This option must be set to `True` if you expect to cut the circuit using `cutqc2`.
- `seed`: A seed for the random number generator to ensure reproducibility. Default `None`.

A typical way to generate and cut one of these circuit types would be:

```python
from cutqc2.library.utils import generate_circ
from cutqc2.core.cut_circuit import CutCircuit


circuit = generate_circ(
    num_qubits=38,
    depth=20,
    circuit_type="aqft",
    reg_name="q", 
    connected_only=True,
    seed=42
)
cut_circuit = CutCircuit(circuit)
cut_circuit.cut(
    max_subcircuit_width=30,
    max_cuts=10,
    num_subcircuits=[2]
)
```

## Supported Circuit Types

At the time of this writing, the following circuit types can be generated using `cutqc2.library.utils.generate_circ`:

### `supremacy`

Quantum Supremacy Circuit based on the implementations in [https://www.nature.com/articles/s41567-018-0124-x](https://www.nature.com/articles/s41567-018-0124-x) and [https://github.com/sboixo/GRCS](https://github.com/sboixo/GRCS).)

### `sycamore`

Quantum Supremacy Circuit as found in [https://www.nature.com/articles/s41586-019-1666-5](https://www.nature.com/articles/s41586-019-1666-5)

### `hwea` 

Hardware efficient ansatz for the QAOA algorithm. Based on the community detection circuit implemented by Francois-Marie Le Régent. This ansatz uses the entangler+rotation block structure like that described  in the paper by Nikolaj Moll et al. (http://iopscience.iop.org/article/10.1088/2058-9565/aab822)

### `bv`

Bernstein-Vazirani algorithm circuit.

### `qft`

Circuit to perform the Quantum Fourier Transform (or its inverse) as described in Mike & Ike Chapter 5.

### `aqft`

An *approximate* QFT that ignores controlled-phase rotations where the angle is beneath a threshold. This is discussed in more detail in https://arxiv.org/abs/quant-ph/9601018 or https://arxiv.org/abs/quant-ph/0403071.

### `adder`

An n-bit ripple-carry adder. Based on the specification given in Cuccaro, Draper, Kutin, Moulton. (https://arxiv.org/abs/quant-ph/0410184v1)

### `regular`

A QAOA-style variational circuit built on a 3-regular random graph with num_qubits nodes.

### `erdos`

A QAOA-style variational circuit (depth P = depth) on an Erdős–Rényi graph over num_qubits nodes.

### `random`

A generic random circuit of given width and depth, with connection degree 0.5 and a fixed number of Hadamards inserted.

