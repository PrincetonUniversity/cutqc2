import ast

import qiskit
from cudaq import draw, kernel

"""
Mapping of Qiskit gates to CUDAq gates.
If a gate is not found here, it is assumed that it has the same name in
CUDAq and Qiskit.
"""
QISKIT_TO_CUDAQ_GATES = {}


class Kernel:
    def __init__(self, kernel_name: str, qiskit_circuit: qiskit.QuantumCircuit):
        self.kernel_name = kernel_name
        self._module = self.get_cudaq_ast_module(self.kernel_name, qiskit_circuit)
        self.src = ast.unparse(self._module)

        """
        `cudaq.kernel.kernel_decorator.PyKernelDecorator` has somewhat strange
        behavior. It inspects `function` to see if its a string to determine
        whether to deserialize, yet then looks at funcSrc to do the actual
        deserialization.
        """
        self.wrapped_kernel = kernel(
            funcSrc=self.src, kernelName=self.kernel_name, signature={}
        )(function="")

    def __call__(self) -> None:
        return self.wrapped_kernel()

    def __str__(self) -> str:
        return draw(self.wrapped_kernel)

    @staticmethod
    def get_cudaq_ast_module(name: str, qc: qiskit.QuantumCircuit) -> ast.Module:
        def gate(
            qc: qiskit.QuantumCircuit,
            gate: str,
            instr: qiskit.circuit.quantumcircuitdata.CircuitInstruction,
        ) -> ast.Expr:
            # First, add any params
            params = []
            for param in instr.params:
                params.append(ast.Constant(param))

            # Second, add any qubit(s) on which this gate operates
            qubits = []
            for qubit in instr.qubits:
                """
                It would be nice to obtain the qubit wire index from the Qubit
                object itself, but there seems to be no way other than to reach
                for its private `._index` attribute.
                So we choose to find it in `qc.qubits` instead.
                """
                qubit_index = qc.qubits.index(qubit)

                qubits.append(
                    ast.Subscript(
                        value=ast.Name(id="qubits", ctx=ast.Load()),
                        slice=ast.Constant(qubit_index),
                        ctx=ast.Load(),
                    )
                )

            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(
                        id=QISKIT_TO_CUDAQ_GATES.get(gate, gate), ctx=ast.Load()
                    ),
                    args=[*params, *qubits],
                    keywords=[],
                )
            )

        body = []
        qvec_assign = ast.Assign(
            lineno=None,
            targets=[ast.Name(id="qubits", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="cudaq", ctx=ast.Load()),
                    attr="qvector",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=qc.num_qubits)],
                keywords=[],
            ),
        )
        body.append(qvec_assign)

        for instr in qc.data:
            instr_name = instr.name
            if instr_name in ("reset", "measure", "measure_all", "barrier"):
                continue  # TODO: Handle these!
            else:  # assume gate
                body.append(gate(qc, instr_name, instr))

        return ast.Module(
            body=[
                ast.FunctionDef(
                    name=name,
                    lineno=None,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=body,
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                )
            ],
            type_ignores=[],
        )
