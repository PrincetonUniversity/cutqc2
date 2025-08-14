from mpi4py import MPI
import click
from cutqc2.core.cut_circuit import CutCircuit


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--file", required=True, help="qasm3 file location.")
@click.option(
    "--max-subcircuit-width", help="Max subcircuit width.", type=int, default=6
)
@click.option(
    "--max-subcircuit-cuts", help="Max subcircuit cuts.", type=int, default=10
)
@click.option(
    "--subcircuit-size-imbalance",
    help="Subcircuit size imbalance.",
    type=int,
    default=3,
)
@click.option("--max-cuts", help="Max cuts.", type=int, default=10)
@click.option(
    "--num-subcircuits",
    help="Number of subcircuits to try.",
    type=int,
    multiple=True,
    default=(5,),
)
@click.option(
    "--output-file",
    type=str,
    help="Output file to save the cut circuit in Zarr format.",
)
def cut(
    file,
    max_subcircuit_width,
    max_subcircuit_cuts,
    subcircuit_size_imbalance,
    max_cuts,
    num_subcircuits,
    output_file,
):
    with open(file, "r") as f:
        circuit_qasm3 = f.read()
    cut_circuit = CutCircuit(circuit_qasm3=circuit_qasm3, preallocate=True)
    cut_circuit.cut(
        max_subcircuit_width=max_subcircuit_width,
        max_subcircuit_cuts=max_subcircuit_cuts,
        subcircuit_size_imbalance=subcircuit_size_imbalance,
        max_cuts=max_cuts,
        num_subcircuits=list(num_subcircuits),
    )

    cut_circuit.run_subcircuits()
    if output_file:
        cut_circuit.to_file(output_file)


@cli.command()
@click.option("--file", required=True, help="Zarr file location.")
@click.option("--capacity", default=16, help="Capacity for postprocessing.")
@click.option(
    "--max-recursion", default=20, help="Maximum recursion depth for postprocessing."
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Verify the results after postprocessing.",
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="Save results to Zarr file after postprocessing.",
)
@click.option("--atol", default=1e-8, help="Absolute tolerance for verification.")
def postprocess(file, capacity, max_recursion, verify, save, atol):
    cut_circuit = CutCircuit.from_file(file)
    probabilties = cut_circuit.postprocess(
        capacity=capacity, max_recursion=max_recursion, compute=False
    )
    if rank == 0:
        if verify:
            cut_circuit.verify(probabilties, atol=atol, raise_error=False)
        if save:
            cut_circuit.to_file(file)


@cli.command()
@click.option("--file", required=True, help="Zarr file location.")
@click.option("--atol", default=1e-8, help="Absolute tolerance for verification.")
def verify(file, atol):
    cut_circuit = CutCircuit.from_file(file)
    cut_circuit.verify(cut_circuit.probabilities, atol=atol, raise_error=True)


if __name__ == "__main__":
    cli()
