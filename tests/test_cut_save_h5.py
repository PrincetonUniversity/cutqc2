import numpy as np

from cutqc2.core.cut_circuit import CutCircuit


def test_figure4_save(figure_4_qiskit_circuit, tmp_path):
    save_path = tmp_path / "test_cut_circuit_figure4_to_file.h5"

    # We should be able to save the cut circuit at arbitrary points of
    # the processing pipeline.
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_circuit.to_file(save_path)

    cut_circuit.cut(
        max_subcircuit_width=3,
        max_subcircuit_cuts=2,
        subcircuit_size_imbalance=3,
        max_cuts=1,
        num_subcircuits=[2],
    )
    cut_circuit.to_file(save_path)

    cut_circuit.run_subcircuits()
    cut_circuit.to_file(save_path)

    cut_circuit.postprocess()
    probabilities = cut_circuit.get_probabilities()
    cut_circuit.to_file(save_path)

    cut_circuit.verify(probabilities)
    cut_circuit.to_file(save_path)


def test_figure4_load_complete_path_map(figure_4_qiskit_circuit, tmp_path):
    save_path = tmp_path / "test_cut_circuit_figure4_to_file.h5"

    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_circuit.cut(
        max_subcircuit_width=3,
        max_subcircuit_cuts=2,
        subcircuit_size_imbalance=3,
        max_cuts=1,
        num_subcircuits=[2],
    )
    cut_circuit.to_file(save_path)

    cut_circuit2 = CutCircuit.from_file(save_path)

    # For now we just compare the subcircuit entry probabilities
    # Note that we haven't done a `run_subcircuits` on `cut_circuit2`,
    # yet we can get the subcircuit entry probabilities from the file.
    assert (
        cut_circuit.subcircuit_entry_probs.keys()
        == cut_circuit2.subcircuit_entry_probs.keys()
    )
    for k, v in cut_circuit.subcircuit_entry_probs.items():
        for initializations_measurements, probabilities in v.items():
            assert (
                initializations_measurements in cut_circuit2.subcircuit_entry_probs[k]
            )
            assert np.allclose(
                probabilities,
                cut_circuit2.subcircuit_entry_probs[k][initializations_measurements],
            )
