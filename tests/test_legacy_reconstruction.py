import numpy as np

from cutqc2.legacy.cutqc.cutqc.dynamic_definition import read_dd_bins


def test_reconstruction():
    """
    Test reconstruction for a 5-qubit circuit with 2 subcircuits (with 2 and 3 qubits respectively).
    """

    bins = np.array(
        [
            0.00,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
            1.10,
            1.20,
            1.30,
            1.40,
            1.50,
        ]
    )

    result = read_dd_bins(
        # Mapping from subcircuit index to qubit indices
        subcircuit_out_qubits={0: [1, 0], 1: [2, 3]},
        # Mapping from recursion level to a dict of bin information
        dd_bins={
            0: {
                # Mapping from subcircuit index to the state of its qubits (active/merged)
                "subcircuit_state": {
                    0: ["active", "active"],
                    1: ["active", "active"],
                },
                # Ordering of subcircuits
                "smart_order": [0, 1],
                # 2^4 = 16 probability values
                # these are made-up values so we can easily keep track
                # of where they get moved
                "bins": bins,
                "expanded_bins": [],
            }
        },
    )

    # 2^4 = 16 "reconstructed" probability values
    assert np.allclose(
        result,
        np.array(
            [
                0.0,
                0.4,
                0.8,
                1.2,
                0.2,
                0.6,
                1.0,
                1.4,
                0.1,
                0.5,
                0.9,
                1.3,
                0.3,
                0.7,
                1.1,
                1.5,
            ]
        ),
    )

    # A simpler approach for the case where all the qubits are "active"
    # would be:
    def apply_lookup(n, L):
        # Apply a lookup table to map the bits of n to the indices in L
        output = 0
        for i in range(len(L)):
            if (n >> i) & 1:  # if the i-th bit of n is set
                output |= 1 << L[i]  # set the bit in output at position L[i]
        return output

    indices = np.array([apply_lookup(i, [3, 2, 0, 1]) for i in range(16)])
    expected_result = np.zeros(len(indices))
    expected_result[indices] = bins
    assert np.allclose(result, expected_result)
