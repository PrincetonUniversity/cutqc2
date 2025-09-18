import heapq
import logging
import warnings
from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt

from cutqc2.core.utils import unmerge_prob_vector

logger = logging.getLogger(__name__)


class Bin:
    """
    A Bin represents a collection of qubits with a specific configuration
    (given by `qubit_spec`) and the associated probabilities.
    """

    def __init__(self, qubit_spec: str, probabilities: np.ndarray):
        self.qubit_spec = qubit_spec
        self.probabilities = probabilities
        self.probability_mass = np.sum(probabilities).item()

    def __str__(self):
        return f"Bin({self.qubit_spec}, {self.probability_mass:.3f})"

    def __lt__(self, other):
        """
        This method is used to compare two `Bin` objects in the min-heap, and
        is thus used to decide whether to prioritize *this* bin over *other*.
        """
        return self.probability_mass > other.probability_mass


class DynamicDefinition:
    """
    DynamicDefinition is a class that implements the dynamic definition
    algorithm for quantum probability distribution reconstruction.
    It recursively zooms-in on qubits (initially "merged") that have a high
    probability mass.
    """

    def __init__(
        self, num_qubits: int, capacity: int, prob_fn: Callable, epsilon: float = 1e-4
    ):
        self.num_qubits = num_qubits
        self.capacity = capacity
        self.prob_fn = prob_fn

        # Probability-mass threshold below which we do not process a bin.
        self.epsilon = epsilon

        # The last recursion level processed, for reporting purposes.
        self.recursion_level = 0
        # A min-heap of `Bin` objects.
        self.bins = []

        # We maintain a set of qubit specifications that are present in
        # any of the bins, to avoid insertion of duplicates.
        self._qubit_specs_in_bins: set[str] = set()

    def __str__(self):
        return f"DynamicDefinition({self.num_qubits} qubits, {self.capacity} capacity, {len(self.bins)} bins)"

    def push(self, bin: Bin):
        if bin.qubit_spec not in self._qubit_specs_in_bins:
            heapq.heappush(self.bins, bin)
            self._qubit_specs_in_bins.add(bin.qubit_spec)

    def pop(self) -> Bin:
        if not self.bins:
            raise IndexError("No bins to pop")
        bin = heapq.heappop(self.bins)
        self._qubit_specs_in_bins.remove(bin.qubit_spec)
        return bin

    def run(self, max_recursion: int = 10, **kwargs) -> np.ndarray:
        # clear key attributes before running
        self.recursion_level = 0
        self.bins = []
        self._qubit_specs_in_bins = set()

        initial_qubit_spec = ("A" * self.capacity) + (
            "M" * (self.num_qubits - self.capacity)
        )
        logger.info(
            f"Calculating initial probabilities for qubit spec {initial_qubit_spec}"
        )
        initial_probabilities = self.prob_fn(initial_qubit_spec, **kwargs)
        initial_bin = Bin(initial_qubit_spec, initial_probabilities)

        self.push(initial_bin)
        if self.capacity < self.num_qubits:
            self._recurse(
                recursion_level=1,
                max_recursion=max_recursion,
                **kwargs,
            )

    def _recurse(
        self,
        recursion_level: int,
        max_recursion: int = 10,
        **kwargs,
    ):
        if not self.bins or (recursion_level > max_recursion):
            logger.info("No more bins to process or max recursion level reached.")
            return

        current_bin = self.pop()
        qubit_spec = current_bin.qubit_spec
        logger.info(f"{recursion_level=}, {qubit_spec=}")

        if (
            "M" not in qubit_spec and "A" not in qubit_spec
        ):  # zoomed-in completely; nothing else to do
            # undo the pop!
            self.push(current_bin)
            return

        self.recursion_level = recursion_level

        # For this bin, mark `capacity` (possibly fewer) merged qubits as active
        bin_qubit_spec = qubit_spec.replace("M", "A", self.capacity)
        bin_num_active_qubits = bin_qubit_spec.count("A")

        for j in range(2**bin_num_active_qubits):
            j_bin_qubit_spec = bin_qubit_spec  # reset
            # Replace all active qubits with the binary representation
            # of j - these become the "zoomed-in" bits.
            j_str = f"{j:0{bin_num_active_qubits}b}"  # `bin_num_active_qubits` length bit-string
            for j_char in j_str:
                j_bin_qubit_spec = j_bin_qubit_spec.replace("A", j_char, 1)

            logger.debug(f"{j + 1}/{2**bin_num_active_qubits}, {j_bin_qubit_spec=}")
            bin_probabilities = self.prob_fn(j_bin_qubit_spec, **kwargs)
            if np.sum(bin_probabilities) >= self.epsilon:
                bin = Bin(j_bin_qubit_spec, bin_probabilities)
                self.push(bin)

        self._recurse(
            recursion_level + 1,
            max_recursion,
            **kwargs,
        )

    def probabilities(self, full_states: np.ndarray | None = None) -> np.ndarray:
        if full_states is None:
            warnings.warn(
                "Generating all 2^num_qubits states. This may be memory intensive.",
                stacklevel=2,
            )
            full_states = np.arange(2**self.num_qubits, dtype="int64")

        probabilities = np.zeros_like(full_states, dtype="float32")
        for bin in self.bins:
            probabilities += unmerge_prob_vector(
                bin.probabilities, bin.qubit_spec, full_states=full_states
            )
        return probabilities

    def plot(
        self,
        plot_bins: bool = False,
        prob_mass_threshold: float = 0.9,
        max_bars: int = 20,
        ax: plt.Axes | None = None,
        full_states: np.ndarray | None = None,
    ):
        if plot_bins:
            mass_sum = 0
            x = []
            y = []
            for j, bin in enumerate(heapq.nsmallest(len(self.bins), self.bins)):
                x.append(bin.qubit_spec)
                y.append(bin.probability_mass)

                mass_sum += bin.probability_mass
                if mass_sum >= prob_mass_threshold or j > max_bars - 1:
                    break
        else:
            y = self.probabilities(full_states=full_states)
            x = np.arange(len(y)) if full_states is None else full_states

        if ax:
            ax.bar(x, y)
            ax.set_xlabel("Bitstring index")
            ax.set_ylabel("Probability")
            ax.set_title(f"Recursion Level {self.recursion_level}")
        else:
            plt.figure(figsize=(12, 4))
            plt.bar(x, y)
            plt.xlabel("Bitstring index")
            plt.ylabel("Probability")
            plt.title(f"Recursion Level {self.recursion_level}")
            plt.show()
