from typing import Callable
import logging
import heapq
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
        self.probability_mass = np.sum(probabilities)

    def __str__(self):
        return f"Bin({self.qubit_spec}, {self.probability_mass:.3f})"

    def __lt__(self, other):
        """
        This method is used to compare two `Bin` objects in the min-heap, and
        is thus used to decide whether to prioritize *this* bin over *other*.

        Since we only ever care about popping bins with merged qubits ("M" in
        qubit_spec), we prioritize this bin only if it has an "M", and then
        compare the probability mass of the two bins to prioritize the one with
        the larger probability mass.
        """
        return "M" in self.qubit_spec and self.probability_mass > other.probability_mass


class DynamicDefinition:
    """
    DynamicDefinition is a class that implements the dynamic definition
    algorithm for quantum probability distribution reconstruction.
    It recursively zooms-in on qubits (initially "merged") that have a high
    probability mass.
    """

    def __init__(
        self,
        num_qubits: int,
        capacity: int,
        prob_fn: Callable,
        epsilon: float = 1e-4,
        preallocated: np.ndarray | None = None,
    ):
        self.num_qubits = num_qubits
        self.capacity = capacity
        self.prob_fn = prob_fn
        self.preallocated = (
            preallocated
            if preallocated is not None
            else np.zeros(2**num_qubits, dtype="float32")
        )

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

    def run(self, max_recursion: int = 10) -> np.ndarray:
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
        initial_probabilities = self.prob_fn(initial_qubit_spec)
        initial_bin = Bin(initial_qubit_spec, initial_probabilities)

        self.push(initial_bin)
        if self.capacity < self.num_qubits:
            self._recurse(recursion_level=1, max_recursion=max_recursion)

        for bin in self.bins:
            unmerge_prob_vector(bin.probabilities, bin.qubit_spec, self.preallocated)
        return self.preallocated

    def _recurse(self, recursion_level: int, max_recursion: int = 10):
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
        for j in range(2**self.capacity):
            bin_qubit_spec = qubit_spec

            # For this bin, mark `capacity` (possibly fewer) merged qubits as active
            bin_qubit_spec = bin_qubit_spec.replace("M", "A", self.capacity)

            # Replace all active qubits with the binary representation
            # of j - these become the "zoomed-in" bits.
            j_str = f"{j:0{self.capacity}b}"  # `capacity` length bit-string
            for j_char in j_str:
                bin_qubit_spec = bin_qubit_spec.replace("A", j_char, 1)

            bin_probabilities = self.prob_fn(bin_qubit_spec)
            if np.sum(bin_probabilities) >= self.epsilon:
                bin = Bin(bin_qubit_spec, bin_probabilities)
                self.push(bin)

        self._recurse(recursion_level + 1, max_recursion)

    def plot(self):
        probabilities = self.probabilities

        x = np.arange(len(probabilities))
        plt.figure(figsize=(12, 4))
        plt.bar(x, probabilities)
        plt.xlabel("Bitstring index")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.title(f"Recursion Level {self.recursion_level}")
        plt.show()
