""" 
    ------------------------- Mutator for Mousify --------------------------
    Takes a map and a selects a sequence among the possible sequences of the
    map. Then the mutator decides whether or not to accept this sequence as 
    the next sequence. It recreates the map from that point and starts over.
    ------------------------- Simulated Annealing --------------------------
    Uses a Simulated Annealing Markov Model to sample the map. Decision is
    made based on the provided discriminator. The step is accepted with a 
    probability calculated by the Kirkpatrick Acceptance Function. The minimum 
    value between the Kirkpatrick Acceptance Function and 1 is the final 
    acceptance probability.
"""
from __future__ import annotations
from typing import Literal, Optional, Union
from tqdm import tqdm
from pathlib import Path

import random
import numpy as np
import numpy.typing as npt
import logging

from Mousify.maps.map import Map
from Mousify.maps.uniform_map import UniformMap
from Mousify.maps.sapiens_map import SapiensMap
from Mousify.mutations.mutator import Mutator
from Mousify.discriminators.discriminator import Discriminator
from Mousify.discriminators.serial_discriminator import SerialDiscriminator
from Mousify.discriminators.oasis_discriminator import OASisDiscriminator
from Mousify.encodings.onehotencoder import OneHot

from biophi.humanization.web.tasks import HumannessTaskError


class SimulatedAnnealing(Mutator):
    """
    ## Simulated Annealing Markov Chain implemented as a mutator.
    Uses the Kirkpatrick function to decide whether or not to accept a proposed sequence.
    ### Args:
        \tdiscriminator {Discriminator} -- Function that calculates the score of the sequences \n
        \tmap {Map} -- Equivalent to the proposal distribution \n
        \tsequence {str} -- Antibody sequence \n
        \tpad_size {int} -- Pad size for the encoding (Used in the Discriminator and Map) \n
        \ttemperature {float} -- Temperature used for the walker (Can be changed with an annealing scheduler) \n
        \tseed {int} -- Seeding number for the random process
    """

    def __init__(
        self,
        discriminator: Discriminator,
        map: Map,
        sequence: str,
        pad_size: int,
        temperature: Optional[float] = None,
        temperature_range: Optional[list[float]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(discriminator, map, sequence, pad_size, seed)
        self.transition_matrix: Union[npt.ArrayLike, None] = None
        self._mutator_type = "MCMC"
        if temperature is None:
            assert (
                temperature_range is not None
            ), "Have to provide either temperature or a list of temperatures to run simulated annealing on"
        self.temperature = temperature
        self.temperature_range = temperature_range

    def propose(self):
        """
        ## Proposes a new sequence based on the map provided.
        ### Updates:
                \tself.proposed_sequence {str} -- Amino acid sequence of the proposed sequence\n
                \tself.transition_probability {float} -- Probability of the transition\n
                \tself.transition_ID {tuple[int, str, str]} -- Transition ID says at what position the\n
                                                                transition happened, what the original\n
                                                                amino acid was and what the new one is
        """

        assert (
            self.transition_matrix is not None
        ), f"Transition matrix has not been set yet. Use set_transition_matrix method before running this method. Current type of transition matrix: {type(self.transition_matrix)}"

        # Get array to figure out locations of residues in transition matrix
        location_array = self._location_array(sequence_length=len(self.sequence))

        # Sample a mutation by using the transition matrix as weights
        proposed_mutation = random.choices(
            population=location_array, weights=self.transition_matrix.flatten()
        )

        # Residue number is at proposed mutation [0][1]/Inverted amino acids translates the amino acid number into sinle letter code
        self.transition_ID = (
            proposed_mutation[0][1],
            self.sequence[proposed_mutation[0][1]],
            self.inverted_amino_acids[proposed_mutation[0][0]],
        )

        # Make sure that it doesn't propose going to the same residue
        if self.transition_ID[1] == self.transition_ID[2]:
            self.propose()

        # Transition probability is the value of the transition matrix at the index of proposed mutation
        self.transition_probability = self.transition_matrix[
            proposed_mutation[0][0], proposed_mutation[0][1]
        ]

        # Setting the proposed sequence
        self.proposed_sequence = (
            self.sequence[: self.transition_ID[0]]
            + self.transition_ID[2]
            + self.sequence[self.transition_ID[0] + 1 :]
        )

    def set_score(self):
        """
        ## Gets the score of the proposed sequence.
        ### Updates:
                \tself.proposed_score {float} -- Score of the proposed sequence
        """
        assert (
            self.proposed_sequence is not None
        ), f"No proposed sequence found. Run propose method first. Current type of proposed_sequence is {type(self.proposed_sequence)}. Needs to be str."

        self.configure_discriminator(
            pad_size=self.pad_size, sequence=self.proposed_sequence
        )
        try:
            self.proposed_score = self.discriminator.calculate_score()
        except HumannessTaskError:
            self.proposed_score = -1

    def set_decision(self, track_rejected: bool = False) -> bool:
        """
        ## Makes a decision on whether a sequence is accepted, rejected or finalized.
        Before running this method, the following methods need to be run to work properly:\n
            \t-set_transition_matrix\n
            \t-propose\n
            \t-set_score\n
        This method first generates a random number between 0 and 1. Then gets the metropolis ratio
        for the proposed seuqence. If the metropolis ratio is >= 1, then the step is always accepted
        and self.sequence = self.proposed_sequence. The sequence and the proposed sequence is also added
        to the sequence registry.
        ### Args:
                \tseed {Optional[int]} -- Seed number for testing purposes\n
                \tprior {bool} -- Wether or not to calculate the prior in the metropolis ratio.\n
                \ttrack_rejected {bool} -- Wether or not to keep track of rejected sequences in the sequence registry.\n
        ### Updates:
                \tself.sequence -- Can either be the same or set to self.proposed_sequence\n
                \tself.sequence_regisrty -- Registry of all sequences so far accepted
        """

        acceptance_value = self.kirkpatrick_function()
        random_value = random.uniform(0, 1)

        if random_value <= acceptance_value:
            # If accepted update the registry
            self.update_registry(mode="accepted")
            return True

        elif track_rejected:
            # Keep track of rejected sequences
            self.update_registry(mode="rejected")

        return False

    # NEED TO SET ALLOW AND DISALLOW AS ATTRIBUTES IN THE MAP MODULE OTHERWISE I ALWAYS HAVE TO PASS THIS AS AN ARGUMENT
    def kirkpatrick_function(self) -> float:
        """
        ## Returns the acceptance value of the kirkpatrick function of the proposed vs original sequence.
        """
        # If HumannessTaskError was caught, set return to -1
        if self.proposed_score == -1:
            return -1
        # Exponent Calculation
        if self.proposed_score > self.initial_score:
            return 1
        difference = self.initial_score - self.proposed_score
        exponent = -(difference / self.temperature)
        return np.exp(exponent)

    def update_registry(self, mode: Literal["accepted", "rejected", "final"]):
        """
        ## Updates the sequence registry when a new sequence has been accepted.
        Also updates the sequence attribute.
        """
        if mode == "accepted":
            self.sequence = self.proposed_sequence
            self.initial_score = self.proposed_score
        self.sequence_registry.append(
            (self.proposed_sequence, mode, self.transition_ID, self.proposed_score)
        )

    @staticmethod
    def _location_array(sequence_length: int) -> npt.ArrayLike:
        """
        ## Returns the location array of a map.
        The location array is just a flattened version of the row and columns of the map
        in the format [[row, column]...]. This way we know what residue position (column)
        was modified to which amino acid (row).
        ### Args:
                \tarray {ArrayLike} -- transition matrix of the sequence

        ### Returns:
                \t ArrayLike -- An array containing the columns and rows of the map positions
        """
        location_array = []
        # Starts at -1 because the for loop immediately adds 1
        row = -1
        column = 0
        for i in range(sequence_length * 20):
            if not i % sequence_length:
                row += 1
                column = 0
            location_array.append([row, column])
            column += 1

        return location_array

    def configure_humanization_run(
        self,
        sequence: str,
        initial_temperature: float,
        discriminator: Discriminator,
        maps: Map,
        model_path: Optional[Path] = None,
    ) -> SimulatedAnnealing:
        """
        ## Configures object for one run of humanization
        """
        return SimulatedAnnealing(
            discriminator=discriminator,
            map=maps,
            pad_size=131,
            sequence=sequence,
            temperature=initial_temperature,
        )

    def run_humanization(
        self,
        scheduler_threshold: int,
        library_size: int,
        max_iterations: int,
        sequence: str,
        discriminator: Discriminator,
        maps: Map,
        allow: Union[None, Literal["CDR", "cdr"], list[int]],
        disallow: Union[None, list[int]],
        model_path: Optional[Path],
        chain_name: str,
    ):
        """
        ## Runs the humanization for Simulated Annealing
        """
        # Set run temperatures
        if self.temperature_range is not None:
            temperature_generator = np.nditer(self.temperature_range)
        else:
            # Putting this into a generator makes things down the line a bit easier
            temperature_generator = np.nditer([self.temperature])

        run_threshold = 0
        # Putting it in try/except block because of key error that keeps occuring
        try:
            self = self.configure_humanization_run(
                sequence=sequence,
                initial_temperature=next(temperature_generator),
                discriminator=discriminator,
                maps=maps,
                model_path=model_path,
            )
            for i in tqdm(range(max_iterations), total=library_size):
                # We only want to call the annealing scheduler if a temperature range is given
                if (
                    self.annealing_scheduler(
                        index=i,
                        scheduler_threshold=scheduler_threshold,
                        run_threshold=run_threshold,
                    )
                    and self.temperature_range is not None
                ):
                    self.temperature = next(temperature_generator)
                    run_threshold += max_iterations / (len(self.temperature_range) - 1)

                self.set_transition_matrix(allow=allow, disallow=disallow)
                self.propose()
                self.set_score()
                self.set_decision(track_rejected=False)

                if len(self.sequence_registry) > library_size:
                    break

            key = chain_name
            return {key: self.sequence_registry}
        except KeyError:
            logging.exception("run_humanization(%r) failed" % (chain_name,))

    def run_warmup(
        self,
        warmup: int,
        sequence: str,
        discriminator: Discriminator,
        maps: Map,
        model_path: Optional[Path],
        chain_name: str,
    ):
        """
        ## Runs the warmup for Simulated Annealing
        """
        # Set warmup temperatures
        if self.temperature is None:
            warmup_temperature = self.temperature_range[0]
        else:
            warmup_temperature = self.temperature
        # Putting it in try/except block because of key error that keeps occuring
        try:
            self = self.configure_humanization_run(
                sequence=sequence,
                initial_temperature=warmup_temperature,
                discriminator=discriminator,
                maps=maps,
                model_path=model_path,
            )
            tqdm.write("Warmup...")
            for _ in tqdm(range(warmup), total=warmup):
                self.set_transition_matrix()
                self.propose()
                self.set_score()
                self.set_decision(track_rejected=False)

            tqdm.write(
                f"Number of accepted warmup sequences: {len(self.sequence_registry)}"
            )
            key = chain_name
            return {key: self.sequence_registry}
        except KeyError:
            logging.exception("run_humanization(%r) failed" % (chain_name,))

    @staticmethod
    def annealing_scheduler(
        index: int, scheduler_threshold: int, run_threshold: int
    ) -> bool:
        """Returns true if the temperature needs to be changed"""
        if index > (scheduler_threshold + run_threshold):
            return True
        else:
            return False


if __name__ == "__main__":
    sequence = "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAVYYCARYYDDHYCLDYWGQGTTLTVSSAKTTAPSVYPLA"
    new_sequence = "QVQLQQPGAELVLPGASVKLSCKASGYTFTNYWMHWVKQRPGHGLEWIGEIDPFDTYIKINQKFKGKSTLTVDTSSSTAYMQLSSLTSEDSAVYYCARPDSSGYPVYFDYWGQGTTLTVSS"
    disc = OASisDiscriminator(
        sequence=sequence,
        chain_type="heavy",
        min_fraction_subjects=0.90,
        score_type="OASis Identity",
    )
    disc.load_model(path="../data/OASis_9mers_v1.db")
    umap = SapiensMap(sequence=sequence)
    A = SimulatedAnnealing(
        discriminator=disc,
        map=umap,
        pad_size=131,
        sequence=sequence,
        seed=None,
        temperature=0.01,
    )
    # Stand-in for main function for now
    for i in tqdm(range(100)):
        A.set_transition_matrix()
        A.propose()
        A.set_score()
        A.set_decision(track_rejected=False)

    print(A.sequence_registry, len(A.sequence_registry))
