""" 
    ------------------------- Mutator for Mousify --------------------------
    Takes a map and a selects a sequence among the possible sequences of the
    map. Then the mutator decides whether or not to accept this sequence as 
    the next sequence. It recreates the map from that point and starts over.
    ------------------------------------------------------------------------
"""

import random
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import numpy.typing as npt

from modules.species_scoring import Discriminator


class Mutator(ABC):
    """
    ## Mutator base class.
    A mutator takes a discriminator object, map object and sequence as input.
    The sequence is the basis on which mutations are performed and the mutations
    selected are determined by the map and some sampling method determined by the
    mutator child class. The discriminator is used to calculate the humanization
    scores of the proposed sequences to compare its score to the inital score.

    ### Args:
        discriminator {Discriminator} -- Discriminator object (See Mousify.discriminators) \n
        map {Map} -- Map object (See Mousify.maps). Needs to be of same dimension as the sequence.\n
        sequence {str} -- Protein sequence to be humanized. Needs to be of same dimension as map.\n
        pad_size {int} -- pad_size used for the encoding scheme in the discriminator. \n
    """

    def __init__(
        self,
        discriminator: Discriminator,
        sequence: str,
        pad_size: int,
        seed: Optional[int] = None,
    ):
        self.discriminator = discriminator
        self.sequence = sequence
        self.pad_size = pad_size
        self.proposed_sequence = None
        self.proposed_score = 0
        self.transition_matrix: Union[npt.ArrayLike, None] = None
        self.transition_id: Union[tuple, None] = None
        self.proposed_sequence: Union[str, None] = None

        # Set the seed
        if seed is not None:
            self.seed = seed
            random.seed(a=self.seed)

        # Get initial score to compare against
        self.configure_discriminator(pad_size=self.pad_size, sequence=self.sequence)
        self.initial_score = self.discriminator.calculate_score()
        self.sequence_registry = [(sequence, "query", "query", self.initial_score)]

        # Dictionary to translate the amino acid single letter code to the internal index system
        self.amino_acid_dict = {
            "A": 0,
            "R": 1,
            "N": 2,
            "D": 3,
            "C": 4,
            "E": 5,
            "Q": 6,
            "G": 7,
            "H": 8,
            "I": 9,
            "L": 10,
            "K": 11,
            "M": 12,
            "F": 13,
            "P": 14,
            "S": 15,
            "T": 16,
            "W": 17,
            "Y": 18,
            "V": 19,
        }
        self.inverted_amino_acids = {
            0: "A",
            1: "R",
            2: "N",
            3: "D",
            4: "C",
            5: "E",
            6: "Q",
            7: "G",
            8: "H",
            9: "I",
            10: "L",
            11: "K",
            12: "M",
            13: "F",
            14: "P",
            15: "S",
            16: "T",
            17: "W",
            18: "Y",
            19: "V",
        }

        self._mutator_type = None

    @abstractmethod
    def propose(self):
        """
        ## Proposes a new sequence based on the map provided.
        Decides which mutations should be proposed to the get_decision method based on the
        probability map calculated in the map module.
        """

    @abstractmethod
    def set_score(self):
        """
        ## Gets the score of the proposed sequence.
        Uses the discriminator to calculate the score of a proposed sequence.
        This is used by the get_decision method on whether the sequence is accepted,
        rejected or finalized.
        """

    @abstractmethod
    def set_decision(self):
        """
        ## Makes a decision on whether a sequence is accepted, rejected or finalized.
        Uses a function specified to take a decision whether a sequence is accepted,
        rejected or presented as the final humanized sequence.
        """

    def configure_discriminator(self, pad_size: int, sequence: str) -> None:
        """
        ## Configures the discriminator object.
        Loads the sequence into the discriminator and encodes the sequence.
        ### Args:
                pad_size {int} -- Size of padding (depends on the discriminator used)
        """
        self.discriminator.load_query(sequence)
        self.discriminator.encode(mode="seq", pad_size=pad_size)

    def set_transition_matrix(
        self,
        allow: Union[None, Literal["CDR", "cdr"], list[int]] = None,
        disallow: Union[None, list[int]] = None,
    ):
        """
        ## Sets the transition matrix as an instance attribute.

        """
        map_dictionary = self.discriminator.create_map(allow=allow, disallow=disallow)
        self.transition_matrix = map_dictionary["transition_matrix"]

    @abstractmethod
    def run_humanization(
        self,
        allow: Union[None, Literal["CDR", "cdr"], list[int]],
        disallow: Union[None, list[int]],
        max_iterations: int,
    ):
        """
        ## Runs humanization for a specific mutator
        """

    @abstractmethod
    def run_warmup(self):
        """
        ## Runs humanization for a specific mutator
        """

    @abstractmethod
    def update_registry(self, mode: Literal["accepted", "rejected"]):
        """
        ## Updates the sequence registry
        """
