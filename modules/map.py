""" 
    ------------------------- Map for Mousify --------------------------
    Takes a sequence or an array of sequences and creates a probability
    map indicating which sequences should be mutated.
    --------------------------------------------------------------------
"""

import re

from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import numpy.typing as npt
import numpy as np

from abnumber import Chain, Position
from modules.immune_vae_scoring import ModelLoadConfig


class Map(ABC):
    """
    ## Abstract class for all maps.
    A map takes a length N amino acid sequence and returns a (Nx20) matrix
    that defines the probability that the residue at position N(i) is mutated
    to another residue.
    """

    def __init__(
        self,
        sequence: str,
        score_config: ModelLoadConfig,
        scheme: Literal["imgt", "chothia", "kabath", "aho"] = "imgt",
        cdr_boundaries: Union[None, list[tuple[int, ...]]] = None,
        amino_acid_order: Optional[npt.ArrayLike] = None,
    ):
        self.sequence = sequence
        self.score_config = score_config
        # If CDR boundaries have been provided, no need to set the scheme
        if cdr_boundaries:
            self.scheme = None
        else:
            self.scheme = scheme

        self.cdr_boundaries = cdr_boundaries

        # List all amino acids for reference
        if amino_acid_order is not None:
            self._check_amino_acid_order(amino_acid_order)
            self.all_amino_acids_order = amino_acid_order
        else:
            self.all_amino_acids_order = (
                "A",
                "R",
                "N",
                "D",
                "C",
                "E",
                "Q",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
            )

    @abstractmethod
    def create_map(
        self,
        allow: Union[None, Literal["CDR", "cdr"], list[int]] = None,
        disallow: Union[None, list[int]] = None,
    ) -> dict[str, npt.ArrayLike]:
        """
        ## Method that creates the map.
        Uses self.sequence as template.
        """

    def _find_cdr_indices(
        self,
        scheme: Union[Literal["imgt", "chothia", "kabath", "aho"], None],
        allow: Union[None, list[int]] = None,
        disallow: Union[None, list[int]] = None,
        manual_cdr: Union[None, list[tuple[int, ...]]] = None,
    ) -> list[int]:
        """
        ## Finds the CDR of the sequence and adds/removes other residues for the Map.
        Uses abnumber to find the CDRs with imgt. Optional to switch to chothia, kabath or aho.
        One can also provide custom CDR indices using the custom_CDR argument.
        After establishing the CDR indices, the return value is packaged into a list of
        tuples containing the boundaries of non-mutatable regions for the Map.
        ### Args:
            \tallow {list[int]} -- Marks which indices should always be included \n
            \tdisallow {list[int]} -- Marks which indices should never be included\n
            \tscheme {str} -- Type of scheme system to be used to find CDR and FWR\n
            \tcustom_CDR {list[tuple[int,...]]} -- Marks the CDR index boundaries if a manual CDR
            needs to be implemented\n
        ### Returns:
            \tlist[int] -- List of indices containing the disallowed regions for the Map.
        """
        # Use abnumber numbering methods
        if scheme is not None:
            chain = Chain(self.sequence, scheme=self.scheme)
            disallowed_regions_abnumber = self._get_cdr_indices_from_abnumber(
                chain=chain
            )
            # The disallowed regions in abnumber are aligned to the imgt numbering
            disallowed_regions = []
            chain_numbering = [pos for pos, aa in chain]
            for index, numbering in enumerate(chain_numbering):
                # convert from position to str and drop the prefix
                position = str(numbering)[1:]
                # Convert to int to compare to the output from abnumber
                position = re.sub("[A-Z]", "", position)
                if int(position) in disallowed_regions_abnumber:
                    disallowed_regions.append(index)

        elif manual_cdr is not None:

            disallowed_regions = []
            # Extract indices from boundaries
            for boundary in manual_cdr:
                disallowed_regions.extend(range(boundary[0], boundary[1] + 1, 1))

        # Remove allowed residues
        if allow:
            disallowed_regions = [
                index for index in disallowed_regions if index not in allow
            ]

        # Add disallowed residues
        if disallow:
            disallowed_regions.extend(disallow)
            disallowed_regions.sort()
        return disallowed_regions

    def _get_cdr_indices_from_abnumber(
        self,
        chain: Chain,
        attribute: Union[Literal["cdr1", "cdr2", "cdr3"], None] = None,
    ) -> list:
        """
        ## Uses Abnumber to get cdr positions
        """
        attr_dict = {
            "cdr1": chain.cdr1_dict,
            "cdr2": chain.cdr2_dict,
            "cdr3": chain.cdr3_dict,
        }

        # If only one cdr wants to be selected as disallowed
        if attribute is not None:
            numbered_list = list(attr_dict[attribute].keys())

        else:
            # Convert the cdr_dict to a abnumber.Position object
            numbered_list: list[Position] = []
            for attr in list(attr_dict.values()):
                numbered_list += list(attr.keys())
            # Convert abnumber.Position entry to str
            numbered_list = [item.format() for item in numbered_list]

        # Define the prefix
        prefix = numbered_list[0][0]
        assert prefix in [
            "H",
            "K",
            "L",
        ], f"Prefix: {prefix} is not an appropriate prefix. Must be H, K or L"
        # In case a subindex of IMGT is found, we need to shif the indices by 1 for each
        shift = 0
        cdr_indices = []
        # Extract indices from numbered_list
        for index in numbered_list:
            index = index.replace(prefix, "")
            # Find subindex, remove the letter, append to indices and shift sequence by 1
            if re.search("[A-Z]", index):
                index = re.sub("[A-Z]", "", index)
                cdr_indices.append(int(index) + shift)
                shift += 1
                continue

            # Abnumber does not 0-index
            cdr_indices.append(int(index) + shift)
        return cdr_indices

    @staticmethod
    def _check_amino_acid_order(amino_acid_order: npt.ArrayLike):
        """
        ## Runs Assert statements to check if the provided iterable contains all amino acids
        """
        amino_acids = (
            "A",
            "R",
            "N",
            "D",
            "C",
            "E",
            "Q",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        )
        # Check if all amino acids are represented
        for aa in amino_acids:
            assert (
                aa in amino_acid_order
            ), f"Amino acid {aa} not found in provided order!"
        # Check if there are no unknown amino acids
        for aa in amino_acid_order:
            assert aa in amino_acids, f"{aa} is an unknown amino acid!"

    @staticmethod
    def filter_transition_matrix(
        transition_dictionary: dict[
            Literal["transition_matrix", "coupling_matrix"], npt.ArrayLike
        ],
        quantile: float,
    ) -> dict[Literal["transition_matrix", "coupling_matrix"], npt.ArrayLike]:
        """
        ## Filters the transition matrix values (high-pass)
        The filter takes in a quantile as a parameter and filters out all
        transition probabilities below the quantile.
        ### Args:
            \ttransition_dictionary {dict[Literal, ArrayLike]} -- Transition matrix
            dictionary from create_map method \n
            \tquantile {float} -- Quantile below which to filter out transition
            probabilities. Has to be in [0, 1] \n
        ### Returns:
            \t {dict[Literal, ArrayLike]} -- Transition matrix dictionary in the
            same format as create_map method
        """
        assert (
            quantile >= 0 and quantile <= 1
        ), f"quantile {quantile} needs to be between 0 and 1"

        # Get quantile
        high_pass_filter = np.quantile(
            a=transition_dictionary["transition_matrix"], q=quantile
        )

        # Set all values below filter to 0
        transition_matrix = transition_dictionary["transition_matrix"]
        transition_matrix[transition_matrix < high_pass_filter] = 0

        # Re-normalize
        norm = np.sum(transition_matrix)
        transition_matrix = transition_matrix / norm

        return {
            "transition_matrix": transition_matrix,
            "coupling_matrix": transition_dictionary["coupling_matrix"],
        }
