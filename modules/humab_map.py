""" 
    ------------------------- Map for Mousify --------------------------
    Takes a sequence or an array of sequences and creates a probability
    map indicating which sequences should be mutated.
    --------------------------- Hu-mAb Map -----------------------------
    Every residues probability is calculated by the humanization score
    increase it achieves which is then normalized. Any arbitrary discriminator
    can be used to calculate the score.
"""

from copy import copy
from typing import Literal, Union, Optional
from multiprocessing import Pool
from functools import partial

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from modules.map import Map
from modules.species_scoring import Discriminator
from modules.immune_vae_scoring import ModelLoadConfig


class HumabMap(Map):
    """
    ## Creates a map by calculating the humanization score from the query to all possible mutants.
    Creates a list of all possible single mutants from the query sequence.
    Uses the discriminator provided to calculate the humanization score of each sequence in
    the list and reshapes the list into an array where the columns are the sequence position
    and the rows are the mutants at that position.
    ### Args:
        \tsequence {str} -- String of the amino acid sequence \n
        \tdiscriminator {Discriminator} -- Humanization scoring function to use for Hu-mAb Map
        \tnumbering {str} -- What numbering system to use to define the CDRs.
         (Options: IMGT, Chothia, Kabath, Martin)\n
        \tcdr_boundaries {list[tuple[int,...]]} -- List of tuples containing the boundary
                indices of the CDR. \n
                Optional if none of the conventional CDR numbering schemes are appropriate.
    """

    def __init__(
        self,
        sequence: str,
        discriminator: Discriminator,
        pad_size: int,
        score_config: ModelLoadConfig,
        scheme: Literal["imgt", "chothia", "kabath", "aho"] = "imgt",
        cdr_boundaries: Optional[list[tuple[int, ...]]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            sequence=sequence,
            scheme=scheme,
            cdr_boundaries=cdr_boundaries,
            score_config=score_config,
        )
        self.discriminator = discriminator
        self.pad_size = pad_size
        self.n_jobs = n_jobs
        self.cdr: Union[None, list[int]] = None

    def create_map(
        self,
        allow: Union[None, Literal["CDR", "cdr"], list[int]] = None,
        disallow: Union[None, list[int]] = None,
    ) -> dict[str, npt.ArrayLike]:
        """
        ## Creates a Hu-mAb Map.
        If CDR is allowed, the score is calculated on every possible single mutant based
        on the score in the discriminator. Otherwise, the score is only calculated for
        allowed residues. The final score is not normalized!
        ### Args:
            \tallow {list[int] or 'CDR'} -- List of allowed indices of the sequence,
            or 'CDR' allows all residues \n
            \tdisallow {list[int]} -- List of disallowed indices of the sequence (e.g.Vernier Zones)
        ### Returns:
            \tdict[npt.ArrayLike, npt.ArrayLike] -- Dictionary containing the transition matrix
            and the coupling matrix. Keys: \n
                \t'transition_matrix': Contains the transition matrix
                \t'coupling_matrix': Contains NONE for the Uniform map
        """
        # Get all single mutants
        all_single_mutants = self.single_mutants()
        if self.n_jobs == 1:
            transition_matrix = self.get_transition_matrix(
                all_single_mutants=all_single_mutants,
                allow=allow,
                disallow=disallow,
            )
        else:
            transition_matrix = self.get_transition_matrix_multiprocess(
                all_single_mutants=all_single_mutants,
                allow=allow,
                disallow=disallow,
            )

        # Return the transition matrix
        return {
            "transition_matrix": transition_matrix,
            "coupling_matrix": np.array(None),
        }

    def get_transition_matrix(
        self,
        all_single_mutants: npt.ArrayLike,
        allow: Union[None, Literal["CDR", "cdr"], list[int]] = None,
        disallow: Union[None, list[int]] = None,
    ) -> npt.ArrayLike:
        """
        ## Calculates the transition matrix
        Goes through all possible next mutations and populates an Array with the scores of each
        mutation by using the provided discriminator.
        ### Args:
            \tall_single_mutants {Array} -- Array of all single mutants\n
            \tallow {Literal or list[int]} -- Optional. List of indices to mutate on top of the
                                              framework. Can choose the entire CDR by setting this
                                              argument to 'cdr' or one can provide a list of indices
                                              to allow.\n
            \tdisallow: {list[int]} -- Optional. List of indices to not mutate
        ### Returns:
            \tArray -- Nx20 array of discriminator scores for each possible mutant
        """
        # If CDR is allowed, everything is uniform
        if allow == "CDR" or allow == "cdr":
            self.cdr = None
        else:
            # Get the indices of the CDR and disallowed residues
            self.cdr = super()._find_cdr_indices(
                allow=allow,
                disallow=disallow,
                manual_cdr=self.cdr_boundaries,
                scheme=self.scheme,
            )

        # Iterate through all sequences to calculate score
        transition_matrix = np.zeros((20, len(self.sequence)))
        column = 0  # The column value also keeps track of the sequence indices
        for i, seq in tqdm(
            enumerate(all_single_mutants), total=len(all_single_mutants)
        ):

            # Once end of column is reached, start populating next column
            if i % 20 == 0 and i != 0:
                column += 1

            # If the index is in self.cdr, it is a disallowed residue and is set to 0
            if self.cdr is not None:
                if column in self.cdr:
                    transition_matrix[i % 20, column] = 0
                    continue

            # Configure the discriminator
            self.configure_discriminator(pad_size=self.pad_size, sequence=seq)
            # Calculate the score
            transition_matrix[i % 20, column] = self.discriminator.calculate_score(
                score_config=self.score_config
            )
        transition_matrix = self.normalize_map(transition_matrix)
        return transition_matrix

    def get_transition_matrix_singleprocess(
        self,
        sequence_list: tuple[int, npt.ArrayLike],
        cdr_indices: Union[npt.ArrayLike, None],
    ) -> npt.ArrayLike:
        """
        ## Processes one batch of sequences for transition matrix
        """
        current_index, sequence_list = sequence_list
        # Iterate through all sequences to calculate score
        transition_matrix = np.zeros(20)
        # Return empty vector if a disallowed index
        if cdr_indices is not None and current_index in cdr_indices:
            return transition_matrix
        for i, seq in enumerate(sequence_list):
            # Configure the discriminator
            self.configure_discriminator(pad_size=self.pad_size, sequence=seq)
            # Calculate the score
            transition_matrix[i] = self.discriminator.calculate_score(
                score_config=self.score_config
            )
        return transition_matrix

    def get_transition_matrix_multiprocess(
        self,
        all_single_mutants: npt.ArrayLike,
        allow: Union[None, Literal["CDR", "cdr"], list[int]] = None,
        disallow: Union[None, list[int]] = None,
    ) -> npt.ArrayLike:
        """
        ## Calculates transition matrix using multiprocess
        Goes through all possible next mutations and populates an Array with the scores of each
        mutation by using the provided discriminator.
        ### Args:
            \tall_single_mutants {Array} -- Array of all single mutants\n
            \tallow {Literal or list[int]} -- Optional. List of indices to mutate on top of the
                                              framework. Can choose the entire CDR by setting this
                                              argument to 'cdr' or one can provide a list of indices
                                              to allow.\n
            \tdisallow: {list[int]} -- Optional. List of indices to not mutate
        ### Returns:
            \tArray -- Nx20 array of discriminator scores for each possible mutant
        """
        if allow == "CDR" or allow == "cdr":
            self.cdr = None
        else:
            self.cdr = super()._find_cdr_indices(
                allow=allow,
                disallow=disallow,
                manual_cdr=self.cdr_boundaries,
                scheme=self.scheme,
            )
        split_sequence_array = np.array_split(
            all_single_mutants, len(all_single_mutants[0])
        )
        indexed_split_sequence_array = []
        for index, array in enumerate(split_sequence_array):
            indexed_split_sequence_array.append((index, array))

        partial_single_process = partial(
            self.get_transition_matrix_singleprocess,
            cdr_indices=self.cdr,
        )

        all_vectors = np.zeros((20, len(self.sequence)))
        with Pool(processes=self.n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(partial_single_process, indexed_split_sequence_array),
                    total=len(indexed_split_sequence_array),
                )
            )
            for index, vector in enumerate(results):
                all_vectors[:, index] = vector

            # for index in range(len(self.sequence)):
            #     if index in self.cdr:
            #         all_vectors[:, index] = np.zeros(20)
        all_vectors = self.normalize_map(np.array(all_vectors))
        return all_vectors

    @staticmethod
    def normalize_map(transition_matrix: npt.ArrayLike) -> npt.ArrayLike:
        """
        ## Normalizes the provided map.
        Divides each value by the sum of the map.
        ### Args:
                \ttransition_matrix {npt.ArrayLike} -- Transition matrix from create_map method.
        ### Returns:
                \tnpt.ArrayLike -- Normalized map
        """
        normalization = np.sum(transition_matrix)
        return np.divide(transition_matrix, normalization)

    def configure_discriminator(self, pad_size: int, sequence: str) -> None:
        """
        ## Configures the discriminator object.
        Loads the sequence into the discriminator and encodes the sequence.
        ### Args:
                pad_size {int} -- Size of padding (depends on the discriminator used)
        """
        self.discriminator.load_query(sequence)
        self.discriminator.encode(mode="seq", pad_size=pad_size)

    def single_mutants(self) -> None:
        """
        ## Creates a list of all possible single mutants from the query sequence.

        ### Updates:
                \tall_single_mutants {list[str]} -- List of all possible single mutants
        """

        all_single_mutants = []
        # Iterate through the residues of the sequence
        for i in range(len(self.sequence)):

            # Iterate through all possible amino acids
            for point in self.all_amino_acids_order:
                # Copy so as to not change the original
                mutant = copy(self.sequence)
                seq = mutant[:i] + point + mutant[i + 1 :]
                # This is the fastest way to replace a character in a string
                all_single_mutants.append(seq)

        return np.array(all_single_mutants)

    def set_sequence(self, sequence: str) -> None:
        """
        ## Sets the sequence attribute to a new sequence.
        ### Args:
                \tsequence {str} -- String of the amino acid sequence

        ### Updates:
                \tself.sequence {str} -- String of the amino acid sequence
        """
        self.sequence = sequence
