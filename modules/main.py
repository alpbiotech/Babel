""" 
    ------------------------- Mutator for Mousify --------------------------
    Takes a map and a selects a sequence among the possible sequences of the
    map. Then the mutator decides whether or not to accept this sequence as 
    the next sequence. It recreates the map from that point and starts over.
    ------------------------- Greedy Walk Mutator --------------------------
    Uses a greedy walk to decide on mutations. A greedy walk mutator always
    chooses the top N mutations that the Map object suggests regardless of
    the change in score of the discriminator. Unless the map takes score
    change into account. This mutator can either be run until a set number 
    of iterations (like Biophi) or until a set score is reached (like HumAb).
"""

from typing import Union, Literal, Optional
from pathlib import Path
from math import isclose
from tqdm import tqdm

import numpy as np
import numpy.typing as npt

from abnumber import Chain, ChainParseError
from modules.mutator_abc import Mutator
from modules.species_scoring import Discriminator
from modules.immune_vae_scoring import AbVAE, ModelLoadConfig
from modules.bytenet_vae_295k import ProtVAE
from modules.onehotencoder import OneHot
from modules.map import Map
from modules.humab_map import HumabMap


class GreedyWalk(Mutator):
    """
    ## Greedy Walk Mutator Class
    Uses a greedy walk to decide on mutations. A greedy walk mutator always
    chooses the top N mutations that the Map object suggests regardless of
    the change in score of the discriminator. Unless the map takes score
    change into account. This mutator can either be run until a set number
    of iterations (like Biophi) or until a set score is reached (like HumAb).
    ### Args:
    """

    def __init__(
        self,
        discriminator: Discriminator,
        mapping: Map,
        sequence: str,
        pad_size: int,
        score_config: ModelLoadConfig,
        seed: int | None = None,
        multishot: Union[int, Literal["all"]] = 1,
        n_rounds: Optional[int] = None,
        score_threshold: Optional[float] = None,
        fully_greedy: bool = True,
    ):
        super().__init__(
            discriminator=discriminator,
            mapping=mapping,
            sequence=sequence,
            pad_size=pad_size,
            seed=seed,
            score_config=score_config,
        )

        self.multishot = multishot
        if isinstance(self.multishot, str):
            assert (
                self.multishot == "all"
            ), "'multishot' argument needs to be an integer or 'all'"
        elif isinstance(self.multishot, int):
            assert (
                self.multishot > 0
            ), "'multishot' argument has to be larger than 0 or 'all'"
            assert self.multishot < len(
                self.sequence
            ), "'multishot' argument has to be smaller than the sequence lenght"
        else:
            raise TypeError("'multishot' argument needs to be an integer or 'all'")

        if n_rounds is not None:
            assert (
                score_threshold is None
            ), "Cannot set score_threshold and number of rounds as end condition simultaneously! \
            One of them needs to be None."

        if score_threshold is not None:
            assert (
                n_rounds is None
            ), "Cannot set score_threshold and number of rounds as end condition simultaneously! \
            One of them needs to be None."

        self.n_rounds = n_rounds
        self.score_threshold = score_threshold
        self.transition_matrix: Optional[npt.ArrayLike] = None
        self._mutator_type = "MLE"
        self.fully_greedy = fully_greedy
        self._cycle_counter = 0

    def propose(self):
        """
        ## Proposes a sequence based on the map provided
        Determines first what method to run based on the multishot attribute.
        ### Updates:
        self.proposed_sequence {str} -- Amino acid sequence of the proposed sequence\n
        self.transition_ID {tuple[int, str, str]} -- Transition ID says at what position the\n
                                                        transition happened, what the original\n
                                                        amino acid was and what the new one is
        """
        assert (
            self.transition_matrix is not None
        ), f"Transition matrix has not been set yet. \
        Use set_transition_matrix method before running this method. \
        Current type of transition matrix: {type(self.transition_matrix)}"

        # Factory to run the correct method based on multishot attribute
        propose_factory = {
            1: self.propose_singleshot,
            2: self.propose_multishot,
            "all": self.propose_allshot,
        }
        # Multishot > 1 is all in the same bin
        if isinstance(self.multishot, int):
            if self.multishot > 1:
                propose_factory[2]()
            else:
                propose_factory[self.multishot]()
        else:
            propose_factory[self.multishot]()

    def propose_singleshot(
        self,
    ):
        """
        ## Propose method called when only one step is taken at a time
        """
        # Propose mutation
        proposed_index = np.unravel_index(
            np.argmax(self.transition_matrix, axis=None), self.transition_matrix.shape
        )
        # Get the residue at that position in the currently stored sequence
        current_residue = self.sequence[proposed_index[1]]
        proposed_residue_change = self.inverted_amino_acids[proposed_index[0]]
        # Rerun if proposed residue is the current residue
        if current_residue == proposed_residue_change:
            self.transition_matrix[proposed_index[0], proposed_index[1]] = 0
            self.propose_singleshot()
        else:
            self.transition_id = (
                proposed_index[1],
                self.sequence[proposed_index[1]],
                self.inverted_amino_acids[proposed_index[0]],
            )
            self.proposed_sequence = (
                self.sequence[: self.transition_id[0]]
                + self.transition_id[2]
                + self.sequence[self.transition_id[0] + 1 :]
            )

    def propose_multishot(
        self,
    ):
        """
        ## Propose method called when N steps are taken at a time
        """
        # Not clear what is happening here, but thanks stackoverflow
        proposed_indices = np.unravel_index(
            np.argpartition(self.transition_matrix.ravel(), -self.multishot, axis=0)[
                -self.multishot :
            ],
            self.transition_matrix.shape,
        )
        proposed_indices = np.column_stack((proposed_indices[0], proposed_indices[1]))

        # Make sure at least one of them proposes something new
        new_residue_counter = 0
        # Go through proposals and compare to original sequence
        for item in proposed_indices:
            current_residue = self.sequence[item[1]]
            proposed_residue = self.inverted_amino_acids[item[0]]
            # Add to count if a new sequence was proposed
            if current_residue != proposed_residue:
                new_residue_counter += 1
            # Set their probability to 0 for the next round in case nothing new is found
            else:
                self.transition_matrix[item[0], item[1]] = 0

        # If nothing new was proposed, we try again
        if new_residue_counter < self.multishot:
            self.propose_multishot()
        else:
            # Extract information for transition_ID
            position_list = []
            current_residue_list = []
            proposed_residue_list = []
            self.proposed_sequence = self.sequence
            for item in proposed_indices:
                position_list.append(item[1])
                current_residue_list.append(self.sequence[item[1]])
                proposed_residue_list.append(self.inverted_amino_acids[item[0]])
                self.proposed_sequence = (
                    self.proposed_sequence[: item[1]]
                    + self.inverted_amino_acids[item[0]]
                    + self.proposed_sequence[item[1] + 1 :]
                )

            self.transition_id = (
                position_list,
                current_residue_list,
                proposed_residue_list,
            )

    def propose_allshot(
        self,
    ):
        """
        ## Propose method called when all possible steps are taken at once
        """
        # Initiate empty proposed sequence to write into
        self.proposed_sequence = ""
        position_list = []
        current_amino_acid_list = []
        proposed_amino_acid_list = []

        # Go through each position and ignore masked positions
        for position, residue_vector in enumerate(self.transition_matrix.T):
            # Ignore masked positions
            if np.sum(residue_vector) == 0:
                # Write original residue at ignored position
                self.proposed_sequence += self.sequence[position]
                continue

            index = np.argmax(residue_vector)
            if index == 20:
                continue
            amino_acid_at_index = self.inverted_amino_acids[index]
            # Write out the proposed sequence
            self.proposed_sequence += amino_acid_at_index
            # Register in the transition_ID if its a change from the original
            if self.sequence[position] != amino_acid_at_index:
                position_list.append(position)
                current_amino_acid_list.append(self.sequence[position])
                proposed_amino_acid_list.append(amino_acid_at_index)

        self.transition_id = (
            position_list,
            current_amino_acid_list,
            proposed_amino_acid_list,
        )

    def set_score(self, score_config: ModelLoadConfig):
        """
        ## Gets the score of the proposed sequence.
        ### Updates:
            self.proposed_score {float} -- Score of the proposed sequence
        """
        assert (
            self.proposed_sequence is not None
        ), f"No proposed sequence found. Run propose method first. \
        Current type of proposed_sequence is {type(self.proposed_sequence)}. \
        Needs to be str."

        self.configure_discriminator(
            pad_size=self.pad_size, sequence=self.proposed_sequence
        )
        self.proposed_score = self.discriminator.calculate_score(
            score_config=score_config
        )

    def set_decision(self) -> bool:
        """
        ## For Greedy Walk, this always accepts the proposed sequence(s)
        Before running this method, the following methods need to be run to work properly:\n
            \t-propose\n
            \t-set_score\n
        ### Updates:
                \tself.sequence -- Can either be the same or set to self.proposed_sequence\n
                \tself.sequence_regisrty -- Registry of all sequences so far accepted
        """
        self.update_registry(mode="accepted")
        return True

    def update_registry(self, mode: Literal["accepted"]):
        """
        ## Updates the sequence registry
        """
        self.sequence = self.proposed_sequence
        self.initial_score = self.proposed_score
        self.sequence_registry.append(
            (self.proposed_sequence, mode, self.transition_id, self.proposed_score)
        )

    def is_cycle(self, last_entry: Optional[tuple] = None) -> bool:
        """
        ## Checks if the walk is going in a cycle
        """
        self._cycle_counter += 1
        if last_entry is None:
            ultimate_entry = self.sequence_registry[-1]
            penultimate_entry = self.sequence_registry[-2]
        else:
            ultimate_entry = last_entry
            penultimate_entry = self.sequence_registry[-1]

        # Check if the transition_ID & score combination has happened already
        for entry in self.sequence_registry[:-1]:
            # Check mutation
            if ultimate_entry[2] == entry[2]:
                # Check score
                return isclose(ultimate_entry[3], entry[3])

        # Check if it is a direct reversion of the previous entry
        reverse_entry = (
            penultimate_entry[2][0],
            penultimate_entry[2][2],
            penultimate_entry[2][1],
        )
        if ultimate_entry[2] == reverse_entry:
            return True

        return False

    def humanization_cycle(
        self,
        score_config: ModelLoadConfig,
        allow: Union[None, Literal["CDR", "cdr"], list[int]],
        disallow: Union[None, list[int]],
    ) -> bool:
        """
        ## Runs one humanization cycle
        """
        self.set_transition_matrix(allow=allow, disallow=disallow)
        self.propose()
        self.set_score(score_config)
        self.set_decision()
        # Check for cycle
        if self.is_cycle():
            # Return False if fully_greedy to break the loop
            if self.fully_greedy:
                return False

            # Usually a sign that the same cycle is started over and over again
            if self._cycle_counter > 3:
                return False

            # Set the mutation that causes the cycle to 0
            self.reset_transition_matrix()
            # Renove the previously accepted element
            self.sequence_registry.pop()
            # Reset the sequence
            self.sequence = self.sequence_registry[-1][0]

            # Redo this round
            self.propose()
            self.set_score(score_config)
            self.set_decision()

        # Return True to continue the loop
        return True

    def reset_transition_matrix(self) -> None:
        """
        ## Resets the previously proposed mutations to zero probability
        """
        last_position = self.transition_id[0]
        proposed_mutation_amino_acids = self.transition_id[-1]
        # If the transition_ID came from multishot or allshot we need all to match
        if isinstance(last_position, list):
            # Axis 0 in transition_matrix represents the AA, and axis 1 the position
            for index, amino_acid in enumerate(proposed_mutation_amino_acids):
                self.transition_matrix[
                    self.amino_acid_dict[amino_acid], last_position[index]
                ] = 0
        else:
            self.transition_matrix[
                self.amino_acid_dict[proposed_mutation_amino_acids], last_position
            ] = 0

    def run_humanization(
        self,
        score_config: ModelLoadConfig,
        allow: Union[None, Literal["CDR", "cdr"], list[int]] = None,
        disallow: Union[None, list[int]] = None,
        max_iterations: int = 100,
    ):
        """
        ## Runs humanization for Greedy Walk
        """
        if self.n_rounds is not None:
            for _ in tqdm(range(self.n_rounds)):
                # Run one cycle of humanization
                continue_flag = self.humanization_cycle(
                    score_config, allow=allow, disallow=disallow
                )
                if not continue_flag:
                    break
            return {"greedy_walk": self.sequence_registry}

        if self.score_threshold is not None:
            for _ in tqdm(range(max_iterations)):
                # Run one cycle of humanization
                continue_flag = self.humanization_cycle(
                    score_config, allow=allow, disallow=disallow
                )
                if not continue_flag:
                    break
                if self.initial_score > self.score_threshold:
                    break
            return {"greedy_walk": self.sequence_registry}

    def run_warmup(self):
        raise NotImplementedError(
            "Maximum likelihood-type mutators do not have warmups"
        )

def main(TEST_SEQUENCE: str) -> dict:
    WEIGHTS_PATH = Path(
        "./model_weights/VAE_1MM_dkl_025_300epochs_LD_32_derivative/VAE_1MM_dkl_025_300epochs_LD_32_derivative.tf"  # pylint: disable=line-too-long
    )
    CALIBRATION_DATA_PATH = Path("./calibration_data/VAE_test_data.csv")
    PCA_DATA_PATH = Path("./calibration_data/pca_calibration_data")
    NORMAL_DISTRIBUTION_DATA = Path(
        "./calibration_data/gaussian_full_representation.json"
    )

    INPUT_SHAPE = (
        130,
        21,
    )
    # Multiprocessing is broken for this particular score as the ML model cannot be serialized
    NCPUS = 1

    # VAE Model (Used for scoring)
    vae_model = ProtVAE(
        input_shape=INPUT_SHAPE,
        latent_dimension_size=32,
        pid_algorithm=True,
        desired_kl=0.25,
        proportional_kl=0.01,
        integral_kl=0.0001,
        derivative_kl=0.0001,
    )
    # Scoring Model
    discriminator_model = AbVAE(
        sequence="",
        model=vae_model,
        encoder=OneHot(),
    )

    # Create Config
    model_configuration = ModelLoadConfig(
        file_format="tf",
        path=WEIGHTS_PATH,
        calibration_data_path=CALIBRATION_DATA_PATH,
        pca_data_path=PCA_DATA_PATH,
        gaussian_data_path=NORMAL_DISTRIBUTION_DATA,
    )

    # Load Model
    discriminator_model.load_model(model_configuration)

    # Program Loop Start
    try:
        Chain(TEST_SEQUENCE, "imgt")
    except ChainParseError as cpe:
        print("Input sequence needs to be a heavy variable domain of an antibody!")
        print(f"An error occurred: {cpe}")
        raise cpe

    # Mapping Model
    map_model = HumabMap(
        sequence=TEST_SEQUENCE,
        discriminator=discriminator_model,
        score_config=model_configuration,
        pad_size=130,
        scheme="imgt",
        n_jobs=NCPUS,
    )

    # Instantiate Greedy Walk
    humanization_pipeline = GreedyWalk(
        discriminator=discriminator_model,
        mapping=map_model,
        pad_size=130,
        sequence=TEST_SEQUENCE,
        multishot=10,
        n_rounds=1,
        score_config=model_configuration,
    )

    humanization_pipeline.run_humanization(score_config=model_configuration)

    de_immunized_seq = humanization_pipeline.sequence_registry[-1][0]
    start_score = humanization_pipeline.sequence_registry[0][-1]
    start_score = round(start_score, 3)
    end_score = humanization_pipeline.sequence_registry[-1][-1]
    end_score = round(end_score, 3)

    print(f"De-immunized Sequence: {de_immunized_seq}")
    print(
        f"Humanness increased from {start_score} to {end_score}"
    )

    return {
        "DeImmunizedSequence": de_immunized_seq,
        "HumannessScoreStart": start_score,
        "HumannessScoreEnd": end_score
    }


if __name__ == "__main__":
    print("This version only supports calls from the API.")