""" 
    ------------------- Discriminator for Mousify --------------------------
    Takes a query sequence and calculates the %Species given a training set.
    Implemented discriminators: Hu-mAb, OASis, CNN, VAE
    ------------------------------------------------------------------------
"""

# Preamb
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join
from typing import Literal, Union, Optional
from pathlib import Path

import numpy.typing as npt
import pandas as pd


# Abstract Base Class for Discriminator module
class Discriminator(ABC):
    """
    ## Discriminator Base Class
    Used to calculate the humanization score of the antibody sequence. \n
    This class is able to train a discriminator model, save and load the trained model to
    be used later. In addition, an encoding scheme can be passed if appropriate to encode
    the sequences before calculating the score.
    """

    def __init__(self):
        self.dataset = None
        self.model = None
        self.model_path: Optional[Path] = None
        self.sequence: Union[None, str] = None

    @abstractmethod
    def calculate_score(self) -> npt.DTypeLike:
        """
        Calculate the sequences' score [0,1] based on the model used.
        ### Returns:
                \t{npt.DTypeLike} -- Score between [0,1] that signifies the %human score.
        """

    @abstractmethod
    def train_discriminator(self, dataset: npt.ArrayLike) -> Union[npt.DTypeLike, None]:
        """
        ## Method to initiate the training of the model that caluculates the %human score.
        Data path should point towards the dataset to be used for training.
        ### Args:
                \tdataset {npt.ArrayLike} -- Path to file that contains dataset

        ### Returns:
                \t{npt.DTypeLike} -- Optional: Array that contains model weights,
                parameters and scores.
        """

    @abstractmethod
    def save_model(
        self, file_format: Literal["pickle", "json", "tf", "h5"], path: Path
    ) -> None:
        """
        ## Method that saves the model as a pickle or json file.
        File is saved to the path provided
        ### Args:
                \tformat {str} -- Desired format to save the model
                                  \tOptions: \n
                                  \t -'pickle'\n
                                  \t -'json' \n
                                  \t -'tf'\n
                                  \t -'h5'\n
                \tpath {str} -- Path where the file is saved
        """

    @abstractmethod
    def load_model(self, file_format: Literal["pickle", "json"], path: Path) -> None:
        """
        ## Method that loads the pickle or json file from a pre-trained model.
        Load model sets the weights contained in the pickle or json file as the
        weights of the current model. For keras models, loads an h5 or tf file.
        ### Args:
                \tformat {str} -- Desired format to save the model
                                  \tOptions: \n
                                  \t -'pickle'\n
                                  \t -'json' \n
                                  \t -'h5'\n
                                  \t -'tf'\n
                \tpath {str} -- Path where the file is saved

        """

    @abstractmethod
    def encode(
        self,
        mode: Literal["Sequence", "seq", "Dataset", "set"],
        pad_size: int,
        dataset: Union[None, npt.ArrayLike] = None,
    ) -> Union[None, npt.ArrayLike]:
        """
        ## Encodes the sequence or list of sequences provided.
        If none is provided, it encodes self.sequence
        ### Args:
            \tmode {str} -- Defines whether to set the encoded attribute based on a single sequence or
            set of sequences, or to return a numpy array with the encoded sequence.\n
                \tOptions: \n
                \t'Sequence' or 'seq' -- Sets self.encoded with the encoded version of self.sequence \n
                \t'Dataset' or 'set' -- Returns npt.ArrayLike with the encoded version of the provided dataset
            \tdataset {npt.ArrayLike} -- Only needs to be defined in 'Dataset' or 'set' mode
        """

    def load_query(self, sequence: str) -> None:
        """
        ## Sets the sequence attribute for the class.
        Sequence can either be an amino acid sequence or a DNA sequence,
        which will be translated by the program.
        ### Args:
                \tsequence {str} -- Amino acid or DNA sequence
        ### Updates:
                    \t self.sequence {str} -- Amino acid or DNA sequence
        """
        # Implement check if it is a DNA or AA sequence
        # if check_sequence_type(sequence, 'DNA'):
        #     sequence = translate(sequence)
        self.sequence = sequence

    def load_dataset(
        self,
        path: Path,
        mode: Literal["folder", "file"] = "file",
        columns: Optional[dict] = None,
    ) -> None:
        """
        ## Loads the dataset as an array.
        Loads the file at the file path as an array and sets the dataset attribute.
        ### Args:
                \tpath {str} -- Path to the dataset location
        ### Updates:
                \tself.dataset {pd.DataFrame} -- Pandas dataframe of the loaded dataset
        """
        if mode == "file":
            if columns is not None:
                self.dataset = pd.read_csv(path, usecols=columns.keys(), dtype=columns)
            else:
                self.dataset = pd.read_csv(path)
        elif mode == "folder":
            self.dataset = self._load_directory(path=path, columns=columns)
        else:
            raise ValueError(f"Reading mode {mode} not recognized.")

    @staticmethod
    def _load_directory(
        path: Path, columns: Optional[dict] = None
    ) -> Union[pd.DataFrame, None]:
        """
        ## Loads a directory as a pd.DataFrame
        Loads all the files in the directory containing a '.csv' suffix and
        do not contain '_DL_' in their name.
        """
        dataset = None
        # Grabs all the file names in the directory
        files = [
            file
            for file in listdir(path)
            if isfile(join(path, file)) and ".csv" in file
        ]
        # Loads all the files that do not contain '_DL_' in their name (Meaning unprocessed form OAS)
        for file in files:
            if "_DL_" not in file:
                # Make sure self.dataset is populated
                if dataset is None:
                    if columns is not None:
                        dataset = pd.read_csv(
                            join(path, file), usecols=columns.keys(), dtype=columns
                        )
                    else:
                        dataset = pd.read_csv(join(path, file))
                else:
                    if columns is not None:
                        df = pd.read_csv(
                            join(path, file), usecols=columns.keys(), dtype=columns
                        )
                    else:
                        df = pd.read_csv(join(path, file))
                    dataset = pd.concat([dataset, df], ignore_index=True)

        # Make sure at least one file was found
        assert dataset is not None, "No files found that could be loaded."

        return dataset

    def train_test_split(
        self,
        values: str,
        label: Union[str, None] = None,
        split_labels: bool = True,
        ratio: float = 0.8,
        seed: Union[int, None] = None,
    ) -> dict:  # type: ignore
        """
        ## Splits dataset into a training set and test set.
        Method returns a dictionary populated with arrays, depending on the options.
        \tDict['training set'] contains the training set \n
        \tDict['test set'] contains the test set
        If split_labels is set to true (Default):
        \tDict['training labels'] contains the training labels \n
        \tDict['test labels'] contains the test labels

        ### Args:
            \t values {str} -- The column in the dataset (dataframe) that contains the values. \n
            \t label {str} -- The column in the dataset (dataframe) that contains the labels.
            If not specified, it is assumed that the data is unsupervised. \n
            \t split_labels {bool} -- If True, the Dict['training labels'],
            Dict['test labels'] are populated with the labels. If False, the labels are kept in the arrays
            in Dict['Training Set'] resp. Dict['Test Set'] as an additional column. \n
            \t ratio {float} -- Fraction of the dataset that will become the training set.
            1 - ratio will be the fraction of the test set \n
            \t seed {int} -- Seed for the RNG (For reproducing train_test_split) \n
        ### Returns:
                    \t dict -- A dictionary containing the training set and test set
        """
        # Make sure everything starts off in order
        assert ratio <= 1, "Ratio must be less or equal to 1."
        assert self.dataset is not None, "Please load dataset."

        # Check if seed has been passed for reproducibility
        if seed is not None:
            # Split the data
            training_set = self.dataset.sample(frac=ratio, random_state=seed)
        else:
            training_set = self.dataset.sample(frac=ratio)

        test_set = self.dataset.drop(training_set.index)  # type: ignore

        # Grab data
        if label is not None:
            training_set = training_set[[values, label]]
            test_set = test_set[[values, label]]

            # Create array
            if split_labels:
                # Convert to numpy
                training_set, training_labels = self._columns_to_numpy(
                    training_set, [values, label]
                )
                test_set, test_labels = self._columns_to_numpy(
                    test_set, [values, label]
                )

                return {
                    "training set": training_set,
                    "test set": test_set,
                    "training labels": training_labels,
                    "test labels": test_labels,
                }

            # Keep the labels in the array
            else:
                training_set = training_set.to_numpy()
                test_set = (
                    test_set.to_numpy()
                )  # Pandas shenannigans drops the type for some reason
        else:
            training_set = training_set[values].to_numpy()
            test_set = test_set[values].to_numpy()

        return {"training_set": training_set, "test_set": test_set}

    @staticmethod
    def _columns_to_numpy(data_frame: pd.DataFrame, cols: Union[list[str], str]):
        """
        ## Private method: Converts a column from a dataframe to a numpy array.
        ### Args:
                    \tdf {pd.DataFrame} -- Dataframe to be converted
                    \tcols {str} -- Name of the columns to be used
        """
        out = []

        for col in cols:
            out.append(data_frame[col].to_numpy())

        return tuple(out)
