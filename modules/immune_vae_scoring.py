""" 
    ------------------- Discriminator for Mousify --------------------------
    Takes a query sequence and calculates the %Species given a training set.
    ------------------ Variational Autoencoder Model -----------------------
    Here we use a VAE to convert every sequence to a latent space value and
    calculate the KL divergence between the latent space value and the cluster
    center of human sequences.
"""

import json
import pickle
from typing import Literal, Optional, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from tensorflow import keras
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial import distance

from modules.species_scoring import Discriminator
from modules.bytenet_vae_295k import ProtVAE
from modules.encoder import Encoder
from modules.onehotencoder import OneHot


@dataclass
class ModelLoadConfig:
    """
    ## Configuration Dataclass to Load AbVAE Model
    """

    file_format: Literal["tf", "h5"]
    path: Path
    pca_data_path: Path
    calibration_data_path: Optional[Path] = None
    calibration_data_column: str = "Species"
    reference_class: str = "human"
    score_type: Literal["probability", "humanness"] = "humanness"


class AbVAE(Discriminator):
    """
    ## Antibody Variational Autoencoder Discriminator calculates humaness on information distance
    The VAE is trained on a dataset of antibody sequences from different species.
    The different species clusters in latent space are then fit with a gaussian distribution.
    A query sequence fits somewhere in the latent space and  is represented as a distribution
    in latent space to calculate the information distance from the query to
    the human cluster. Various distance calculations have been implemented. \n
    ### Args: \n
        \tsequence {str} -- Antibody sequence \n
        \tchain_type {Literal} -- Option between 'heavy' or 'light' \n
        \tscheme {Literal} -- Antibody numbering scheme \n
        \tcdr_definition {Literal} -- CDR definition scheme, only needed in 'aho' scheme \n
        \tdistance_function {Literal} -- Name of the distance function implementation
    """

    def __init__(
        self,
        sequence: str,
        model: ProtVAE,
        encoder: Union[Encoder, list[Encoder]],
        distance_function: Literal["kl", "mi"] = "kl",
    ):
        super().__init__()
        self.sequence = sequence
        self.distance_function = distance_function
        self.model = model
        self.encoder = encoder
        self.encoded: Union[None, npt.ArrayLike] = None
        self.gaussian_parameter_dict: dict[str, npt.ArrayLike] = {}
        self.pca_model: Union[None, PCA] = None

    def calculate_score(self, score_config: ModelLoadConfig) -> npt.DTypeLike:
        """
        ## Calculates the Mahalanobis Probability of a Sequence
        """
        assert self.gaussian_parameter_dict, "Load models first!"
        assert self.pca_model, "Load models first!"

        # Encode Sequence
        self.encoded = self.encode(mode="Sequence", pad_size=130)

        # Get Latent Representation
        latent_representation = self.model.encoder.predict(self.encoded)
        latent_representation = latent_representation[
            0
        ]  # Tensor output needs to be sliced

        # Get PCA Representation
        pca_representation = self.pca_model.transform(latent_representation)
        pca_representation = pca_representation[0]  # PCA output is a matrix

        # Calculate Mahalanobis Probability to each Gaussian
        mahalanobis_probability = {}
        for key, value in self.gaussian_parameter_dict.items():
            mahalanobis_probability[key] = self.calculate_mahalanobis_probability(
                vector=pca_representation, gaussian_model=value
            )

        if score_config.score_type == "probability":
            # Calculate Calibrated Probability
            numerator = 0
            denominator = 0
            for key, value in mahalanobis_probability.items():
                if key == score_config.reference_class:
                    numerator = value
                denominator += value
            calibrated_probability = numerator / denominator
            return calibrated_probability

        else:
            return mahalanobis_probability[score_config.reference_class]

    @staticmethod
    def calculate_mahalanobis_probability(
        vector: npt.ArrayLike, gaussian_model: tuple[npt.ArrayLike, npt.ArrayLike]
    ) -> float:
        """
        ## Calculates the probability that a vector belongs to a given distribution
        """
        mean, covariance = gaussian_model
        inverse_covariance = np.linalg.inv(covariance)
        mahalanobis_distance = distance.mahalanobis(vector, mean, inverse_covariance)
        degrees_of_freedom = len(vector)
        return 1 - stats.chi2.cdf(mahalanobis_distance, degrees_of_freedom)

    def train_discriminator(
        self,
        dataset: npt.ArrayLike,
        epochs: int = 10,
        batch_size: int = 10,
        optimizer: Optional[str] = None,
        validation_data: Optional[npt.ArrayLike] = None,
        callbacks: Optional[list[keras.callbacks.Callback]] = None,
        data_api: bool = False,
    ) -> npt.DTypeLike:
        """
        ## Trains the VAE
        Takes as input the training data, epoch number, batch sizes and an
        optional input for the optimizer. By default the optimizer is Adam.
        ### Args:
            dataset {Array} -- Training data \n
            epochs {int} -- Number of training epochs \n
            batch_size {int} -- Number of sequences per training batch \n
            optimizer {Literal} --  Name of optimizer (See Keras website)\n
            callbacks {list[Callback]} -- List of callbacks (See Keras website) \n
            data_api {bool} -- If the data api is used as a dataset, batch_size is ignored.
        """

        # Setup strategy to use more than one GPU
        if data_api:
            batch_size = None
        # strategy = tf.distribute.MirroredStrategy()
        # Compile model
        if optimizer is None:
            # with strategy.scope():
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=0.0005, weight_decay=0.0001
                )
            )
        else:
            # with strategy.scope():
            self.model.compile(optimizer=optimizer)

        # Fit model
        try:
            with tf.device("device:GPU:0"):
                self.model.fit(
                    x=dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    callbacks=callbacks,
                )
        except RuntimeError as error:
            print(error)

    def save_model(self, file_format: Literal["tf", "h5"], path: str) -> None:
        assert (
            file_format == "h5" or file_format == "tf"
        ), f"For VAE, save format needs to be tf or h5, not {file_format}"
        self.model.save_weights(filepath=path, save_format=file_format)

    def load_model(
        self,
        loading_config: ModelLoadConfig,
    ) -> None:

        # Check the correct file format
        assert (
            loading_config.file_format == "h5" or loading_config.file_format == "tf"
        ), f"For VAE, load format needs to be tf or h5, not {loading_config.file_format}"
        if loading_config.file_format == "h5":
            raise NotImplementedError("H5 is no longer supported for subclassed models")
        self.model.load_weights(filepath=loading_config.path).expect_partial()

        # Get Principal Component Transform of Latent Data
        json_file_path, pickle_file_path = self._generate_json_and_pickle_paths(
            loading_config.pca_data_path
        )
        with open(json_file_path, "r", encoding="utf-8") as file:
            pca_representation_of_latent_data = json.load(file)

        with open(pickle_file_path, "rb") as file:
            self.pca_model = pickle.load(file)

        # Get Gaussian Representation
        for key, value in pca_representation_of_latent_data.items():
            self.gaussian_parameter_dict[key] = self._calculate_gaussian_parameters(
                value
            )

    @staticmethod
    def _calculate_gaussian_parameters(
        pca_representation: npt.ArrayLike,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        # Calculates the mean and covariance of a gaussian based-on the provided data
        """
        mean = np.mean(pca_representation, axis=0)
        covariance = np.cov(pca_representation, rowvar=False)
        return mean, covariance

    def save_pca_representation(
        self,
        loading_config: ModelLoadConfig,
    ):
        """
        ## Calculates the PCA Transformed Latent Representation and Saves It
        FOR INTERNAL USE ONLY
        """
        # Check the correct file format
        assert (
            loading_config.file_format == "h5" or loading_config.file_format == "tf"
        ), f"For VAE, load format needs to be tf or h5, not {loading_config.file_format}"
        if loading_config.file_format == "h5":
            raise NotImplementedError("H5 is no longer supported for subclassed models")
        self.model.load_weights(filepath=loading_config.path).expect_partial()

        # Load Calibration Data
        data = pd.read_csv(loading_config.calibration_data_path, index_col=0)
        data = data.reset_index()
        categories = data[loading_config.calibration_data_column].unique()

        category_dict = {}
        for cat in categories:
            category_data = data.loc[
                data[loading_config.calibration_data_column] == cat
            ]
            category_dict[cat] = category_data["Sequence_aa"].to_numpy()

        # Get latent representation
        latent_data = {}
        # We want a per-species representation
        for key, value in category_dict.items():
            # Encode Calibration data
            tqdm.write(f"Inferring latent representation of {key}...")
            encoded_sequences = self.encode(mode="Dataset", pad_size=130, dataset=value)
            latent_representation = self.model.encoder.predict(encoded_sequences)
            # Grab first slice of the tensor as latent representation
            latent_data[key] = latent_representation[0]

        # Get Principal Component Transform of Latent Data
        pca_representation_of_latent_data, pca_model = self._calculate_pca(latent_data)
        # Dump data points to json
        json_file_path, pickle_file_path = self._generate_json_and_pickle_paths(
            loading_config.pca_data_path
        )

        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump(pca_representation_of_latent_data, file)

        # Dump PCA model to pkl (sklearn can't handle much else...)
        with open(pickle_file_path, "wb") as file:
            pickle.dump(pca_model, file)

    @staticmethod
    def _calculate_pca(
        latent_representations: dict[str, npt.ArrayLike],
        components: int = 4,
        verbose: bool = True,
    ) -> dict[str, npt.ArrayLike]:
        """
        ## Calculates the PCA of a given set of latent representations
        """
        all_representations = []
        for species in latent_representations.values():
            for data in species:
                all_representations.append(data)
        pca_model = PCA(n_components=components)
        pca_model.fit(all_representations)
        if verbose:
            print(
                "PCA Explained Variance Ratio: ",
                sum(pca_model.explained_variance_ratio_),
            )

        # Transform latent data
        pca_representation = {}
        for key, value in latent_representations.items():
            pca_representation[key] = pca_model.transform(value).tolist()

        return pca_representation, pca_model

    @staticmethod
    def _generate_json_and_pickle_paths(path_prefix: Path) -> tuple[Path]:
        """
        ## Generates json and pkl suffixes for path
        """
        json_file_path = Path(f"{path_prefix}.json")
        pickle_file_path = Path(f"{path_prefix}.pkl")
        return json_file_path, pickle_file_path

    def encode(
        self,
        mode: Literal["Sequence", "seq", "Dataset", "set"],
        pad_size: int,
        dataset: Optional[npt.ArrayLike] = None,
        ncpus: Optional[int] = None,
    ) -> Union[None, npt.ArrayLike]:
        """
        ## Encodes the sequence or list of sequences provided.
        If none is provided, it encodes self.sequence
        ### Args:
            mode {str} -- Defines whether to set the encoded attribute based on a single sequence or
            set of sequences, or to return a numpy array with the encoded sequence.\n
                Options: \n
                'Sequence' or 'seq' -- Sets self.encoded with the encoded version of self.sequence\n
                'Dataset' or 'set' -- Returns npt.ArrayLike with the encoded version of the provided
                 dataset \n
            dataset {npt.ArrayLike} -- Optional: Only needs to be defined in 'Dataset'
            or 'set' mode \n
            pad_size {int} -- How much to pad by if a fixed pad size is needed for
            the model to work \n
            ncpus {int} -- Optional: Can be defined in dataset mode to accelerate encoding
        """
        # Figure out the encoding mode
        encoding_mode_factory = {
            "Sequence": self._encode_single_sequence,
            "seq": self._encode_single_sequence,
            "Dataset": self._encode_dataset,
            "set": self._encode_dataset,
        }

        # Check if its a list of encoders or just a single encoder
        if isinstance(self.encoder, list):
            return self._encode_from_multiple_encoders(
                mode=mode,
                dataset=dataset,
                pad_size=pad_size,
                ncpus=ncpus,
                encoding_mode_factory=encoding_mode_factory,
            )
        try:
            return encoding_mode_factory[mode](
                dataset=dataset, pad_size=pad_size, ncpus=ncpus
            )
        # The factory has two possible signatures, probably not a great design choice
        except TypeError:
            return encoding_mode_factory[mode](pad_size=pad_size)

    def _encode_from_multiple_encoders(
        self,
        mode: Literal["Sequence", "seq", "Dataset", "set"],
        dataset: npt.ArrayLike,
        pad_size: int,
        ncpus: Optional[int],
        encoding_mode_factory: dict,
    ) -> list[npt.ArrayLike]:
        """
        ## Encoding used when there are multiple encoders needed
        """
        encoding_mode = encoding_mode_factory[mode]

        # Append encoded data from each encoder to list
        encoded_data = []
        for encoder in self.encoder:
            encoded_data.append(
                encoding_mode(
                    dataset=dataset, pad_size=pad_size, ncpus=ncpus, encoder=encoder
                )
            )
        return encoded_data

    def _encode_single_sequence(
        self,
        pad_size: int,
        encoder: Optional[Encoder] = None,
    ) -> None:
        """
        ## Encodes single sequence
        dataset and ncpus as arguments are just there to keep the same signature for the factory.
        """
        # In case an encoder is provided
        if encoder is not None:
            encoder.set_sequence([self.sequence])
            return encoder.encode(flatten=False, pad_size=pad_size)

        # Use default encoder
        self.encoder.set_sequence([self.sequence])
        encoded = self.encoder.encode(flatten=False, pad_size=pad_size)
        return encoded

    def _encode_dataset(
        self,
        dataset: npt.ArrayLike,
        pad_size: int,
        ncpus: Optional[int] = None,
        encoder: Optional[Encoder] = None,
    ) -> npt.ArrayLike:
        """
        ## Encodes in dataset mode
        """
        assert (
            dataset is not None
        ), "Need to provide a dataset if run in 'Dataset' or 'set' mode."

        # In case an encoder is provided
        if encoder is not None:
            if ncpus is None:
                ncpus = 1
            encoder.set_sequence(dataset)
            return encoder.encode(pad_size=pad_size)

        # Encode with default encoder
        if ncpus is None:
            ncpus = 1
        self.encoder.set_sequence(dataset)
        return self.encoder.encode(pad_size=pad_size)

    @staticmethod
    def with_callbacks(
        tensorboard: bool = True,
        reduce_lr_on_plateau: bool = False,
        path: str = "../data/callbacks",
    ) -> list[keras.callbacks.Callback]:
        """
        # Returns pre-set callbacks
        The tensorboard callback is the
        """
        callbacks = []

        if tensorboard:
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=path)
            callbacks.append(tensorboard_callback)

        if reduce_lr_on_plateau:
            lr_callback = keras.callbacks.ReduceLROnPlateau(
                monitor="val_reconstruction_loss",
                factor=0.1,
                verbose=1,
                min_delta=0.01,
            )
            callbacks.append(lr_callback)

        return callbacks

    def create_map(
        self,
    ):  # latent_representation: npt.ArrayLike):
        """
        ## Calculates the Transition Matrix for a given sequence input
        """
        # Get Latent Representation
        latent_representation = self.model.encoder.predict(self.encoded)
        latent_representation = latent_representation[
            0
        ]  # Tensor output needs to be sliced
        transition_matrix = self.model.decoder.predict(latent_representation)
        return transition_matrix


if __name__ == "__main__":
    TEST_SEQUENCE_MOUSE = "QVKLQQSGPELKKPGETVKISCKASGYTFTDYSMHWVKQAPGKGLKWLGRINTETGEAKYVDDFMGHLAFSLETSASTAYLQINNLKNEDTATYFCARYDGYSWDAMDYWGQGTSVIVSS"
    TEST_SEQUENCE_HUMAN = "QVQLVQSGAEVKKPGSSVRVSCKASGDTFSSYSITWVRQAPGHGLQWMGGIFPIFGSTNYAQKFDDRLTITTDDSSRTVYMELTSLRLEDTAVYYCARGASKVEPAAPAYSDAFDMWGQGTLVTVSS"
    WEIGHTS_PATH = Path(
        "./model_weights/VAE_1MM_dkl_025_300epochs_LD_32_derivative/VAE_1MM_dkl_025_300epochs_LD_32_derivative.tf"  # pylint: disable=line-too-long
    )
    CALIBRATION_DATA_PATH = Path("./calibration_data/VAE_test_data.csv")
    PCA_DATA_PATH = Path("./calibration_data/pca_calibration_data")

    INPUT_SHAPE = (
        130,
        21,
    )

    model_used = ProtVAE(
        input_shape=(
            130,
            21,
        ),
        latent_dimension_size=32,
        pid_algorithm=True,
        desired_kl=0.25,
        proportional_kl=0.01,
        integral_kl=0.0001,
        derivative_kl=0.0001,
    )
    Abmodel = AbVAE(
        sequence="",
        model=model_used,
        encoder=OneHot(),
    )
    # Create Config
    model_configuration = ModelLoadConfig(
        file_format="tf",
        path=WEIGHTS_PATH,
        calibration_data_path=CALIBRATION_DATA_PATH,
        pca_data_path=PCA_DATA_PATH,
    )
    # Load Model
    Abmodel.load_model(model_configuration)
    Abmodel.sequence = TEST_SEQUENCE_MOUSE
    print(Abmodel.calculate_score(model_configuration))
    transition_map = Abmodel.create_map()
    print(transition_map[0][0])
