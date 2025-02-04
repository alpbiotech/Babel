""" 
    ------------------- Discriminator for Mousify --------------------------
    Takes a query sequence and calculates the %Species given a training set.
    ------------------ Variational Autoencoder Model -----------------------
    Here we use a VAE to convert every sequence to a latent space value and
    calculate the KL divergence between the latent space value and the cluster
    center of human sequences.
"""

from typing import Literal, Optional, Union
from pathlib import Path

import numpy.typing as npt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from modules.species_scoring import Discriminator
from modules.bytenet_vae_295k import ProtVAE
from modules.encoder import Encoder
from modules.onehotencoder import OneHot


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

    def calculate_score(self) -> npt.DTypeLike:
        raise NotImplementedError

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
        file_format: Literal["tf", "h5"],
        path: Path,
    ) -> None:
        assert (
            file_format == "h5" or file_format == "tf"
        ), f"For VAE, load format needs to be tf or h5, not {file_format}"
        if file_format == "h5":
            raise NotImplementedError("H5 is no longer supported for subclassed models")
        # self.model.build(input_shape=input_shape)
        self.model.load_weights(filepath=path).expect_partial()

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
        return encoding_mode_factory[mode](
            dataset=dataset, pad_size=pad_size, ncpus=ncpus
        )

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
        self.encoded = self.encoder.encode(flatten=False, pad_size=pad_size)
        return None

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
            return encoder.encode_multiprocess(pad_size=pad_size, ncpus=ncpus)

        # Encode with default encoder
        if ncpus is None:
            ncpus = 1
        self.encoder.set_sequence(dataset)
        return self.encoder.encode_multiprocess(pad_size=pad_size, ncpus=ncpus)

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


if __name__ == "__main__":

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
        derivative_kl=0.001,
    )
    Abmodel = AbVAE(
        sequence="",
        model=model_used,
        encoder=OneHot(),
    )

    BATCH_SIZE_PARAM = 10

    onehot_train_data_path = Path(
        "/central/groups/smayo/lschaus/vae_test_data/VAE_1MM_OH_Encoded_Training_Set.npy"
    )
    onehot_test_data_path = Path(
        "/central/groups/smayo/lschaus/vae_test_data/VAE_1MM_OH_Encoded_Test_Set.npy"
    )

    one_hot_train_data = np.load(onehot_train_data_path)
    one_hot_test_data = np.load(onehot_test_data_path)

    one_hot_train_data = tf.data.Dataset.from_tensor_slices(one_hot_train_data).batch(
        BATCH_SIZE_PARAM
    )
    one_hot_test_data = tf.data.Dataset.from_tensor_slices(one_hot_test_data).batch(
        BATCH_SIZE_PARAM
    )

    callbacks_model = Abmodel.with_callbacks(
        tensorboard=True,
        reduce_lr_on_plateau=True,
        path="/central/groups/smayo/lschaus/vae_test_data/VAE_1MM_dkl_025_300epochs_LD_32_derivative/callbacks",
    )

    Abmodel.train_discriminator(
        dataset=one_hot_train_data,
        epochs=300,
        batch_size=BATCH_SIZE_PARAM,
        validation_data=one_hot_test_data,
        callbacks=callbacks_model,
        data_api=True,
    )
    Abmodel.save_model(
        file_format="tf",
        path="/central/groups/smayo/lschaus/vae_test_data/VAE_1MM_dkl_025_300epochs_LD_32_derivative/VAE_1MM_dkl_025_300epochs_LD_32_derivative.tf",
    )
