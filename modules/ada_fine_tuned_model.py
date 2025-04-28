"""
----------- Protein VAE Fine-Tuned Model for Mousify -------------------
Defines and trains the Protein VAE for Mousify. This model structure is
inspired by the paper ProteinVAE:
https://www.biorxiv.org/content/10.1101/2023.03.04.531110v1.full
------------------------------------------------------------------------
"""

import tensorflow as tf

from tensorflow import keras
from keras import layers

from Babel.modules.bytenet_vae_295k import ProtVAE

FILTER_NUMBER = 8
INPUT_DIMENSION = 32


class ADAModel(tf.keras.Model):
    """
    ## ADA Regression Fine-Tuning Model for ByteNetVAE
    """

    def __init__(self, vae_model: ProtVAE, verbose: bool = True):
        super().__init__()
        self.vae_model = vae_model
        self.verbose = verbose

        for layer in self.vae_model.encoder.layers:
            layer.trainable = False

        for layer in self.vae_model.decoder.layers:
            layer.trainable = True

        self.regression_model = self.feed_forward_regression_model()

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        ## Call method required for subclassing a keras Model
        """
        # Encode sequence
        latent_vector = self.vae_model.encoder(inputs)

        # Decode sequence
        decoded_sequence = self.vae_model.decoder(latent_vector)

        # ADA Regression
        ada_score = self.regression_model(latent_vector)

        return {"decoder_output": decoded_sequence, "regression_output": ada_score}

    def feed_forward_regression_model(self):
        """
        ## Feed forward network for ADA regression
        """
        # Input
        input_layer = keras.Input(
            shape=(
                None,
                INPUT_DIMENSION,
            )
        )

        # ByteNet Block 1
        bytenet_layer = layers.LayerNormalization(axis=-1)(input_layer)
        bytenet_layer = layers.Conv1D(
            filters=FILTER_NUMBER,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="gelu",
        )(bytenet_layer)
        bytenet_layer = layers.LayerNormalization(axis=-1)(bytenet_layer)
        bytenet_layer = layers.Conv1D(
            filters=FILTER_NUMBER,
            kernel_size=5,
            padding="same",
            dilation_rate=3,
            activation="gelu",
        )(bytenet_layer)
        bytenet_layer = layers.LayerNormalization(axis=-1)(bytenet_layer)
        bytenet_layer = layers.Conv1D(
            filters=FILTER_NUMBER,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="gelu",
        )(bytenet_layer)
        forward_layer = layers.Add()([input_layer, bytenet_layer])

        # Dense
        forward_layer = layers.Dense(
            INPUT_DIMENSION,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(forward_layer)
        forward_layer = layers.Dropout(0.25)(forward_layer)
        forward_layer = layers.Dense(
            INPUT_DIMENSION / 2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(forward_layer)
        forward_layer = layers.Dropout(0.25)(forward_layer)

        # ByteNet Block 2
        bytenet_layer = layers.LayerNormalization(axis=-1)(forward_layer)
        bytenet_layer = layers.Conv1D(
            filters=FILTER_NUMBER,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="gelu",
        )(bytenet_layer)
        bytenet_layer = layers.LayerNormalization(axis=-1)(bytenet_layer)
        bytenet_layer = layers.Conv1D(
            filters=FILTER_NUMBER,
            kernel_size=5,
            padding="same",
            dilation_rate=3,
            activation="gelu",
        )(bytenet_layer)
        bytenet_layer = layers.LayerNormalization(axis=-1)(bytenet_layer)
        bytenet_layer = layers.Conv1D(
            filters=FILTER_NUMBER,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="gelu",
        )(bytenet_layer)
        forward_layer = layers.Add()([forward_layer, bytenet_layer])

        # Dense
        forward_layer = layers.Dense(
            INPUT_DIMENSION / 2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(forward_layer)
        forward_layer = layers.Dropout(0.25)(forward_layer)
        forward_layer = layers.Dense(
            INPUT_DIMENSION / 4,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(forward_layer)
        forward_layer = layers.Dropout(0.25)(forward_layer)
        output_layer = layers.Dense(1, activation="sigmoid")(forward_layer)

        # Define Model
        feed_forward_model = keras.Model(input_layer, output_layer, name="regression")

        if self.verbose:
            feed_forward_model.summary()

        return feed_forward_model
