"""
----------- Protein VAE Fine-Tuned Model for Mousify -------------------
Defines and trains the Protein VAE for Mousify. This model structure is
inspired by the paper ProteinVAE:
https://www.biorxiv.org/content/10.1101/2023.03.04.531110v1.full
------------------------------------------------------------------------
"""

from pathlib import Path

import tensorflow as tf

from tensorflow import keras
from keras import layers
from sklearn.model_selection import RepeatedKFold

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


if __name__ == "__main__":
    # Input parameters
    WEIGHTS_PATH = Path(
        "./model_weights/VAE_1MM_dkl_025_300epochs_LD_32_derivative/VAE_1MM_dkl_025_300epochs_LD_32_derivative.tf"  # pylint: disable=line-too-long
    )
    INPUT_SHAPE_TRAINING = (
        130,
        21,
    )
    HUBER_DELTA = 1.0
    K_FOLDS = 5
    REPEATS = 100
    BATCH_SIZE = 4
    EPOCHS = 50
    REGRESSION_WEIGHT = 1.0
    RECONSTRUCTION_WEIGHT = 1.0
    LEARNING_RATE = 1e-5

    # Training Data
    training_data = [None, None, None]
    regression_data = {"decoder_output": training_data, "regression_output": [0, 0, 0]}

    # VAE Model
    protvae_model = ProtVAE(
        input_shape=INPUT_SHAPE_TRAINING,
        latent_dimension_size=INPUT_DIMENSION,
        pid_algorithm=True,
        desired_kl=0.25,
        proportional_kl=0.01,
        integral_kl=0.0001,
        derivative_kl=0.0001,
    )
    protvae_model.load_weights(filepath=WEIGHTS_PATH)

    # Setup K-Fold CV
    repeated_k_fold_data = RepeatedKFold(n_splits=K_FOLDS, n_repeats=REPEATS)
    all_models = []

    for repeat, (training_indices, validation_indices) in enumerate(
        repeated_k_fold_data.split(training_data), start=1
    ):
        print(f"\n Repeat {repeat}")

        # Split data along k-folds
        data_train, data_validation = (
            training_data[training_indices],
            training_data[validation_indices],
        )
        labels_train = {
            key: value[training_indices] for key, value in regression_data.items()
        }
        labels_validation = {
            key: value[validation_indices] for key, value in regression_data.items()
        }

        # Instantiate ADAModel
        model = ADAModel(vae_model=protvae_model)
        model.compile(
            optimizer="adam",
            loss={
                "decoder_output": "categorical_crossentropy",
                "regression_output": tf.keras.losses.Huber(delta=HUBER_DELTA),
            },
            loss_weights={
                "decoder_output": RECONSTRUCTION_WEIGHT,
                "regression_output": REGRESSION_WEIGHT,
            },
            metrics={"decoder_output": "accuracy", "regression_output": "mae"},
        )

        # Train
        trained_model = model.fit(
            x=data_train,
            y=labels_train,
            validation_data=(data_validation, labels_validation),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
            verbose=1,
        )
        all_models.append(trained_model)
