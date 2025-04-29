"""
## Training run script for the ADA fine-tuned model
"""

from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import RepeatedKFold, train_test_split

from Babel.modules.ada_fine_tuned_model import ADAModel, INPUT_DIMENSION
from Babel.modules.bytenet_vae_295k import ProtVAE
from Babel.modules.onehotencoder import OneHot


if __name__ == "__main__":
    # Input parameters
    WEIGHTS_PATH = Path(
        "/home/lschaus/vscode/alpai/Babel/model_weights/VAE_1MM_dkl_025_300epochs_LD_32_derivative/VAE_1MM_dkl_025_300epochs_LD_32_derivative.tf"  # pylint: disable=line-too-long
    )
    DATA_PATH = Path("/home/lschaus/vscode/data/Approved_mAbs_with_sequence.csv")

    INPUT_SHAPE_TRAINING = (
        130,
        21,
    )
    HUBER_DELTA = 1.0
    K_FOLDS = 5
    REPEATS = 100
    BATCH_SIZE = 1
    EPOCHS = 50
    REGRESSION_WEIGHT = 1.0
    RECONSTRUCTION_WEIGHT = 1.0
    LEARNING_RATE = 1e-2

    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )

    # Training Data
    # Load Data
    train_dataframe = pd.read_csv(filepath_or_buffer=DATA_PATH, index_col=0)
    heavy_sequences = train_dataframe["Sequence_VH"].to_numpy()
    ada_scores = train_dataframe["ADA"].to_numpy() / 100
    # Hold out test set
    heavy_sequences_train, heavy_sequences_test, ada_scores_train, ada_scores_test = (
        train_test_split(
            heavy_sequences, ada_scores, test_size=0.2, random_state=4893, shuffle=True
        )
    )
    # Encode Data
    encoder = OneHot(sequence=heavy_sequences_train)
    training_data = encoder.encode(pad_size=130)
    regression_data = {
        "decoder_output": training_data,
        "regression_output": ada_scores_train,
    }

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
                ),
                learning_rate_scheduler,
            ],
            verbose=1,
        )
        all_models.append(trained_model)

    # Get model weights
    all_weights = [
        individual_model.model.get_weights() for individual_model in all_models
    ]
    # Average weights
    averaged_weights = []
    for weights in zip(*all_weights):
        weights_stack = np.stack(weights, axis=0)
        mean_weight = np.mean(weights_stack, axis=0)
        averaged_weights.append(mean_weight)

    # Create final model
    final_model = ADAModel(vae_model=protvae_model)
    final_model.compile(
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
    final_model.set_weights(averaged_weights)

    final_model.model.save(
        "/home/lschaus/vscode/ada_training/20250428_ADAModel_Average_Repeat_100.keras"
    )
    np.save(
        "/home/lschaus/vscode/ada_training/20250428_Test_Set_Sequences.npy",
        heavy_sequences_test,
    )
    np.save(
        "/home/lschaus/vscode/ada_training/20250428_Test_Set_ADA_Scores.npy",
        ada_scores_test,
    )
