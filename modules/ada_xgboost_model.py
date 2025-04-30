"""
----------- Protein VAE XGBoost Model for Mousify -------------------
Defines and trains the Protein VAE for Mousify. This model structure is
inspired by the paper ProteinVAE:
https://www.biorxiv.org/content/10.1101/2023.03.04.531110v1.full
------------------------------------------------------------------------
"""

from typing import Optional
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
import numpy.typing as npt
import xgboost as xgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score

from Babel.modules.bytenet_vae_295k import ProtVAE
from Babel.modules.onehotencoder import OneHot


class LatentRegressor:
    def __init__(
        self,
        vae_model: ProtVAE,
        test_size=0.2,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ):
        self.encoder_model = vae_model
        self.test_set_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.model_history = {}

    def encode(self, data: npt.NDArray) -> npt.NDArray:
        """
        ## Encodes sequences
        Encodes One-Hot encoded antibody sequences to a 32D latent space
        """
        return self.encoder_model.encoder.predict(data)

    def fit(
        self,
        training_data: npt.NDArray,
        training_labels: npt.NDArray,
        update_history: bool = True,
        store_model: bool = True,
        test_label: Optional[npt.NDArray] = None,
        test_data: Optional[npt.NDArray] = None,
    ):
        """
        ## Fits the XGBoost model
        """
        # Encode
        latent_mean, _, __ = self.encode(training_data)

        # Train test split
        if test_data is None:
            latent_training, test_data, label_training, test_label = train_test_split(
                latent_mean,
                training_labels,
                test_size=self.test_set_size,
                random_state=self.random_state,
            )
        else:
            latent_training = latent_mean
            label_training = training_labels

        # Instantiate Model and Fit
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )

        model.fit(latent_training, label_training)

        if store_model:
            self.model = model

        # Get metrics
        metrics_dict = self.metrics(
            latent_training, test_data, label_training, test_label
        )
        if update_history:
            self.model_history.update(metrics_dict)

        if self.verbose:
            print("\n", "Fit complete...", "\n")
            print(
                f"Train MAE: {metrics_dict['mae']['train']}",
                f"Test MAE: {metrics_dict['mae']['test']}",
                "\n",
            )
            print(
                f"Train R2: {metrics_dict['r2']['train']}",
                f"Test R2: {metrics_dict['r2']['test']}",
                "\n",
            )

        return model, metrics_dict

    def metrics(
        self,
        latent_training: npt.NDArray,
        latent_test: npt.NDArray,
        label_training: npt.NDArray,
        label_test: npt.NDArray,
    ) -> dict[str, dict[str, float]]:
        """
        ## Calculate model metrics
        """
        test_prediction = self.model.predict(latent_test)
        train_prediction = self.model.predict(latent_training)

        train_loss = mean_absolute_error(label_training, train_prediction)
        test_loss = mean_absolute_error(label_test, test_prediction)

        train_r2 = r2_score(label_training, train_prediction)
        test_r2 = r2_score(label_test, test_prediction)

        return {
            "mae": {"train": train_loss, "test": test_loss},
            "r2": {"train": train_r2, "test": test_r2},
        }

    def predict(self, data: npt.NDArray) -> float:
        """
        ## Predicts regression output
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded...")

        latent_vector, _, __ = self.encoder_model.encoder.predict(data)
        return self.model.predict(latent_vector)

    def score(
        self,
    ) -> dict:
        """
        ## Returns MAE and R2 metrics
        """
        return self.model_history

    def cross_validation_fit(
        self, sequence_data: npt.NDArray, label_data: npt.NDArray, n_splits=5
    ):
        """
        ## Train with Cross-Validation
        """
        k_fold_object = KFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        models, mae_scores, r2_scores = [], [], []

        for fold, (training_indices, test_indices) in enumerate(
            k_fold_object.split(sequence_data)
        ):
            if self.verbose:
                print("\n", f"Fold {fold+1}")

            training_data, test_data = (
                sequence_data[training_indices],
                sequence_data[test_indices],
            )
            training_labels, test_labels = (
                label_data[training_indices],
                label_data[test_indices],
            )

            model, metrics = self.fit(
                training_data=training_data,
                training_labels=training_labels,
                test_data=test_data,
                test_label=test_labels,
                store_model=False,
                update_history=False,
            )
            models.append(model)
            mae_scores.append(metrics["mae"])
            r2_scores.append(metrics["r2"])

        self.model_history["mae"] = mae_scores
        self.model_history["r2"] = r2_scores

        return models, mae_scores, r2_scores

    def save_model(self, filepath: Path) -> None:
        """
        ## Saves model in self.model
        """
        if self.model is None:
            raise ValueError("No trained model to save...")
        joblib.dump(self.model, filepath)
        if self.verbose:
            print("\n", f"Model saved to {filepath}")

    def load_model(self, filepath: Path) -> None:
        """
        ## Loads model to self.model
        """
        self.model = joblib.load(filepath)
        if self.verbose:
            print("\n", f"Model loaded from {filepath}")


if __name__ == "__main__":
    WEIGHTS_PATH = Path(
        "/home/lschaus/vscode/alpai/Babel/model_weights/VAE_1MM_dkl_025_300epochs_LD_32_derivative/VAE_1MM_dkl_025_300epochs_LD_32_derivative.tf"  # pylint: disable=line-too-long
    )
    DATA_PATH = Path("/home/lschaus/vscode/data/Approved_mAbs_with_sequence.csv")
    MODEL_SAVE_PATH = Path(
        "/home/lschaus/vscode/ada_training/20250429_XGB_ADA_Model_1.joblib"
    )
    INPUT_DIMENSION = 32
    INPUT_SHAPE_TRAINING = (
        130,
        21,
    )

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

    # Training Data
    # Load Data
    train_dataframe = pd.read_csv(filepath_or_buffer=DATA_PATH, index_col=0)
    heavy_sequences = train_dataframe["Sequence_VH"].to_numpy()
    ada_scores = train_dataframe["ADA"].to_numpy() / 100

    # Hold out test set
    heavy_sequences_train, heavy_sequences_test, ada_scores_train, ada_scores_test = (
        train_test_split(
            heavy_sequences, ada_scores, test_size=0.1, random_state=4893, shuffle=True
        )
    )

    # Encode Data
    encoder = OneHot(sequence=heavy_sequences)
    encoded_data = encoder.encode(pad_size=130)

    ada_regression_model = LatentRegressor(vae_model=protvae_model, random_state=4893)
    ada_regression_model.fit(training_data=encoded_data, training_labels=ada_scores)
    ada_regression_model.save_model(MODEL_SAVE_PATH)

    np.save(
        "/home/lschaus/vscode/ada_training/20250429_Test_Set_Sequences.npy",
        heavy_sequences_test,
    )
    np.save(
        "/home/lschaus/vscode/ada_training/20250429_Test_Set_ADA_Scores.npy",
        ada_scores_test,
    )
    np.save(
        "/home/lschaus/vscode/ada_training/20250429_Training_Set_Sequences.npy",
        heavy_sequences_train,
    )
    np.save(
        "/home/lschaus/vscode/ada_training/20250429_Training_Set_ADA_Scores.npy",
        ada_scores_train,
    )
