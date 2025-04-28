"""
-------------------- Protein VAE for Babel ---------------------------
Defines and trains the Protein VAE for Babel. This model structure is
inspired by the paper ProteinVAE:
https://www.biorxiv.org/content/10.1101/2023.03.04.531110v1.full
------------------------------------------------------------------------
"""

from typing import Optional

import numpy.typing as npt

import tensorflow as tf
from tensorflow import keras
from keras import layers


class Sampler(layers.Layer):
    """
    ## Uses latent_mean and latent_var to sample the latent vector encoding the protein sequence.
    The sampling is performed as a tensorflow layer with the reparametrization trick. This allows
    the gradient to "flow" through this layer.
    """

    # Note: Keras call() != __call__() because it also builds weight and bias tensors
    def call(
        self, mean: npt.ArrayLike, variance: npt.ArrayLike
    ):  # pylint: disable=arguments-differ
        """
        ## Calls a Sampler instance as a function
        Samples the latent vector as a tensorflow layer.
        ### Args:
            - \tmean {ArrayLike} -- latent dimension mean layer
            - \tvariance {ArrayLike} -- latent dimension variance layer
        """
        batch = tf.shape(mean)[0]
        dimension = tf.shape(mean)[1]
        reparametrizer = tf.random.normal(shape=(batch, dimension))
        return mean + tf.exp(0.5 * variance) * reparametrizer


class ProtVAE(keras.Model):
    """
    ## Implementation of the protein VAE
    Uses three layers of 1D convolutions with ELU activation functions in the encoder.
    In the decoder step, we use 8 consecutive "UpBlock"s.
    Each "UpBlock" consists of a batch norm-ELU-Conv1D block,
    followed by a batch norm-ELU-ConvTranspose1D block.
    The consecutive blocks increase the shape of the output.
    Lastly, the last UpBlock output is reduced to the final output size using Conv1D.
    ### Args: \n
        - \t See tf.keras.Inputs
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dimension_size: int,
        encoder: Optional[keras.Model] = None,
        decoder: Optional[keras.Model] = None,
        pid_algorithm: bool = False,
        desired_kl: Optional[float] = None,
        proportional_kl: Optional[float] = None,
        integral_kl: Optional[float] = None,
        derivative_kl: Optional[float] = None,
        beta_max: float = 1.0,
        beta_min: float = 0.00001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define encoders and decoders
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self.encoder_block(input_shape, latent_dimension_size)
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = self.decoder_block(latent_dimension_size)

        # Define loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_beta_tracker = keras.metrics.Mean(name="beta_score")

        # If PID Controller KL-Divergence is enabled
        if pid_algorithm:
            assert (
                desired_kl
                and proportional_kl
                and integral_kl
                and derivative_kl is not None
            ), "desired_kl, proportional_kl, integral_kl and derivative_kl cannot \
            be None if PID is enabled!"
            # Set PID flag
            self.pid = True
            # Initialize KL scores
            self.desired_kl = desired_kl
            self.proportional_kl = proportional_kl
            self.integral_kl = integral_kl
            self.derivative_kl = derivative_kl
            self.beta_max = tf.constant(beta_max)
            self.beta_min = tf.constant(beta_min)
            # Initialize tensor of errors for the integral term
            self.beta_errors = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )
            # Since the tensor is written dynamically, a positional argument needs to be created
            self.beta_iteration_counter = 0

        else:
            self.pid = False
        # Initialize current_control_score
        self.current_control_score = 1

    def call(self, inputs):  # pylint: disable=arguments-differ
        """
        ## Call method required for subclassing a keras Model
        """
        samples = self.encoder(inputs)
        if self.pid:
            self.beta_iteration_counter += 1
        return self.decoder(samples[2])

    def encoder_block(
        self,
        input_shape: tuple[int, int, int],
        latent_dimension_size: int,
        verbose: bool = True,
        initial_filter_number: int = 32,
    ) -> keras.Model:
        """
        ## Defines the encoder layer of the VAE
        Encoder is made up of four consecutive Conv1D layers, followed by a flattening layer.
        Then we create two separate dense layers from that to get the mean and the
        log_variance vectors.
        ### Args: \n
            \tinput_shape {tuple} -- shape of the input data \n
            \tlatent_dimension_size {int} -- Length of the latent dimension vector \n
            \tverbose {bool} -- Summarizes the encoder block \n
            \tinitial_filter_number {int} -- Number of filters to learn at each step \n

        Note: Filter increases by a factor of 2 in each layer
        """
        encoder_inputs = keras.Input(shape=input_shape)

        # Encoder layer 1
        encoder_layer = layers.Conv1D(
            filters=initial_filter_number,
            kernel_size=3,
            strides=2,
            padding="same",
            dilation_rate=1,
            activation="elu",
        )(encoder_inputs)

        # Encoder layer 2
        encoder_layer = layers.Conv1D(
            filters=initial_filter_number * 2,
            kernel_size=3,
            strides=2,
            padding="same",
            dilation_rate=1,
            activation="elu",
        )(encoder_layer)

        # Encoder layer 3
        encoder_layer = layers.Conv1D(
            filters=initial_filter_number * 4,
            kernel_size=3,
            strides=2,
            padding="same",
            dilation_rate=1,
            activation="elu",
        )(encoder_layer)

        # Encoder layer 4
        encoder_layer = layers.Conv1D(
            filters=initial_filter_number * 8,
            kernel_size=3,
            strides=2,
            padding="same",
            dilation_rate=1,
            activation="elu",
        )(encoder_layer)

        # Flatten and then feed-forward
        encoder_layer = layers.Flatten()(encoder_layer)
        encoder_layer = layers.Dense(units=latent_dimension_size, activation="relu")(
            encoder_layer
        )

        # Reparametrization
        latent_mean = layers.Dense(latent_dimension_size, name="latent_mean")(
            encoder_layer
        )
        latent_log_variance = layers.Dense(
            latent_dimension_size, name="latent_log_variance"
        )(encoder_layer)
        latent_sample = Sampler()(mean=latent_mean, variance=latent_log_variance)

        # Create model
        encoder = keras.Model(
            encoder_inputs,
            [latent_mean, latent_log_variance, latent_sample],
            name="encoder",
        )
        # Summary
        if verbose:
            encoder.summary()

        return encoder

    def decoder_block(
        self, latent_dimension_size: int, verbose: bool = True
    ) -> keras.Model:
        """
        ## Defines the decoder layer of the VAE
        It consists of 3 upblock layers, followed by a Conv1D & linear layer that brings the
        sequences to a 130 x 21 shape, followed by softmax to get residue probabilites
        and then an argmax to return a sequence. Each upblock consists of 2 ConvTranspose1D
        blocks to upsample.
        This upsample (L(out) x d(up)) is concatenated with a L(out) x d(in) to produce a
        matrix of shape L(out) x d(out), where d(out) = d(up) + d(in).
        At the end of the final upblock, the shape of the output will be ~2*L(antibody) x d(CNN),
        where 2*L(antibody) is approximately double the maximum length of an antibody
        and d(CNN) depends on the parameter size of the deconvolution.
        ### Note:
        The final L x 21 shape, is L for the protein sequence length and 21 for 20 amino acids and
        one position to define "no amino acid".
        """
        latent_inputs = keras.Input(
            shape=(
                None,
                latent_dimension_size,
            )
        )

        # Upblock layer 1
        decoder_layer = layers.Reshape((latent_dimension_size, 1))(latent_inputs)
        decoder_layer = layers.BatchNormalization()(decoder_layer)
        decoder_layer = layers.Conv1DTranspose(
            filters=64, kernel_size=1, activation="elu"
        )(decoder_layer)
        decoder_layer = layers.BatchNormalization()(decoder_layer)
        decoder_layer = layers.Conv1DTranspose(
            filters=32, kernel_size=3, strides=2, activation="elu"
        )(decoder_layer)
        concatenation_layer = layers.Dense(65, activation="relu")(latent_inputs)
        concatenation_layer = layers.Reshape((65, 1))(concatenation_layer)
        decoder_layer = layers.Concatenate(axis=2)([decoder_layer, concatenation_layer])

        # Upblock layer 2
        upblock_layer = layers.BatchNormalization()(decoder_layer)
        upblock_layer = layers.Conv1DTranspose(
            filters=64, kernel_size=1, activation="elu"
        )(upblock_layer)
        upblock_layer = layers.BatchNormalization()(upblock_layer)
        upblock_layer = layers.Conv1DTranspose(
            filters=32, kernel_size=3, strides=2, activation="elu"
        )(upblock_layer)
        concatenation_layer = layers.Dense(131, activation="relu")(decoder_layer)
        concatenation_layer = layers.Reshape((131, 65))(concatenation_layer)
        decoder_layer = layers.Concatenate(axis=2)([upblock_layer, concatenation_layer])

        # Upblock layer 3 (Maximum length of 130 achieved after Conv1D)
        upblock_layer = layers.BatchNormalization()(decoder_layer)
        upblock_layer = layers.Conv1DTranspose(
            filters=64, kernel_size=1, activation="elu"
        )(upblock_layer)
        upblock_layer = layers.BatchNormalization()(upblock_layer)
        upblock_layer = layers.Conv1DTranspose(
            filters=32, kernel_size=3, strides=2, activation="elu"
        )(upblock_layer)
        concatenation_layer = layers.Dense(263, activation="relu")(decoder_layer)
        concatenation_layer = layers.Reshape((263, 131))(concatenation_layer)
        decoder_layer = layers.Concatenate(axis=2)([upblock_layer, concatenation_layer])

        # Dropout
        decoder_layer = layers.Dropout(rate=0.3)(decoder_layer)

        # Final convolution
        decoder_layer = layers.Conv1D(
            filters=32, kernel_size=5, strides=2, activation="elu"
        )(decoder_layer)
        # Final dense layer
        decoder_layer = layers.Dense(21, activation="relu")(decoder_layer)
        # Softmax and Argmax
        decoder_layer = layers.Softmax()(decoder_layer)
        # One hot encode
        # decoder_layer = tf.one_hot(tf.argmax(decoder_layer, 2), depth=21)
        # Define model
        decoder = keras.Model(latent_inputs, decoder_layer, name="decoder")

        if verbose:
            decoder.summary()

        return decoder

    @property
    def metrics(self) -> list:
        """
        ## Returns Mean Model Metrics
        Returns the mean total-, reconstruction- and KL-divergence loss tracker.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_beta_tracker,
        ]

    def kullback_leibler_loss(self, mean: npt.ArrayLike, log_variance: npt.ArrayLike):
        """
        ## Returns KL Loss
        Uses the Shao et al. 2020 ControlVAE PID algorithm to calculate the KL-divergence
        loss to prevent KL vanishing.
        See Bowman et al. 2015, Liu et al 2019.
        ### Args:
            - \tmean {ArrayLike} -- Mean vector of latent space
            - \tlog_variance {ArrayLike} -- Log of the variance vector of latent space
        """
        return -0.5 * (1 + log_variance - tf.square(mean) - tf.exp(log_variance))

    def train_step(self, data: npt.ArrayLike) -> dict:
        """
        ## Training of the VAE
        By calling the function train_step, it overrides the function in keras.Models.
        Since we are using a custom loss tracker, we need to define a metrics method as well.
        Therefore when model.fit() is called, only the optimizer needs to be passed, all else
        is overridden. Control score is adapted from the Shao et al. 2020
        ControlVAE PID algorithm to change the KL divergence loss to prevent KL vanishing.
        See Bowman et al. 2015, Liu et al 2019.
        """

        # Set gradient context manager
        with tf.GradientTape() as tape:
            # Get latent values
            mean, log_variance, sample = self.encoder(data)
            # Reconstruct from the sample
            reconstruction = self.decoder(sample)
            # Calculcate reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.categorical_crossentropy(data, reconstruction),
                    axis=0,
                )
            )
            # Calculate KL Loss
            kl_loss = self.kullback_leibler_loss(mean=mean, log_variance=log_variance)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Calculate control score (beta) of the PID algorithm
            if self.pid:
                # Control score stays 1 if PID flag is false
                # Get error vs desired KL
                error = self.desired_kl - kl_loss
                # Add new error to TensorArray of errors and add to iteration
                self.beta_errors = self.beta_errors.write(
                    self.beta_iteration_counter, error
                )
                # Calculate proportional term
                proportional_term = self.proportional_kl / (1 + tf.exp(error))
                # Calculate integral term
                integral_term = self.integral_kl * tf.reduce_sum(
                    self.beta_errors.stack()
                )
                # Calculate derivative term
                gathered_error_terms = self.beta_errors.gather(
                    [self.beta_iteration_counter, self.beta_iteration_counter - 1]
                )
                derivative_term = self.derivative_kl * tf.math.subtract(
                    gathered_error_terms[0],
                    gathered_error_terms[1],
                )
                # Get control score
                control_score = proportional_term - integral_term + derivative_term
                # If control score is larger than beta_max, then switch to beta_max
                control_score = tf.cond(
                    control_score < self.beta_max,
                    lambda: control_score,
                    lambda: self.beta_max,
                )
                # If control score is lower than beta_min, then switch to beta_min
                control_score = tf.cond(
                    control_score > self.beta_min,
                    lambda: control_score,
                    lambda: self.beta_min,
                )

            else:
                control_score = 1

            # Calculate total loss
            total_loss = reconstruction_loss + control_score * kl_loss

        # Apply gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update losses
        self.total_loss_tracker.update_state(total_loss)  # pylint: disable=not-callable
        self.reconstruction_loss_tracker.update_state(  # pylint: disable=not-callable
            reconstruction_loss  # pylint: disable=not-callable
        )  # pylint: disable=not-callable
        self.kl_loss_tracker.update_state(kl_loss)  # pylint: disable=not-callable
        self.kl_beta_tracker.update_state(  # pylint: disable=not-callable
            control_score * kl_loss  # pylint: disable=not-callable
        )  # pylint: disable=not-callable

        # Return dictionary of losses
        return {
            "total_loss": self.total_loss_tracker.result(),  # pylint: disable=not-callable
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),  # pylint: disable=not-callable
            "kl_loss": self.kl_loss_tracker.result(),  # pylint: disable=not-callable
            "beta_score": self.kl_beta_tracker.result(),  # pylint: disable=not-callable
        }

    def test_step(self, data: npt.ArrayLike):
        """
        ## Validation of VAE
        Returns the validation error. Needs to update the loss tracker at the end.
        Note: test_step does not include the PID correction term to the kl divergence since
        the tensor is not accessible inside this function.
        """
        # Only need one copy of data since the model is unsupervised
        validation_data = data
        mean, log_variance, sample = self.encoder(validation_data)
        reconstruction = self.decoder(sample)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(validation_data, reconstruction),
                axis=0,
            )
        )

        # Calculate KL Loss
        kl_loss = self.kullback_leibler_loss(mean=mean, log_variance=log_variance)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Calculate total loss
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)  # pylint: disable=not-callable
        self.reconstruction_loss_tracker.update_state(  # pylint: disable=not-callable
            reconstruction_loss  # pylint: disable=not-callable
        )  # pylint: disable=not-callable
        self.kl_loss_tracker.update_state(kl_loss)  # pylint: disable=not-callable
        return {
            "total_loss": self.total_loss_tracker.result(),  # pylint: disable=not-callable
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),  # pylint: disable=not-callable
            "kl_loss": self.kl_loss_tracker.result(),  # pylint: disable=not-callable
        }


if __name__ == "__main__":
    A = ProtVAE(input_shape=(130, 21), latent_dimension_size=32)
