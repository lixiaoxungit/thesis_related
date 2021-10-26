import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

# autoencoder


class CompressionNet(Layer):
    def __init__(self, hidden_layer_size, input_size):
        # latent_layer_size: list of hidden layer size
        # input_size: int
        super(CompressionNet, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.latent_dim = hidden_layer_size[-1]
        self.input_size = input_size
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.Input(shape=(self.input_size,)))
        for size in self.hidden_layer_size[:-1]:
            self.encoder.add(tf.keras.layers.Dense(size, activation='tanh'))
        self.encoder.add(tf.keras.layers.Dense(self.latent_dim))
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.Input(shape=(self.latent_dim,)))
        for size in self.hidden_layer_size[::-1][1:]:
            self.decoder.add(tf.keras.layers.Dense(size, activation='tanh'))
        self.decoder.add(tf.keras.layers.Dense(self.input_size))

    def reconstruction_error_feature(self, x):
        # This is a part of estimation net input
        # In original paper, relative Euclidean distance and
        # cosine similarity is considered
        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        hidden_state = self.encoder(x)
        x_new = self.decoder(hidden_state)
        # relative Euclidean distance
        loss_m = euclid_norm(x - x_new) / euclid_norm(x)
        # cosine similarity
        loss_c = tf.reduce_sum(x * x_new, axis=1) / \
            (euclid_norm(x) * euclid_norm(x_new))
        return tf.concat([loss_m[:, None], loss_c[:, None]], axis=1)

    def compute_loss(self, x):
        mse = tf.keras.losses.MeanSquaredError()
        hidden_state = self.encoder(x)
        x_new = self.decoder(hidden_state)
        loss = mse(x, x_new)
        return loss

    def call(self, inputs):
        hidden_state = self.encoder(inputs)
        # reconstruction_error = self.reconstruction_error_feature(inputs)
        # features = tf.concat([hidden_state, reconstruction_error], axis=1)
        # return features
        return hidden_state


class EstimationNet(Layer):
    def __init__(self, hidden_layer_size, input_size):
        # latent_layer_size: list of hidden layer size
        # input_size: int
        super(EstimationNet, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.inference = tf.keras.Sequential()
        self.inference.add(tf.keras.Input(shape=(self.input_size,)))
        for size in self.hidden_layer_size[:-1]:
            self.inference.add(tf.keras.layers.Dense(size, activation='tanh'))
        self.inference.add(tf.keras.layers.Dense(
            self.hidden_layer_size[-1], activation='softmax'))

    def call(self, inputs):
        outputs = self.inference(inputs)
        return outputs


class GMM:
    def __init__(self, z, gamma):
        # z : tf.Tensor, shape (n_samples, n_features)
        # gamma : tf.Tensor, shape (n_samples, n_component)
        gamma_sum = tf.reduce_sum(gamma, axis=0)  # shape (n_component, )
        self.phi = tf.reduce_mean(gamma, axis=0)  # shape (n_component, )
        # shape (n_component, n_features)
        self.mu = tf.matmul(gamma, z, transpose_a=True) / gamma_sum[:, None]
        z_centered = tf.sqrt(gamma[:, :, None]) * \
            (z[:, None, :] - self.mu[None, :, :])
        self.sigma = tf.einsum(
            'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:, None, None]
        # Calculate a cholesky decomposition of covariance in advance
        n_features = z.shape[1]
        min_vals = tf.linalg.diag(
            tf.ones(n_features, dtype=tf.float32)) * 1e-6
        self.L = tf.linalg.cholesky(self.sigma + min_vals[None, :, :])

    def compute_prob(self, z):
        # Instead of inverse covariance matrix, exploit cholesky decomposition
        # for stability of calculation.
        # shape (n_samples, n_component, n_features)
        z_centered = z[:, None, :] - self.mu[None, :, :]
        v = tf.linalg.triangular_solve(
            self.L, tf.transpose(z_centered, [1, 2, 0]))

        # log(det(Sigma)) = 2 * sum[log(diag(L))]
        log_det_sigma = 2.0 * \
            tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

        # To calculate prob, use "log-sum-exp" (different from orginal paper)
        d = z.shape[1]
        logits = tf.math.log(self.phi[:, None]) - 0.5 * (tf.reduce_sum(
            tf.square(v), axis=1) + d * tf.math.log(2.0 * np.pi) +
            log_det_sigma[:, None])
        prob = - tf.reduce_logsumexp(logits, axis=0)
        return prob

    def energy_loss(self, z):
        return tf.reduce_mean(self.compute_prob(z))

    def diag_loss(self):
        diag_loss = tf.reduce_sum(
            tf.divide(1, tf.linalg.diag_part(self.sigma)))
        return diag_loss


class DAGMMnet(Model):
    def __init__(self, compress_hidden_layer,
                 estimate_hidden_layer, input_size):
        super(DAGMMnet, self).__init__()
        self.compress_hidden_layer = compress_hidden_layer
        self.latent_dim = compress_hidden_layer[-1]
        self.estimate_hidden_layer = estimate_hidden_layer
        self.input_size = input_size
        self.compress_net = CompressionNet(
            self.compress_hidden_layer, self.input_size)
        # +2 means reconstruct features
        self.estimate_net = EstimationNet(
            self.estimate_hidden_layer, self.latent_dim)

    def compute_loss(self, x):
        lambda1 = 0.1
        lambda2 = 0.005
        z = self.compress_net(x)
        gamma = self.estimate_net(z)
        gmm_output = GMM(z, gamma)
        loss_1 = self.compress_net.compute_loss(x)
        loss_2 = gmm_output.energy_loss(z)
        loss_3 = gmm_output.diag_loss()
        return loss_1 + lambda1 * loss_2 + lambda2 * loss_3

    def call(self, x):
        z = self.compress_net(x)
        gamma = self.estimate_net(z)
        gmm_output = GMM(z, gamma)
        result = gmm_output.compute_prob(z)
        return result, z


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as t:
        loss = model.compute_loss(x)
        gradient = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))


def train(dataset, model, epochs, optimizer):
    for epoch in range(epochs):
        start = time.time()
        for data in dataset:
            train_step(model, data, optimizer)
        batch_num = 0
        total_loss = 0
        for data in dataset:
            batch_num += 1
            total_loss += model.compute_loss(data)
        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))
        print('Loss: {}'.format(total_loss/batch_num))
