import time

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class Attention(Layer):
    def __init__(self, input_size, hidden_size, output_size, is_gated=True):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.is_gated = is_gated
        if self.is_gated:
            self.V = tf.keras.layers.Dense(
                hidden_size, activation='tanh', use_bias=False)
            self.U = tf.keras.layers.Dense(
                hidden_size, activation='sigmoid', use_bias=False)
            self.w = tf.keras.layers.Dense(output_size, use_bias=False)

        else:
            self.V = tf.keras.layers.Dense(
                hidden_size, activation='tanh', use_bias=False)
            self.w = tf.keras.layers.Dense(output_size, use_bias=False)

    def call(self, inputs):
        # attention_weight (batch_size, output_size)
        if self.is_gated:
            attention_weight = self.w(
                tf.multiply(self.V(inputs), self.U(inputs)))
            attention_weight = tf.nn.softmax(attention_weight, axis=0)
            return attention_weight
        else:
            attention_weight = self.w(self.V(inputs))
            attention_weight = tf.nn.softmax(attention_weight, axis=0)
            return attention_weight


# feature_extract_layer = tf.keras.Sequential([
#     tf.keras.Input(shape=(28,28,1)),
#     tf.keras.layers.Conv2D(20, kernel_size=5, activation='relu'),
#     tf.keras.layers.MaxPool2D(strides=2),
#     tf.keras.layers.Conv2D(50, kernel_size=5, activation='relu'),
#     tf.keras.layers.MaxPool2D(strides=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(500, activation='relu')
# ])

# attention_layer = Attention(500, 128, 1)

# classifer_layer = tf.keras.layers.Dense(1, activation='sigmoid')

class MIL_attention(Model):
    def __init__(self, feature_extract_layer, attention_layer,
                 classifer_layer):
        super(MIL_attention, self).__init__()
        self.feature_extract = feature_extract_layer
        self.attention = attention_layer
        self.classifier = classifer_layer

    def call(self, inputs):
        # we consider samples in bag as a batch and set common batch size is 1
        # inputs = tf.squeeze(inputs)
        # inputs (bag_size, origin_feature_shape)
        extracted_feature = self.feature_extract(inputs)
        # extracted_feature (bag_size, feature_size)
        attention_weight = self.attention(extracted_feature)
        # attention_weight (bag_size, k)
        # we can set k > 1 to apply multi head attention mechanism
        pooling_feature = tf.matmul(
            attention_weight, extracted_feature, transpose_a=True)
        # pooling_feature (k, feature_size)
        prob = self.classifier(pooling_feature)
        return prob, attention_weight

    def compute_loss(self, data):
        inputs = data[0]
        label = data[1]
        prob, _ = self.call(inputs)
        crossentropy = tf.keras.losses.BinaryCrossentropy()
        loss = crossentropy(label, prob)
        pred = 1 if prob > 0.5 else 0
        return loss, pred != label


# @tf.function
def train_step(model, data, optimizer):
    with tf.GradientTape() as t:
        loss, _ = model.compute_loss(data)
        gradient = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))


def train(dataset, model, epochs, optimizer):
    for epoch in range(epochs):
        start = time.time()
        for data in dataset:
            train_step(model, data, optimizer)
        batch_num = 0
        total_loss = 0
        total_error = 0
        for data in dataset:
            batch_num += 1
            loss, error = model.compute_loss(data)
            total_loss += loss
            total_error += error
        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))
        print('Loss: {}'.format(total_loss/batch_num))
        print('Accuracy: {}'.format(1-total_error/batch_num))
