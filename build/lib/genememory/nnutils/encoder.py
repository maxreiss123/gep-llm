import tensorflow as tf

"""
T-Net component of the pointNet https://github.com/charlesq34/pointnet/blob/master/models/transform_nets.py
"""

class TNet(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(TNet, self).__init__()
        self.moment = 0.99
        self.eps = 1e-3
        self.activation_func2 = tf.keras.activations.relu
        self.num_units = num_units
        self.conv1 = tf.keras.layers.Conv1D(filters=self.num_units, kernel_size=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=self.num_units * 2, kernel_size=1)
        self.conv3 = tf.keras.layers.Conv1D(filters=self.num_units * 4, kernel_size=1)
        self.fc1 = tf.keras.layers.Dense(units=self.num_units * 2)
        self.fc2 = tf.keras.layers.Dense(units=self.num_units)

        self.input_batch_norm = tf.keras.layers.BatchNormalization(axis=-1, epsilon=self.eps,
                                                                   momentum=self.moment)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=self.eps, momentum=self.moment)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=self.eps, momentum=self.moment)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=self.eps, momentum=self.moment)
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=self.eps, momentum=self.moment)
        self.bn5 = tf.keras.layers.BatchNormalization(axis=-1, epsilon=self.eps, momentum=self.moment)

    def call(self, x, training=False):
        x = self.input_batch_norm(x, training)
        x = self.activation_func2(self.bn1(self.conv1(x),training))
        x = self.activation_func2(self.bn2(self.conv2(x),training))
        x = self.activation_func2(self.bn3(self.conv3(x),training))
        x = tf.math.reduce_max(x, axis=1)  # global max pooling
        x = self.activation_func2(self.bn4(self.fc1(x), training))
        x = self.activation_func2(self.bn5(self.fc2(x), training))
        return x


def create_point_encoder(embed_dim):
    encoder = TNet(embed_dim)
    return encoder
