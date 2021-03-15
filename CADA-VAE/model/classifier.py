import tensorflow as tf
from tensorflow.keras.layers import Dense

from model.model_utils import get_initializer


class LinearClassifier(tf.keras.Model):

    def __init__(self, config, num_classes, **kwargs):
        super(LinearClassifier, self).__init__(**kwargs)
        self.num_classes = num_classes

        kernel_initializer = None
        if config.kernel_initializer:
            kernel_initializer = get_initializer(config.kernel_initializer)
        self.layer = Dense(self.num_classes, activation=None, kernel_initializer=kernel_initializer)

    def call(self, x):
        logits = self.layer(x)
        return logits
