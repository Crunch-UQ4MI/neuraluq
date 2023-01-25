"""Parallelized fully-connected neural networks, to be used with Deep Ensemble method."""

from ..config import tf, dtype
from .variables import _Trainable


def glorot_normal(batch_size, shape):
    """
    Initializes multiple weights using Glorot Normal initializer, according to `shape`.

        Args:
            batch_size (int): The number of weights initialized.
            shape (array): The shape of weights initialized, which shoule be of 2-D.
        Returns:
            weights (tensor): The initialization of weights using the same initializer.
    """
    stddev = tf.math.sqrt(2 / tf.cast(tf.reduce_sum(shape), dtype))
    weights = stddev * tf.random.normal(shape=[batch_size] + shape)
    return weights


def glorot_uniform(batch_size, shape):
    """
    Initializes multiple weights using Glorot Uniform initializer, according to `shape`.

        Args:
            batch_size (int): The number of weights initialized.
            shape (array): The shape of weights initialized, which shoule be of 2-D.
        Returns:
            weights (tensor): The initialization of weights using the same initializer.
    """
    limit = tf.math.sqrt(6 / tf.cast(tf.reduce_sum(shape), dtype))
    weights = -limit + 2 * limit * tf.random.uniform(shape=[batch_size] + shape)
    return weights


class Trainable(_Trainable):
    """Trainable type variable of a parallelized fully-connected neural network."""

    def __init__(
        self, layers, num, regularizer=None,
    ):
        super().__init__()
        self._num_tensors = 2 * (len(layers) - 1)
        self.num = num

        self.regularizer = regularizer
        self.weights, self.biases = [], []
        for i in range(len(layers) - 1):
            # TODO: generalize to all other initializers
            self.weights += [
                tf.Variable(
                    glorot_normal(num, shape=[layers[i], layers[i + 1]]), dtype=dtype
                )
            ]
            self.biases += [
                tf.Variable(tf.zeros(shape=[num, 1, layers[i + 1]]), dtype=dtype)
            ]

        self._trainable_variables = self.weights + self.biases

    @property
    def losses(self):
        # Note: regularization is only performed on weights
        if self.regularizer is not None:
            return [self.regularizer(w) / self.num for w in self.weights]
        return []
