"""
Parallelized constants, to be used with parallelized fully-connected neural 
networks and Deep Ensemble method.
"""

from ..config import tf, dtype
from .variables import _Trainable


class Trainable(_Trainable):
    """Trainable type variable of a constant."""

    def __init__(
        self, value, num, shape=[1, 1], initializer=None, regularizer=None,
    ):
        super().__init__()
        self._num_tensors = 1
        self.num = num

        init = tf.keras.initializers.Constant(value=value)
        if initializer is not None:
            init = initializer
        self.value = tf.Variable(init(shape=[num] + shape), dtype=tf.float32)
        self.regularizer = regularizer

        self._trainable_variables = [self.value]

    @property
    def losses(self):
        if self.regularizer is not None:
            return [self.regularizer(v) / self.num for v in self.trainable_variables]
        return []
