from ..config import tf, tfp
from .variables import _Trainable, _Samplable, _Variational


class Trainable(_Trainable):
    """Trainable type variable of a constant."""

    def __init__(
        self, value, shape=[], initializer=None, regularizer=None,
    ):
        super().__init__()
        self._num_tensors = 1
        init = tf.keras.initializers.Constant(value=value)
        if initializer is not None:
            init = initializer
        self.value = tf.Variable(init(shape=shape), dtype=tf.float32)
        self.regularizer = regularizer

        self._trainable_variables = [self.value]

    @property
    def losses(self):
        if self.regularizer is not None:
            return [self.regularizer(v) for v in self.trainable_variables]
        return []


class Samplable(_Samplable):
    """Samplable type variable of a constant with independent Normal distribution."""

    def __init__(self, mean, sigma=1.0, shape=[], initializer=None):
        super().__init__()

        mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        self._num_tensors = 1
        # TODO: support other initializers
        if initializer is None:
            init = tf.keras.initializers.zeros()
        else:
            init = initializer

        self._initial_values = [init(shape=shape)]
        if shape == []:
            # make sure constant has at least 1 dimension
            self._initial_values = [self.initial_values[0][None, ...]]
        self.dist = tfp.distributions.Normal(loc=mean, scale=sigma)

    def log_prob(self, samples):
        # Note: here, because a constant is considered, `samples` is a list of only
        # one element.
        return self.dist.log_prob(samples[0])

    def sample(self, sample_shape):
        return [self.dist.sample(sample_shape=sample_shape)]


class Variational(_Variational):
    """
    Variational type variable of a constant with Normal distribution. If the constant is a tensor,
    then each element of that tensor is independently identically distributed with Normal
    distribution.
    """

    def __init__(self, mean, sigma, shape=[], initializer=None, trainable=False):
        super().__init__()
        self._num_tensors = 1
        # mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        init = (
            initializer
            if initializer is not None
            else tf.keras.initializers.Constant(value=mean)
        )
        self.mean = tf.Variable(
            init(shape=shape), dtype=tf.float32, trainable=trainable
        )
        # TODO: support user specified initializer for standard deviation
        self.log_std = tf.Variable(
            tf.math.log(sigma) * tf.ones(shape=shape),
            dtype=tf.float32,
            trainable=trainable,
        )

        self.dist = tfp.distributions.Normal(
            loc=self.mean, scale=tf.math.exp(self.log_std)
        )

        self._trainable = trainable
        if trainable:
            self._trainable_variables = [self.mean, self.log_std]

    def sample(self, sample_shape=[]):
        return [self.dist.sample(sample_shape)]

    def log_prob(self, samples):
        return self.dist.log_prob(samples[0])


class VI_Laplace(_Variational):
    """
    VI-type variable of a constant with Laplace distribution. If the constant is a tensor,
    then each element of that tensor is independently identically distributed with Laplace
    distribution.
    """

    def __init__(self, mean, sigma, shape=[], initializer=None, trainable=False):
        super().__init__()
        self._num_tensors = 1
        # mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        init = (
            initializer
            if initializer is not None
            else tf.keras.initializers.Constant(value=mean)
        )
        self.mean = tf.Variable(
            init(shape=shape), dtype=tf.float32, trainable=trainable
        )
        # TODO: support user specified initializer for standard deviation
        self.log_std = tf.Variable(
            tf.math.log(sigma) * tf.ones(shape=shape),
            dtype=tf.float32,
            trainable=trainable,
        )

        self.dist = tfp.distributions.Laplace(
            loc=self.mean, scale=tf.math.exp(self.log_std)
        )

        self._trainable = trainable
        if trainable:
            self._trainable_variables = [self.mean, self.log_std]

    def sample(self, sample_shape=[]):
        return self.dist.sample(sample_shape)

    def log_prob(self, samples):
        return self.dist.log_prob(samples)


class VI_Normal(_Variational):
    """VI-type variable of a constant with independent truncated Normal distributions."""

    def __init__(self, mean, sigma, shape=[], initializer=None, trainable=False):
        super().__init__()
        self._num_tensors = 1
        # mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        init = (
            initializer
            if initializer is not None
            else tf.keras.initializers.Constant(value=mean)
        )
        self.mean = tf.Variable(
            init(shape=shape), dtype=tf.float32, trainable=trainable
        )
        # TODO: support user specified initializer for standard deviation
        self.log_std = tf.Variable(
            tf.math.log(sigma) * tf.ones(shape=shape),
            dtype=tf.float32,
            trainable=trainable,
        )

        self.dist = tfp.distributions.TruncatedNormal(
            loc=self.mean, scale=tf.math.exp(self.log_std), low=-1, high=3,
        )

        self._trainable = trainable
        if trainable:
            self._trainable_variables = [self.mean, self.log_std]

    def sample(self, sample_shape=[]):
        return self.dist.sample(sample_shape)

    def log_prob(self, samples):
        return self.dist.log_prob(samples)
