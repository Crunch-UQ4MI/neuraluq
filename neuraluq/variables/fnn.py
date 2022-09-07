from ..config import tf, tfp
from .variables import _Samplable, _Trainable, _Variational


class Trainable(_Trainable):
    """Trainable type variable of a fully-connected neural network"""

    def __init__(
        self, layers, initializer=None, regularizer=None,
    ):
        super().__init__()
        self._num_tensors = 2 * (len(layers) - 1)
        # initializer for weights is glorot normal by defaults, or specified otherwise
        w_init = (
            tf.keras.initializers.glorot_normal()
            if initializer is None
            else initializer
        )
        b_init = tf.keras.initializers.zeros()
        self.regularizer = regularizer
        self.weights, self.biases = [], []
        for i in range(len(layers) - 1):
            self.weights += [
                tf.Variable(w_init(shape=[layers[i], layers[i + 1]]), dtype=tf.float32)
            ]
            self.biases += [
                tf.Variable(b_init(shape=[1, layers[i + 1]]), dtype=tf.float32,)
            ]

        self._trainable_variables = self.weights + self.biases

    @property
    def losses(self):
        # Note: regularization is only performed on weights
        if self.regularizer is not None:
            return [self.regularizer(w) for w in self.weights]
        return []


class Samplable(_Samplable):
    """
    Samplable type variable of a fully-connected neural network with independent
        Normal distributions.
    """

    def __init__(
        self, layers, mean, sigma=0.1, initializer=None,
    ):
        super().__init__()

        mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        self._num_tensors = 2 * (len(layers) - 1)
        # TODO: support other initializers
        w_init = tf.keras.initializers.zeros()
        b_init = tf.keras.initializers.zeros()

        self._initial_values = []
        self.dists = []
        for i in range(len(layers) - 1):
            shape = [layers[i], layers[i + 1]]
            # add one axis before axis 0, for MCMC sampler
            self._initial_values += [w_init(shape=shape, dtype=tf.float32)]
            _mean, _sigma = mean * tf.ones(shape=shape), sigma * tf.ones(shape=shape)
            self.dists += [tfp.distributions.Normal(loc=_mean, scale=_sigma)]
            # self.dists += [tfp.distributions.Laplace(loc=mean, scale=sigma)]
        for i in range(len(layers) - 1):
            shape = [1, layers[i + 1]]
            # add one axis before axis 0, for MCMC sampler
            self._initial_values += [b_init(shape=shape, dtype=tf.float32)]
            _mean, _sigma = mean * tf.ones(shape=shape), sigma * tf.ones(shape=shape)
            self.dists += [tfp.distributions.Normal(loc=_mean, scale=_sigma)]

    def log_prob(self, samples):
        # For NN, `samples` is a list of tensors, each one which is of shape [N, :, :], where N
        # is the sample size. Hence, the log probability should be of shape [N]
        _log_prob = tf.zeros(shape=[samples[0].shape[0]])
        for s, dist in zip(samples, self.dists):
            _log_prob += tf.reduce_sum(dist.log_prob(s), axis=[-1, -2])
        return _log_prob

    def sample(self, sample_shape=[]):
        return [dist.sample(sample_shape=sample_shape) for dist in self.dists]


class Variational(_Variational):
    """
    Variational type variable of a fully-connected neural network, parameterized by independently identically 
    distributed Normal random varibales, each of which has optimizable mean and standard deviation.
    """

    def __init__(self, layers, mean, sigma=0.1, initializer=None, trainable=False):
        super().__init__()

        mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        self._num_tensors = 2 * (len(layers) - 1)
        self._trainable = trainable
        # TODO: support customized initializations
        # all weights and biases have the same customized mean and standard deviation
        self.w_means, self.w_log_stds = [], []
        self.b_means, self.b_log_stds = [], []
        self.dists = []
        for i in range(len(layers) - 1):
            self.w_means += [
                tf.Variable(
                    mean * tf.ones(shape=[layers[i], layers[i + 1]]),
                    dtype=tf.float32,
                    trainable=trainable,
                )
            ]
            self.w_log_stds += [
                tf.Variable(
                    tf.math.log(sigma) * tf.ones(shape=[layers[i], layers[i + 1]]),
                    dtype=tf.float32,
                    trainable=trainable,
                )
            ]
            # TODO: it may fail for Tensorflow 2 because distributions are defined out of scope
            self.dists += [
                tfp.distributions.Normal(
                    loc=self.w_means[-1], scale=tf.math.exp(self.w_log_stds[-1])
                )
            ]
        for i in range(len(layers) - 1):
            self.b_means += [
                tf.Variable(
                    mean * tf.ones(shape=[1, layers[i + 1]]),
                    dtype=tf.float32,
                    trainable=trainable,
                )
            ]
            self.b_log_stds += [
                tf.Variable(
                    tf.math.log(sigma) * tf.ones(shape=[1, layers[i + 1]]),
                    dtype=tf.float32,
                    trainable=trainable,
                )
            ]
            # TODO: it may fail for Tensorflow 2 because distributions are defined out of scope
            self.dists += [
                tfp.distributions.Normal(
                    loc=self.b_means[-1], scale=tf.math.exp(self.b_log_stds[-1])
                )
            ]

        if trainable is True:
            self._trainable_variables = (
                self.w_means + self.w_log_stds + self.b_means + self.b_log_stds
            )
        # if trainable is True:
        #     self._trainable_variables = (
        #         self.w_means + self.b_means
        #     )

    def sample(self, sample_shape=[]):
        return [dist.sample(sample_shape=sample_shape) for dist in self.dists]

    def log_prob(self, samples):
        _log_prob = tf.zeros(shape=[samples[0].shape[0]])
        for s, dist in zip(samples, self.dists):
            _log_prob += tf.reduce_sum(dist.log_prob(s), axis=[-1, -2])
        return _log_prob


class MCD(_Variational):
    """
    Variational type variable of a fully-connected neural network, parameterized by independent identically 
    distributed Bernoulli random varibales, each of which has a fixed failing probability p and is multiplied 
    by an optimizable value. This is the Bayesian (variational) version of dropout for uncertainty. Notice that
    the output layer is not dropped, neither are biases.
    """

    def __init__(self, layers, dropout_rate=0.2, initializer=None, trainable=True):
        super().__init__()
        self._num_tensors = 2 * (len(layers) - 1)
        self._trainable = trainable
        if initializer is None:
            initializer = tf.keras.initializers.glorot_normal()

        self.dropout_rate = dropout_rate
        self.weights, self.biases = [], []
        for i in range(len(layers) - 1):
            self.weights += [
                tf.Variable(
                    initializer(shape=[layers[i], layers[i + 1]]),
                    dtype=tf.float32,
                    trainable=trainable,
                )
            ]
            self.biases += [
                tf.Variable(
                    initializer(shape=[1, layers[i + 1]]),
                    dtype=tf.float32,
                    trainable=trainable,
                )
            ]

        for v in self.weights + self.biases:
            if v.trainable:
                self._trainable_variables += [v]

    def sample(self, sample_shape=1):
        samples = []
        for i in range(len(self.weights) - 1):
            v = tf.tile(self.weights[i][None, ...], [sample_shape, 1, 1])
            drop_or_not = tf.cast(
                tf.random.uniform(shape=[sample_shape, 1, v.shape[-1]])
                > self.dropout_rate,
                tf.float32,
            )
            samples += [drop_or_not * v]
        # weight in the output layer is not dropped out.
        samples += [tf.tile(self.weights[-1][None, ...], [sample_shape, 1, 1])]
        # biases are not dropped out.
        for _v in self.biases:
            samples += [tf.tile(_v[None, ...], [sample_shape, 1, 1])]
        return samples

    def log_prob(self, samples):
        # Probability density depends only on p, which is a constant here. Hence, setting the density to be any
        # constant yields the same optimization results
        return 1.0
