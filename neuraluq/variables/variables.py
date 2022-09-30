# There are in total three types of variables considered in this package:
# 1. Trainable type: regular traced variables, work for standard deterministic neural network
# training, e.g. one `tf.Variable`` is used to define one variable.
# 2. Samplable type: untraced variables, treated like common tensors, e.g. one `tf.Tensor`
# is used to define one variable, traced and collected by sampling methods, e.g. MCMC.
# 3. Variational type: traced variables, each of which is defined by a parameterization of one
# pre-set distribution, work for standard variational inference training, e.g. multiple
# `tf.Variable`s are used to parameterize the distribution.


class Variable:
    """Base class for all variables."""

    def __init__(self):
        self._num_tensors = None
        self._trainable_variables = []

    @property
    def num_tensors(self):
        """Returns the total number of tensors, i.e. variables, in this `Variable."""
        return self._num_tensors

    @property
    def trainable_variables(self):
        """Returns `Variable`'s trainable variables."""
        return self._trainable_variables

    @trainable_variables.setter
    def trainable_variables(self):
        raise AttributeError("can't set this attribute")


class _Trainable(Variable):
    """
    Base class for all trainable variables, e.g. `tf.Variable`, used for setting up and 
    conventional machine learning framework.

    Typically, a `Trainable` variable has the following two components:
        1. trainable variables, as a list of trainable variables in certain order, e.g. 
            a list of weights + a list of biases for neural networks
        2. a function named `losses` to compute the regularization loss, as method.
    """

    def __init__(self):
        super().__init__()

    def __add__(self, v):
        """Adds two Trainables by adding their trainable variables."""
        new = _Trainable()
        new._trainable_variables = self.trainable_variables + v.trainable_variables
        new._num_tensors = self.num_tensors + v.num_tensors
        return new

    @property
    def losses(self):
        return []


class _Samplable(Variable):
    """
    Base class for all samplable variables, i.e. variables that are capable of being sampled. 
    Currently, we only consider MCMC-type sampling methods.

    Typically, a `Sampling` variable has the following three components:
        1. initial values, as a list of tensors in certain order, e.g. `tf.Tensor`, and 
            with shapes of [1, ...]. That is, three-dimension tensors with the first dimension 
            equal to 1.
        2. a function named `log_prob` to compute the log probability density, as a method
        3. a functoin named `sample` to sample with respect to pre-defined distributions. Note 
            that this function is not necessary to perform posterior sampling
    """

    def __init__(self):
        super().__init__()
        self._initial_values = None

    def __add__(self, v):
        """Adds two Sampables by adding their initial values and rewriting log probability function."""
        new = _Samplable()
        new._initial_values = self.initial_values + v.initial_values
        new._num_tensors = self.num_tensors + v.num_tensors

        def _log_prob(samples):
            return self.log_prob(samples[: self.num_tensors]) + v.log_prob(
                samples[self.num_tensors :]
            )

        def _sample(sample_shape=[]):
            return self.sample(sample_shape) + v.sample(sample_shape)

        new.log_prob = _log_prob
        new.sample = _sample
        return new

    @property
    def initial_values(self):
        return self._initial_values

    def log_prob(self, samples):
        """
        Computes the log probability of `samples`. Note that the order of `samples` should be consistent 
        with the output of `sample` method.
            Args:
                samples (list of tensors): The list of samples. The first dimension of each tensor is the number
                    of samples, and it needs to be the same across all tensors.
            Returns:
                log_prob (tensor): The log probability of `samples` in vector format.
        """
        raise NotImplementedError("log_prob to be implemented.")

    def sample(self, sample_shape=[]):
        """
        Samples `sample_shape` realizations of the variable.

            Args:
                sample_shape (int): The number of samples.
            Returns:
                samples (list of tensors): The list of samples on neural networks.
        """
        raise NotImplementedError("sample to be implemented.")


class _Variational(Variable):
    """
    Base class for all variational variables, for variational inference (VI).

    The key idea behind VI is that we parameterize distributions and then learn the parameters
    by optimization. 

    Four things are needed for a typical VI-type variable:
        1. distributions, as a list of distributions in certain order, e.g. tfp.distributions
        2. variables used to parameterized the distributions, as a list of variables in consistent
            order.
        3. a function named `log_prob` to compute the log probability density, as a method
        4. a functoin named `sample` to sample with respect to pre-defined distributions. 
    """

    def __init__(self):
        super().__init__()
        self._trainable = False

    @property
    def trainable(self):
        return self._trainable

    def log_prob(self, samples):
        """
        Computes the log probability of `samples`. Note that the order of `samples` should be consistent 
        with the output of `sample` method.
            Args:
                samples (list of tensors): The list of samples. The first dimension of each tensor is the number
                    of samples, and it needs to be the same across all tensors.
            Returns:
                log_prob (tensor): The log probability of `samples` in vector format.
        """
        raise NotImplementedError("log_prob to be implemented.")

    def sample(self, sample_shape=[]):
        """
        Samples `sample_shape` realizations of the variable.

            Args:
                sample_shape (int): The number of samples.
            Returns:
                samples (list of tensors): The list of samples on neural networks.
        """
        raise NotImplementedError("sample to be implemented.")
