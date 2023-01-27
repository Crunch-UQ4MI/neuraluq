import numpy as np
from .config import tf, tfp


class Loss:
    """Base class for all likelihoods and losses"""

    def __init__(self):
        self._inputs = None
        self._targets = None
        self._processes = []
        self._in_dims = None
        self._out_dims = None
        self._pde = None
        self._trainable_variables = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def processes(self):
        return self._processes

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def pde(self):
        return self._pde


class MSE(Loss):
    """Mean-squared loss function over all observations."""

    def __init__(
        self,
        inputs,
        targets,
        processes,
        pde=None,
        in_dims=None,
        out_dims=None,
        multiplier=1.0,
    ):
        super().__init__()
        # set attributes
        self._inputs = tf.constant(inputs, tf.float32)
        self._targets = tf.constant(targets, tf.float32)
        if not isinstance(processes, list):
            processes = [processes]
        self._processes = processes

        self._in_dims = in_dims if in_dims is not None else len(processes) * [None]
        self._out_dims = out_dims if out_dims is not None else len(processes) * [None]

        self._pde = pde

        self.multiplier = multiplier

    def get_batch(self, batch_size):
        # TODO: find more efficient shuffling methods
        idx = tf.random_shuffle(self.idx)[:batch_size]
        batch_inputs = tf.gather(self.inputs, idx, axis=0)
        batch_targets = tf.gather(self.targets, idx, axis=0)
        return batch_inputs, batch_targets

    def loss(self, training=False):
        """Return regular mean-squared error."""
        # TODO: support mini-batch computation
        inputs = self.inputs
        targets = self.targets
        if self.pde is None:
            for p, in_dim, out_dim in zip(self.processes, self.in_dims, self.out_dims):
                p_inp = inputs if in_dim is None else tf.gather(inputs, in_dim, axis=-1)
                _, out = p.surrogate(inputs, p.trainable_variables)
                p_out = out if out_dim is None else tf.gather(out, out_dim, axis=-1)
            out = p_out
        else:
            args = []
            p_inp = None
            # TODO: support in_dims and out_dims
            for p in self.processes:
                if p_inp is None:
                    p_inp, p_out = p.surrogate(inputs, p.trainable_variables)
                else:
                    _, p_out = p.surrogate(p_inp, p.trainable_variables)
                args += [p_out]
            out = self.pde(p_inp, *args)
        return self.multiplier * tf.reduce_mean((out - targets) ** 2)


class Normal(Loss):
    """Independent Normal distribution for likelihood over all observations"""

    def __init__(
        self,
        inputs,
        targets,
        processes,
        pde=None,
        in_dims=None,
        out_dims=None,
        sigma=0.1,
    ):
        """Initializes distribution"""
        super().__init__()
        # set attributes
        self._inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        self._targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        if not isinstance(processes, list):
            processes = [processes]
        self._processes = processes

        self._in_dims = in_dims if in_dims is not None else len(processes) * [None]
        self._out_dims = out_dims if out_dims is not None else len(processes) * [None]

        self._pde = pde
        # build the distribution
        self.sigma = tf.constant(sigma, dtype=tf.float32)

        def _log_prob(x):
            return (
                -tf.math.log(tf.math.sqrt(2 * np.pi) * self.sigma)
                - x ** 2 / self.sigma ** 2 / 2
            )

        self._log_prob = _log_prob

    def log_prob(self, global_var_dict):
        # it's necessary to declare inp here, to prevent bugs in MCMC methods
        inputs = self.inputs
        targets = self.targets
        if self.pde is None:
            # for direct observation
            for p, in_dim, out_dim in zip(self.processes, self.in_dims, self.out_dims):
                # TODO: support multiple observations scenario
                p_inp = inputs if in_dim is None else tf.gather(inputs, in_dim, axis=-1)
                _, out = p.surrogate(p_inp, global_var_dict[p.key])
                p_out = out if out_dim is None else tf.gather(out, out_dim, axis=-1)
            out = p_out
        else:
            # for observation through PDE
            args = []  # arguments of the PDE function
            p_inp = None
            # TODO: support in_dims and out_dims
            for p in self.processes:
                if p_inp is None:
                    p_inp, p_out = p.surrogate(inputs, global_var_dict[p.key])
                else:
                    _, p_out = p.surrogate(p_inp, global_var_dict[p.key])
                args += [p_out]
            out = self.pde(p_inp, *args)
        return tf.reduce_sum(
            -tf.math.log(self.sigma) - (out - targets) ** 2 / 2 / self.sigma ** 2,
            axis=[-1, -2],
        )
        # return tf.reduce_sum(self.dist.log_prob(out-targets), axis=[-1, -2])

    def get_fn_list(self, global_var_dict):
        """
        Gets corresponding samples/values for this likelihood, from global samples/variables,
        and then forms a list of functions to compute the probabilistic density of likelihood
        distribution.
        """
        # TODO: to be deleted
        fn_list = []
        for key, p in self.processes.items():
            fn_list += [lambda inp: p.surrogate(inp, global_var_dict[key])[1]]
        return fn_list


class MSE_operator(Loss):
    """
    Mean-squared loss function over all observations in operator learning, specifically 
    DeepONet.
    Currently there are two main differences between operator learning and conventional 
    machine learning:
    1. The input to DeepONet has two elements, one for trunk net and one for branch net.
    2. DeepONet supports minibatch training.

    MSE_operator only supports single process.

    Args:
        inputs (list or tuple of two tensors): The inputs (training data) to the DeepONet. 
            The first element is the input to the trunk net and the second element is the 
            input to the branch net.
        targets (tensor): The outputs (training data) of the DeepONet.
    """

    def __init__(self, inputs, targets, processes, batch_size=None):
        super().__init__()
        # set attributes
        self._inputs = [tf.constant(e, tf.float32) for e in inputs]
        self._targets = tf.constant(targets, tf.float32)
        if not isinstance(processes, list):
            processes = [processes]
        self._processes = processes

        self.batch_size = 1 if batch_size is None else batch_size
        self.idx = np.arange(targets.shape[0])

    def get_batch(self, batch_size):
        # TODO: find more efficient shuffling methods
        idx = tf.random_shuffle(self.idx)[:batch_size]
        batch_inputs = self.inputs[0], tf.gather(self.inputs[1], idx, axis=0)
        batch_targets = tf.gather(self.targets, idx, axis=0)
        return batch_inputs, batch_targets

    def loss(self, training=True):
        """Return regular mean-squared error."""
        batch_size = self.batch_size
        batch_inputs, batch_targets = self.get_batch(batch_size)
        # TODO: support multiple processes
        p = self.processes[0]
        _, out = p.surrogate(batch_inputs, p.trainable_variables)
        return tf.reduce_mean((out - batch_targets) ** 2)
