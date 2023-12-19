import numpy as np


from . import config
from .process import GlobalProcesses
from .config import backend_name, tf
from . import surrogates


class Model:
    """
    A model collects processes and likelihoods, and then performs inferences.
    """

    def __init__(self, processes, likelihoods):
        # create a instance of GlobalProcess, for control over all used processes
        self.global_processes = GlobalProcesses()
        self.global_processes.update(processes)

        self.processes = processes
        self.likelihoods = likelihoods

        if backend_name == "tensorflow.compat.v1":
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        elif backend_name == "tensorflow":
            # TODO: support Tensorflow 2
            self.sess = None
        else:
            raise ValueError("Backend {} is not supported".format(backend_name))
        print("Supporting backend " + str(backend_name) + "\n")

        self.trainable_variables = []
        for p in processes:
            # warning: use key to collect trainable_variables
            if p.trainable_variables is not None:
                self.trainable_variables += p.trainable_variables

        self.method = None

    def compile(self, method):
        """Compiles the Bayesian model with a method"""
        # different method has different way of compiling
        print("Compiling a {} method\n".format(method.method_type))
        if method.method_type == "MCMC":
            self._compile_mcmc(method)
        elif method.method_type == "VI":
            self._compile_vi(method)
        elif method.method_type == "Ensemble":
            self._compile_ens(method)
        else:
            raise NotImplementedError(
                "Support for {} to be implemented.".format(method.method_type)
            )

    def run(self):
        """Performs posterior estimate over the Bayesian model with a compiled method"""
        if self.method is None:
            raise ValueError("Model has not been compiled with a method.")
        samples = self.method.sampling(self.sess)
        return samples

    def predict(self, inputs, samples, processes, pde_fn=None):
        """
        Performs prediction over `processes` at `inputs`, given posterior samples stored in
        `samples` as a Python dictionary. Every process in `processes` needs to be stored in
        the model.
        If `pde_fn` is not None, then the prediction is on the quantity defined by `pde_fn`,
        and `processes` has be stored in order in a list such that, together with `inputs`,
        it matches the arguments of `pde_fn`.
        If `pde_fn` is None, then the prediction is on all processes in `processes`.
        """
        # TODO: support Variational and Trainable setups.
        # convert to tensor first, for tensorflow.compat.v1 and further computation on
        # derivatives.
        if isinstance(inputs, (list, tuple)):
            inputs = [tf.constant(e, config.dtype) for e in inputs]
        else:
            inputs = tf.constant(inputs, config.dtype)

        tf_samples = [tf.constant(s, tf.float32) for s in samples]
        # assign samples to processes
        samples_dict = self.global_processes.assign(tf_samples)
        if pde_fn is None:
            # no PDE
            _predictions = [
                p.surrogate(inputs, samples_dict[p.key])[1] for p in processes
            ]
        else:
            # PDE: reshape and tile the inputs, from the first process
            p_inp = None
            args = []
            for p in processes:
                if p_inp is None:
                    p_inp, p_out = p.surrogate(inputs, samples_dict[p.key])
                else:
                    _, p_out = p.surrogate(p_inp, samples_dict[p.key])
                    if isinstance(p.surrogate, surrogates.Identity):
                        if len(p_out.shape) == 1:
                            p_out = p_out[..., None, None]
                        elif len(p_out.shape) == 2:
                            p_out = p_out[..., None]
                args += [p_out]
            _predictions = [pde_fn(p_inp, *args)]

        if backend_name == "tensorflow.compat.v1":
            predictions = self.sess.run(_predictions)
        elif backend_name == "tensorflow":
            predictions = [v.numpy() for v in _predictions]

        return predictions

    def _compile_mcmc(self, method):
        """Compiles the model with a MCMC-type inference method"""

        # build log posterior function
        def log_posterior_fn(*var_list):
            # The computation of log probabilistic density of posterior distribution is
            # decomposed into three steps.
            # Step 1: assign var_list to corresponding processes, e.g. BNNs, constants.
            # Step 2: for each process, compute its prior
            # Step 3: for each likelihood, compute its likelihood
            global_var_dict = self.global_processes.assign(var_list)
            log_prior = []
            for key, p in self.global_processes.processes.items():
                log_prior += [tf.reduce_sum(p.prior.log_prob(global_var_dict[key]))]
            log_prior = tf.reduce_sum(log_prior)

            log_likeli = tf.reduce_sum(
                [lh.log_prob(global_var_dict) for lh in self.likelihoods]
            )
            return log_prior + log_likeli

        # compile the method
        method.make_sampler(log_posterior_fn, self.global_processes.initial_values)
        # assign the method
        self.method = method

    def _compile_vi(self, method):
        """Compiles the model with a VI-type inference method."""

        # build negative elbo function
        def neg_elbo_fn(batch_size):
            # The computation of negative ELBO is decomposed into three steps
            # Step 1: generate a batch of samples, subject to (parameterized)
            # current posterior distributions
            # Step 2: for each process, compute its prior and posterior
            # Step 3: for each likelihood, comptue its likelihood
            global_var_dict, log_posterior, log_prior = {}, 0.0, 0.0
            for key, p in self.global_processes.processes.items():
                _samples = p.posterior.sample(sample_shape=batch_size)
                if len(_samples) == 1 and len(_samples[0].shape) == 1:
                    # for constant's samples, reshape them to shapes [batch_size, 1, 1]
                    # in the training
                    # TODO: to be substituted with a smarter way
                    _samples = [tf.reshape(_samples[0], [-1, 1, 1])]
                global_var_dict.update({key: _samples})
                log_posterior += tf.reduce_sum(p.posterior.log_prob(_samples))
                log_prior += tf.reduce_sum(p.prior.log_prob(_samples))

            log_likeli = tf.reduce_sum(
                [lh.log_prob(global_var_dict) for lh in self.likelihoods]
            )
            neg_elbo = (log_posterior - log_prior - log_likeli) / batch_size
            return neg_elbo

        # compile the method
        method.make_sampler(self.sess, neg_elbo_fn, self.processes)
        # assign the method
        self.method = method

    def _compile_ens(self, method):
        """Compiles the model with an ensemble-type inference method."""

        def loss_fn():
            losses = [loss.loss(training=True) for loss in self.likelihoods]
            regularization = [tf.reduce_sum(p.posterior.losses) for p in self.processes]
            return tf.reduce_sum(losses) + tf.reduce_sum(regularization)

        # compile the method
        method.make_sampler(self.sess, loss_fn, self.trainable_variables)
        # assign the method
        self.method = method
