from .config import backend_name, tf


class Process:
    """
    Class for all processes, both deterministic and stochastic ones.
    Under all circumstances, surrogate and at least one of prior and posterior have to be provided. There
    are in total three scenarios:
        1. Ensemble-based models/inferences: prior is `None` and posterior is not. In this case, posterior,
            although named statisitically, is in fact a set of trainable variables, which are treated in 
            the same way as in conventional machine learning problems.
        2. MCMC-based models/inference: prior is not `None` and posterior is. In this case, Bayesian model
            is considered and sampling method is deployed. Hence, log of posterior probability density is
            computed by the model according to the prior and likelihood(s), so that posterior is not needed.
        3. VI-based models/inference: prior and posterior are not `None`. Variational inference, although 
            belongs to Bayesian inference, treats unknown variables differently in the way that variables are 
            random with distributions paramterized by trainable variables, and it needs to maximize the ELBO.
            Hence, both prior and parametrized posterior are required.

        Args:
            surrogate: The process's surrogate, whose arguments are the process's input, e.g. time and/or
                location, and the stochasticity, represented in one or multiple samples.
            prior: The process's prior distribution.
            posterior: The process's posterior distribution.

    """

    def __init__(self, surrogate, prior=None, posterior=None):
        self._surrogate = surrogate
        self._prior = prior  # None for ensemble-based method
        self._posterior = posterior  # None for MCMC method

        # Note: abuse of notations appears here. It seems that here we use the id of prior to identify
        # process. But in fact, we are instead identifying process's latent variables, e.g. parameters
        # of a neural network. Notice that there is no point identifying process using some index,
        # because users have direct access to all the processes. However, they don't have explicit control
        # over processes' latent variables. This setup is particularly useful when processes sharing the
        # same latent variables.
        # Another important thing to notice is that using prior distribution to identify latent variables
        # is accurate, even though they may also have explicitly defined posterior distribution. That is
        # because the fact that latent variables sharing the same prior distribution, implies that they
        # also share the same posterior distribution. Otherwise, they are not well-defined.
        # Lastly, for VI-based and Ensemble-based models, process does not need to be traced, because for
        # those cases, optimization is performed on trainable variables, which are are automatically traced
        # by the deep learning frameworks we are using, e.g. Tensorflow. However, for consistency reason,
        # its key is defined here, in similar ways. It may be deleted in the future.
        if prior is not None:
            # Bayesian inference: MCMC-type and VI-type
            self._key = id(prior)
            self._num_variables = prior.num_tensors
            self._stochastic = True
        else:
            # Ensemble-type
            self._key = id(posterior)
            self._num_variables = len(posterior.trainable_variables)
            self._stochastic = False
        # self._key = id(prior) if prior is not None else id(posterior)

        # self._num_variables = prior.num_tensors if prior is not None else len(posterior.trainable_variables)
        # self._type = "stochastic" if prior is None else "deterministic"
        self._trainable_variables = (
            None if posterior is None else posterior.trainable_variables
        )

    @property
    def key(self):
        """Returns hashable key."""
        return self._key

    @property
    def num_variables(self):
        """Returns `Variable`'s posterior distribution."""
        return self._num_variables

    @property
    def surrogate(self):
        return self._surrogate

    @property
    def prior(self):
        return self._prior

    @property
    def posterior(self):
        return self._posterior

    @property
    def trainable_variables(self):
        return self._trainable_variables

    @property
    def stochastic(self):
        return self._stochastic


class GlobalProcesses:  # AllProcesses
    """Collection of all created processes. Works as a Python dictionary/list"""

    def __init__(self):
        self._processes = {}
        self._total_num_variables = 0
        self._initial_values = []
        self._posterior_sample_fns = []

    # def __add__(self, other):
    #     if not isinstance(other, Process):
    #         raise TypeError("can't add {}".format(str(other)))
    #     self._processes += other
    #     return self

    def update(self, processes):
        if not isinstance(processes, list):
            processes = [processes]
        # check if all processes have the right types. If not, do not trace any
        # input process.
        for p in processes:
            if (
                isinstance(p, Process) is False
                and isinstance(p, DeterministicProcess) is False
            ):
                raise TypeError("{} is not a well-defined process.".format(str(p)))
        # store and trace
        for p in processes:
            if p.key not in self.processes:
                self._processes.update({p.key: p})
                self._total_num_variables += p.num_variables
                if p.stochastic is True:
                    if p.posterior is None:
                        self._initial_values += p.prior.initial_values
                    else:
                        self._posterior_sample_fns += [p.posterior.sample]
                    # else:
                    #     if len(self._posterior_sample_fns) != 0:
                    #         # either all processes have posterior distributions, or none does
                    #         raise ValueError(
                    #             "either all processes have posterior distributions, or none does"
                    #         )

    def keys(self):
        return self.processes.keys()

    def assign(self, var_list):
        """
        Decomposes a list, which is the list of samples of all variables, into
        multiple disjoint lists, each one of which belongs to a random
        variable/tensor/neural network. And assigns them accordingly.

            Args:
                var_list (list of tensors): The list of samples.

            Returns:
                sublists (dict): The deoomposed list.
        """
        if self.total_num_variables != len(var_list):
            raise ValueError("inconsistent number of variables")
        beg = 0
        sublists = {}
        for key, value in self.processes.items():
            end = beg + value.num_variables
            sublists[key] = var_list[beg:end]
            beg = end
        return sublists

    @property
    def processes(self):
        """Returns all traced processes."""
        return self._processes

    @property
    def total_num_variables(self):
        """Returns the total number of variables from all traced processes."""
        return self._total_num_variables

    @property
    def initial_values(self):
        """Returns the initial values of all traced variables, for MCMC."""
        return self._initial_values

    @property
    def posterior_sample_fns(self):
        """Returns the list of sampling functions, subject to posterior distributions."""
        return self._posterior_sample_fns


# class Process:
#     """Class for all processes, deterministic ones and stochastic ones."""

#     def __init__(self, surrogate, prior, posterior=None):
#         self._surrogate = surrogate
#         self._prior = prior
#         self._posterior = posterior  # None for MCMC method, parameterized distribution for MFVI method
#         self._type = "stochastic"

#         # Note: abuse of notations appears here. It seems that here we use the id of prior to identify
#         # process. But in fact, we are instead identifying process's latent variables, e.g. parameters
#         # of a neural network. Notice that there is no point identifying process using some index,
#         # because users have direct access to all the processes. However, they don't have explicit control
#         # over processes' latent variables. This setup is particularly useful when processes sharing the
#         # same latent variables.
#         # Another important thing to notice is that using prior distribution to identify latent variables
#         # is accurate, even though they may also have explicitly defined posterior distribution. That is
#         # because the fact that latent variables sharing the same prior distribution, implies that they
#         # also share the same posterior distribution. Otherwise, they are not well-defined.
#         self._key = id(prior)

#         self._num_variables = prior.num_tensors

#     @property
#     def key(self):
#         """Returns hashable key."""
#         return self._key

#     @property
#     def num_variables(self):
#         """Returns `Variable`'s posterior distribution."""
#         return self._num_variables

#     @property
#     def surrogate(self):
#         return self._surrogate

#     @property
#     def prior(self):
#         return self._prior

#     @property
#     def posterior(self):
#         return self._posterior

#     @property
#     def type(self):
#         return self._type


# class DeterministicProcess:
#     """Class for all deterministic processes, i.e. functions."""

#     def __init__(self, surrogate, variable):
#         self._surrogate = surrogate
#         self._variable = variable
#         self._trainable_variables = variable.trainable_variables
#         # TODO: key is to be changed
#         self._key = id(
#             variable
#         )  # warning: the key is the address of a variable instance.
#         self._num_variables = len(self.trainable_variables)
#         self._type = "deterministic"

#     @property
#     def key(self):
#         """Returns hashable key."""
#         return self._key

#     @property
#     def num_variables(self):
#         """Returns `Variable`'s posterior distribution."""
#         return self._num_variables

#     @property
#     def surrogate(self):
#         return self._surrogate

#     @property
#     def variable(self):
#         return self._variable

#     @property
#     def trainable_variables(self):
#         return self._trainable_variables

#     @property
#     def type(self):
#         return self._type
