"""This file contains MCMC sampling inference methods, e.g. Hamiltonian Monte Carlo."""


import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np


from .inference import Inference
from ..config import backend_name


class MCMC(Inference):
    """Base class for all MCMC inference methods."""

    def __init__(self):
        super().__init__()
        self._params = {"seed": None}
        self.method_type = "MCMC"
        self.mcmc_kernel = None
        self.trace_fn = lambda _, pkr: pkr

    def set_sampler(self, init_state):
        if self.mcmc_kernel is None:
            raise ValueError("MCMC kernel is not found.")

        def sampler():
            samples, results = tfp.mcmc.sample_chain(
                num_results=self.params["num_samples"],
                num_burnin_steps=self.params["num_burnin"],
                current_state=init_state,
                kernel=self.mcmc_kernel,
                trace_fn=self.trace_fn,
                seed=self.params["seed"],
            )
            return samples, results

        # set sampler
        if backend_name == "tensorflow.compat.v1":
            self.sampler = sampler
        elif backend_name == "tensorflow":
            # tf.function makes the function executed in graph mode
            self.sampler = tf.function(sampler)
        else:
            raise NotImplementedError(
                "Backend {} is not supported for MCMC inference methods.".format(
                    backend_name
                )
            )

    def sampling(self, sess=None):
        """Perform sampling with Hamiltonian Monte Carlo."""
        print("sampling from posterior distribution ...\n")
        if self.sampler is None:
            raise ValueError("Sampler is not found.")
        if backend_name == "tensorflow.compat.v1":
            # tensorflow.compat.v1
            samples, results = sess.run(self.sampler())
        elif backend_name == "tensorflow":
            samples, results = self.sampler()
        else:
            raise NotImplementedError(
                "Backend {} is not supported.".format(backend_name)
            )
        print("Finished sampling from posterior distribution ...\n")
        return samples, results


class HMC(MCMC):
    """Adaptive Hamiltonian Monte Carlo inference method."""

    def __init__(
        self, num_samples, num_burnin, init_time_step=0.1, leapfrog_step=30, seed=None,
    ):
        super().__init__()
        self._params.update(
            {
                "num_samples": int(num_samples),
                "num_burnin": int(num_burnin),
                "init_time_step": init_time_step,
                "leapfrog_step": leapfrog_step,
                "seed": seed,
            }
        )
        # to compute acceptance rate
        self.trace_fn = lambda _, pkr: pkr.inner_results.is_accepted

    def make_sampler(self, target_log_prob_fn, init_state):
        self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=self.params["leapfrog_step"],
                step_size=self.params["init_time_step"],
            ),
            num_adaptation_steps=int(self.params["num_burnin"] * 0.8),
        )
        self.set_sampler(init_state)


class LD(MCMC):
    """Langevin dynamics method."""

    def __init__(self, num_samples, num_burnin, time_step):
        super().__init__()
        self._params.update(
            {
                "num_samples": num_samples,
                "num_burnin": num_burnin,
                "time_step": time_step,
            }
        )

    def make_sampler(self, target_log_prob_fn, init_state):
        self.mcmc_kernel = tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn=target_log_prob_fn, step_size=self.params["time_step"],
        )
        self.set_sampler(init_state)


class MALA(MCMC):
    """Metropolis adjusted Langevin algorithm."""

    def __init__(self, num_samples, num_burnin, time_step):
        super().__init__()
        self._params.update(
            {
                "num_samples": num_samples,
                "num_burnin": num_burnin,
                "time_step": time_step,
            }
        )

    def make_sampler(self, target_log_prob_fn, init_state):
        self.mcmc_kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob_fn, step_size=self.params["time_step"],
        )
        self.set_sampler(init_state)


class NUTS(MCMC):
    """NUTS method."""

    def __init__(self, num_samples, num_burnin, time_step, seed=None):
        super().__init__()
        self._params.update(
            {
                "num_samples": num_samples,
                "num_burnin": num_burnin,
                "time_step": time_step,
                "seed": seed,
            }
        )

    def make_sampler(self, target_log_prob_fn, init_state):
        """Initializes a MCMC chain."""
        self.mcmc_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn, step_size=self.params["time_step"],
        )
        self.set_sampler(init_state)
