import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np


from .inference import Inference, Optimizer


class VI(Inference):
    """Variational inference method: optimize and sample, or optimize first and sample then."""

    def __init__(
        self,
        batch_size,
        num_samples,
        num_iterations,
        optimizer=tf.train.AdamOptimizer(1e-3),
    ):
        self._params = {
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "optimizer": Optimizer(optimizer, batch_size=batch_size),
        }
        self.sampler = None
        self.method_type = "VI"

    def make_sampler(self, sess, neg_elbo_fn, all_processes, restart=True):
        """Creates a sampler by minimizing a negative ELBO first."""
        # collect trainable variables first
        trainable_variables = []
        for p in all_processes:
            trainable_variables += p.posterior.trainable_variables
        optimizer = self.params["optimizer"]
        optimizer.make_train_op(neg_elbo_fn, trainable_variables)

        if restart:
            # initialize all variables
            # TODO: support Tensorflow 2
            sess.run(
                tf.variables_initializer(trainable_variables + optimizer.variables)
            )
        # TODO: support Tensorflow 2
        optimizer.train(
            num_iterations=int(self.params["num_iterations"]),
            sess=sess,
            display_every=100,
        )

        def sampler(num_samples):
            # TODO: support Tensorflow 2
            samples = []
            for p in all_processes:
                # posterior.sample(num_samples) returns a list
                samples += sess.run(p.posterior.sample(num_samples))
            return samples

        self.sampler = sampler

    def sampling(self, sess=None):
        """Performs sampling with mean-field variational inference."""
        if self.sampler is None:
            raise ValueError("Sampler is not found.")
        return self.sampler(self.params["num_samples"])


# class Dropout(Inference):
#     """Monte Carlo dropout method."""

#     def __init__(
#         self,
#         num_samples,
#         num_iterations,
#         dropout_rate,
#         optimizer=tf.train.AdamOptimizer(1e-3),
#     ):
#         self._params = {
#             "num_samples": int(num_samples),
#             "num_iterations": int(num_iterations),
#             "dropout_rate": dropout_rate,
#             "optimizer": Optimizer(optimizer),
#         }
#         self.sampler = None

#     def make_sampler(self, sess, loss_fn, trainable_variables):
#         """Creates a sampler for Monte Carlo dropout."""
#         optimizer = self.params["optimizer"]
#         optimizer.make_train_op(loss_fn, trainable_variables)
#         # TODO: support Tensorflow 2
#         sess.run(
#             tf.variable_initializer(var_list=trainable_variables + optimizer.variables)
#         )
#         optimizer.train(
#             num_iterations=int(self.params["num_iterations"]),
#             sess=sess,
#             display_every=100,
#         )
#         # TODO: add dropout sampler
#         self.sampler = None

#     def sampling(self, sess):
#         """Performs sampling with Monte Carlo dropout."""
#         if self.sampler is None:
#             raise ValueError("Sampler is not found.")
#         return sess.run(self.sampler(sess))


# class DropConnect(Inference):
#     """Monte Carlo drop-connect method."""

#     def __init__(self):
#         self._params = None
#         self.sampler = None

#     def make_sampler(self):
#         """Create a sampler for Snapshot Ensemble method."""
#         raise NotImplementedError("Method make_sampler is not implemented.")

#     def sampling(self):
#         """Performs sampling."""
#         raise NotImplementedError("Mehtod sampling is not implemented.")
