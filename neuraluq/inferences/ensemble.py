"""This file contains ensemble-based inference methods, e.g. Deep Ensemble"""


import tensorflow.compat.v1 as tf
import numpy as np


from .inference import Inference, Optimizer
from .. import utils


class DEns(Inference):
    """Deep ensemble inference method: optimize to sample, or sampling while optimizing."""

    def __init__(
        self,
        num_iterations,
        num_samples=None,
        optimizer=tf.train.AdamOptimizer(1e-3),
        is_parallelized=False,
    ):
        if num_samples is None and is_parallelized is False:
            raise ValueError(
                "The number of samples cannot be None when sequential training is performed."
            )
        self._params = {
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "optimizer": Optimizer(optimizer),
            "is_parallelized": is_parallelized,
        }
        self.sampler = None
        self.method_type = "Ensemble"

    def make_sampler(self, sess, loss_fn, trainable_variables):
        """Creates a sampler for deep ensemble."""
        optimizer = self.params["optimizer"]
        optimizer.make_train_op(loss_fn, trainable_variables)

        def sampler(_sess):
            # TODO: support Tensorflow 2
            _sess.run(
                tf.variables_initializer(
                    var_list=trainable_variables + self.params["optimizer"].variables
                )
            )
            optimizer.train(
                num_iterations=int(self.params["num_iterations"]),
                sess=_sess,
                display_every=1000,
            )
            return _sess.run(trainable_variables)

        self.sampler = sampler

    def sampling(self, sess):
        """Performs sampling with deep ensemble."""
        if self.sampler is None:
            raise ValueError("Sampler is not found.")

        if self.params["is_parallelized"] is False:
            # Obtain ensembles sequentially
            samples = []
            for i in range(self.params["num_samples"]):
                print("Generating {}th sample by deep ensemble...".format(str(i)))
                samples += [self.sampler(sess)]
            samples = utils.batch_samples(samples)
        else:
            # Obtain ensembles in parallel and ignore `num_samples`
            samples = self.sampler(sess)
        # Note: each element of samples represents one network. For future computation,
        # it is recommended to stack them to a list, each element of which is a collection
        # of all samples for one weight or bias.
        return samples


class SEns(Inference):
    """Snapshot Ensemble inference method: optimize to sample, or sampling while optimizing."""

    def __init__(self, num_samples, num_iterations):
        # TODO: support Snapshot Ensemeble method
        self.global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
        lr_schedule = tf.train.cosine_decay(
            learning_rate=0.1, global_step=self.global_step, decay_steps=num_iterations,
        )
        # TODO: support user-specified optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_schedule)
        self._params = {
            "num_samples": num_samples,
            "num_iterations": num_iterations,
            "optimizer": Optimizer(optimizer),
        }
        self.sampler = None
        self.method_type = "Ensemble"

    def make_sampler(self, sess, loss_fn, trainable_variables):
        """Create a sampler for Snapshot Ensemble method."""
        optimizer = self.params["optimizer"]
        optimizer.make_train_op(loss_fn, trainable_variables, self.global_step)

        def sampler(_sess, from_start):
            # TODO: support Tensorflow 2
            _sess.run(
                tf.variables_initializer(
                    self.params["optimizer"].variables + [self.global_step]
                )
            )
            if from_start:
                # also initialize trainable variables
                _sess.run(tf.variables_initializer(trainable_variables))
            optimizer.train(
                num_iterations=int(self.params["num_iterations"]),
                sess=_sess,
                display_every=100,
            )
            return _sess.run(trainable_variables)

        self.sampler = sampler

    def sampling(self, sess):
        """Performs sampling with snapshot ensemble."""
        if self.sampler is None:
            raise ValueError("Sampler is not found.")
        samples = []
        from_start = True
        for i in range(self.params["num_samples"]):
            print("Generating {}th sample by snapshot ensemble...".format(str(i)))
            samples += [self.sampler(sess, from_start)]
            from_start = False
        # Note: each element of samples represents one network. For future computation,
        # it is recommended to stack them to a list, each element of which is a collection
        # of all samples for one weight or bias.
        return samples
