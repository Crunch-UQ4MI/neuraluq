class Inference:
    """Base class for all inference methods. """

    def __init__(self):
        self._params = None
        self.sampler = None

    @property
    def params(self):
        return self._params

    def make_sampler(self):
        """Creates a sampler for the inference."""
        raise NotImplementedError("Make sampler method to be implemented.")

    def sampling(self, sess):
        """Performs sampling."""
        raise NotImplementedError("Sampling method to be implemented.")


# inference_type_dict = ["sample", "optimize_to_sample", "optimize_and_sample"]


class Optimizer:
    """Method for optimization-based models."""

    def __init__(self, optimizer, batch_size=None):
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.train_op = None
        self.loss_op = None

    def make_train_op(self, loss_fn, trainable_variables, global_step=None):
        if not isinstance(trainable_variables, (list, tuple)):
            trainable_variables = [trainable_variables]
        # TODO: support Tensorflow 2 eager mode
        if self.batch_size is None:
            loss_op = loss_fn()
        else:
            loss_op = loss_fn(self.batch_size)
        train_op = self.optimizer.minimize(
            loss_op, var_list=trainable_variables, global_step=global_step,
        )
        self.loss_op, self.train_op = loss_op, train_op

    def reset(self, sess=None):
        raise NotImplementedError("reset optimizer is not implemented.")

    def train(self, num_iterations, sess=None, display_every=100):
        if self.train_op is None:
            raise ValueError(
                "train_op is not found. It is likely that this optimizer is not compiled with a loss function."
            )
        for it in range(num_iterations):
            # TODO: support Tensorflow 2 eager mode
            _ = sess.run(self.train_op)
            if it % display_every == 0:
                print("Iteration: ", it, ", loss: ", sess.run(self.loss_op))

    @property
    def variables(self):
        return self.optimizer.variables()
