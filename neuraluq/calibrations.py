import numpy as np

from .config import tf, backend_name
from .inferences import Optimizer


def NLL(predictions, targets, std):
    NLL = tf.reduce_sum(
        tf.math.log(tf.sqrt(2 * np.pi) * std)
        + (predictions - targets) ** 2 / 2 / std ** 2,
    )
    return NLL


class CalibrationVar:
    def __init__(
        self,
        targets,
        predictions,
        stds,
        optimizer=tf.compat.v1.train.AdamOptimizer(1e-3),
    ):
        self.targets = targets
        self.predictions = predictions
        self.stds = stds
        self.optimizer = Optimizer(optimizer)

        # homogeneous standard deviation calibrator
        self.log_s = tf.Variable(0.0, tf.float32)
        self.trainable_variables = [self.log_s]

    def calibrate(self, num_iterations, sess=None, display_every=100):
        calibration_loss = NLL(
            self.predictions, self.targets, tf.math.exp(self.log_s) * self.stds,
        )
        self.optimizer.make_train_op(
            loss_fn=lambda: calibration_loss, trainable_variables=[self.log_s],
        )
        if backend_name == "tensorflow.compat.v1":
            sess.run(
                tf.variables_initializer(
                    self.trainable_variables + self.optimizer.variables
                )
            )
            self.optimizer.train(num_iterations, sess=sess, display_every=display_every)
            results = sess.run(tf.math.exp(self.log_s))
        else:
            raise NotImplementedError(
                "Calibration for backend {} is not implemented.".format(backend_name)
            )
        return results
