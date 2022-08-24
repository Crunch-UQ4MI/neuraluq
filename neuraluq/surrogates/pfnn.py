import tensorflow.compat.v1 as tf


class PFNN(tf.keras.Model):
    """Parallelized fully connected neural network, for deep ensemble method."""

    def __init__(
        self, layers, activation, num_ensemble=10, initializer="glorot_uniform", l2=1e-3
    ):
        super().__init__()
        W, b = [], []
        W_init = tf.keras.initializers.get(initializer)
        b_init = tf.keras.initializers.Zeros()
        for i in range(len(layers) - 1):
            W_i = tf.Variable(
                tf.stack(
                    [
                        W_init(shape=[layers[i], layers[i + 1]])
                        for _ in range(num_ensemble)
                    ],
                    axis=0,
                ),
                dtype=tf.float32,
                name="stacked_W_" + str(i),
            )
            b_i = tf.Variable(
                tf.stack(
                    [b_init(shape=[1, layers[i + 1]]) for _ in range(num_ensemble)],
                    axis=0,
                ),
                dtype=tf.float32,
                name="stacked_b_" + str(i),
            )
            W += [W_i]
            b += [b_i]
        self.W, self.b = W, b
        self.activation = activation
        self.l2 = l2
        self.input_dim = layers[0]
        self.output_dim = layers[-1]

    def call(self, inputs):
        # inputs with shape [num_ensemble, None, input_dim]
        outs = inputs
        for i in range(len(self.W) - 1):
            outs = self.activation(
                tf.einsum("Nik,Nkj->Nij", outs, self.W[i]) + self.b[i]
            )
        return tf.einsum("Nik,Nkj->Nij", outs, self.W[-1]) + self.b[-1]

    def losses(self):
        # Only L2 regularization on weights is supported
        return self.l2 * tf.stack(
            [tf.reduce_sum(W ** 2, axis=[-1, -2]) for W in self.W], axis=-1
        )

    # def make_surrogate(self, sess):
    #     """
    #     Returns a light-weight Python Callable, with values of weights and biases at current
    #     states. This is particularly useful when we need to record an instance of a neural
    #     network as a Python Callable at some point, e.g. in Deep Ensemble method.
    #     """
    #     # retrieve the numpy values of weights and biases
    #     # TODO: retrieve numpy values from tf.Variable by name
    #     variables = sess.run(self.denses.trainable_variables)
    #     W, b = variables[::2], variables[1::2]
    #     W = [tf.convert_to_tensor(W_i, dtype=tf.float32) for W_i in W]
    #     b = [tf.convert_to_tensor(b_i, dtype=tf.float32) for b_i in b]
    #     L = len(W)

    #     def _fn(inputs):
    #         x = inputs
    #         for i in range(L - 1):
    #             x = self.activation(tf.matmul(x, W[i]) + b[i])
    #         return tf.matmul(x, W[-1]) + b[-1]

    #     return _fn
