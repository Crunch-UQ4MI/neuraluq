from ..config import tf, tfp
from .surrogate import Surrogate


class FNN(Surrogate):
    """Fully-connected neural network that works as a Python callable."""

    def __init__(
        self, layers, activation=tf.tanh, input_transform=None, output_transform=None,
    ):
        self.L = len(layers) - 1
        self.activation = activation
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, inputs, var_list):
        return self.forward(inputs, var_list)

    def forward(self, inputs, var_list):
        """
        Returns the outputs of the network(s), defined by sample(s) of weights and biases. There are in total two scenarios
        here:
        1. `var_list` contains one sample of weights and biases, whose shapes are 2-dimensional, e.g. [layers[i], layers[i+1]]
            for weights. For this case, the batch size of the outputs is equal to the batch size of `inputs`, which is 1.
        2. `var_list` contains multiple samples of weights and biases, whose shapes are 3-dimensional, e.g. [N, layers[i], 
            layers[i+1]] for weights, where N is the number of samples, which is also equal to the number of independent
            networks. For this case, the batch size of the outputs is also N, since we have N networks. To perform automatic
            differentiation correctly in all backends, `inputs` should be taken care of. Specifically, `inputs` needs to be
            reshaped and tiled such that it is consistent with the networks and the outputs, and returned as well for future
            computation. Note that, the forward propagation could also be done correctly if `inputs` keeps as it is. That is
            due to the support of batch-wise matrix multiplication, e.g. tf.matmul. However, the computation of the gradient 
            with respect to `inputs` will be wrong. Hence, `inputs` is reshaped and tiled anyway, for this scenario, such
            that it works for either function approximation problems, where no PDE is involved, or PDE problems.

            Args:
                inputs (tensor): The network inputs.
                var_list (list of tensors): The list of weights and biases for one network. First half is for weights in
                    sequential order, and the second half is for biases.

            Returns:
                inputs (tensor): The reshaped and/or tiled inputs of the network, which are for later usage, e.g. against
                    which the derivatives are taken.
                outputs (tensor): The outputs of the network.
        """
        # Note: in this case, no reshape or tile of the inputs is needed.
        w, b = var_list[: self.L], var_list[self.L :]
        if len(w[0].shape) == 3:
            # scenario 2
            if len(inputs.shape) == 2:
                # inputs is of 2-dimensional shape
                inputs = tf.tile(inputs[None, ...], [w[0].shape[0], 1, 1])
            elif inputs.shape[0] != w[0].shape[0]:
                # inputs is of 3-dimensional shape but its shape[0] is inconsistent with the number of the networks.
                raise ValueError(
                    "Shape of inputs is wrong or not supported. It probably happens because inputs is tiled incorrectly."
                )
        if self.input_transform is not None:
            outputs = self.input_transform(inputs)
        else:
            outputs = inputs

        for i in range(self.L - 1):
            outputs = self.activation(tf.matmul(outputs, w[i]) + b[i])
        outputs = tf.matmul(outputs, w[-1]) + b[-1]

        if self.output_transform is not None:
            outputs = self.output_transform(outputs)
        return inputs, outputs
