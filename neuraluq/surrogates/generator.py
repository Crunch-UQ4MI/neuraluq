# This file contains the classes for generators.
# Note: currently, only one-dimensional-output generator is supported.


from ..config import tf


class Generator:
    """
    A Generator in a Physics-informed GAN format, whose input to trunk net is unknown and learnable.
    The rest of the network's parameters is fixed. This particular structure is for PI-GAN, FP, etc.
    Note that, to create a `Generator` instance, you need to load a pre-trained branchnet and a
    pre-trained trunknet, and rewrite `__call__` method.

    Args:
        branch: A (pre-trained) Tensorflow.keras.Model instance, whose input is the unknown, learnable
            (random) variable. Together with a trunk, it becomes a (stochastic) process.
        trunk: A (pre-trained) Tensorflow.keras.Model instance, whose input is the input of the surrogate.
            Together with a branch, it becomes a (stochastic) process.
        variable_shape: A list or tuple of integers as the shape of trainable/samplable variables. That is,
            the shape of the input to the branch.
    """

    def __init__(self, branch, trunk, variable_shape):
        self.branch = branch
        self.trunk = trunk
        self.variable_shape = variable_shape  # shape of branch net's input

    def __call__(self, inputs, var_list):
        """
        Returns the outputs of a `Generator`, with respect to `inputs`, which is the input to the
        trunk net, and `var_list`, which is the input to the branch net. Unlike DeepONet, Generator's
        `var_list` contains only one element, xi. 
        Like most surrogates, `Generator` has two scenarios:
        1. `var_list` contains one sample. In this case, the batch size is 1 and `inputs` should be 
            2-dimensional. Hence, the output is 2-dimensional with batch size 1.
        2. `var_list` contains multiple samples. In this case, the batch size is determined by the 
            batch size of `var_list`'s element, and `inputs` should be reshaped and tiled, if it has 
            not been, to match the batch size.

            Args:
                inputs (tensor): The inputs to the trunk net, with shape [None, dim_in], [batch_size, 
                    None, dim_in].
                var_list (list of tensors): The list contains only one tensor, xi, which is a batch of
                    input to the branch net, with shape [batch_size, latent_dim].

            Returns:
                y (tensor): The `Generator`'s outputs, with shape [batch_size, None, dim_out]
        """
        # Currently, only one-dimenional var_listt[0] is supported.
        xi = var_list[0]
        if len(xi.shape) == 1:
            # add one dimension such that matrix multiplication could be performed
            xi = xi[None, ...]
        elif len(inputs.shape) == 2:
            # for 2 dimensional inputs, reshape and tile it.
            inputs = tf.tile(inputs[None, ...], [xi.shape[0], 1, 1])
        elif xi.shape[0] != inputs.shape[0]:
            # inputs is 3-dimensional but with inconsistent shape
            raise ValueError(
                "Shape of inputs is wrong or not supported. It probably happens because inputs is tiled incorrectly."
            )

        trunk = self.trunk(inputs)
        branch = self.branch(xi)
        if len(trunk.shape) == 2:
            # trunk is of shape [batch_size_trunk, latent_dim]
            # branch is of shape [1, latent_dim]
            outputs = tf.matmul(trunk, tf.transpose(branch))  # [batch_size_trunk, 1]
        elif len(trunk.shape) == 3:
            # trunk is of shape [batch_size, batch_size_trunk, latent_dim]
            # branch is of shape [batch_size, latent_dim]
            outputs = tf.einsum("Bij,Bj->Bi", trunk, branch)[..., None]
            # [batch_size, batch_size_trunk, 1]
        return inputs, outputs
