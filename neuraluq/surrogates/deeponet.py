# This file contains classes for deep operator networks (DeepONets) that work with dataset in the format of
# Cartesian product. This type of DeepONets is the simplest one and is able to deal with well-structured
# data. There are two types of such DeepONets are considered for uncertainty quantification in this package:
# 1. a completely pre-trained one, which is defined loading its branchnet and trunknet and works as a
#   determined surrogate. For this type, no specification for the DeepONet is needed and the surrogate is
#   completely determined by loaded branch and trunk nets.
# 2. a trainable one, for which specification for the DeepONet, e.g. layers of the branch and trunk nets, is
#   needed.
# Note: currently, only one-dimensional-output DeepONet is supported.


from ..config import tf
from .fnn import FNN
from .surrogate import Surrogate


class DeepONet:
    def __init__(
        self, layers_trunk, layers_branch, activation=tf.tanh, trunk=None, branch=None,
    ):
        self.L_branch = len(layers_branch) - 1
        self.L_trunk = len(layers_trunk) - 1
        self.activation = activation
        if trunk is None:
            self.trunk = FNN(layers_trunk, activation=activation)
        else:
            self.trunk = trunk
        if branch is None:
            self.branch = FNN(layers_branch, activation=activation)
        else:
            self.branch = branch

    def __call__(self, inputs, var_list):
        """
        Performs a forward propagation of DeepONet.

        Args:
            inputs (a list or tuple of tensors): The collection of inputs to the DeepONet. It
                has two elements. The first one is the input to the trunk net, and the second
                one is the input to the branch net.
            var_list (a list of tensors): The collection of (latent) variables of the DeepONet.
                The total length is 2`L_trunk` + 2`L_branch`. It contains, the weights of the 
                trunk net, the biases of the trunk net, the weights of the branch net, and the
                biases of the trunk net in order.
        Returns:
            trunk_inputs (tensor): The input, possibly reshaped and tiled, to the trunk net.
            outputs (tensor): The output of the DeepONet.
        """
        trunk_var_list = var_list[: 2 * self.L_trunk]
        branch_var_list = var_list[2 * self.L_trunk :]
        trunk_inputs, branch_inputs = inputs
        _, branch = self.branch(branch_inputs, branch_var_list)
        trunk_inputs, trunk = self.trunk(trunk_inputs, trunk_var_list)
        if len(trunk.shape) == 2:
            outputs = tf.matmul(branch, tf.transpose(trunk))
        elif len(trunk.shape) == 3:
            outputs = tf.matmul(branch, tf.transpose(trunk, [0, 2, 1]))
        else:
            raise ValueError("Shape error.")
        return trunk_inputs, outputs


class DeepONet_pretrained:
    def __init__(self, trunk, branch):
        self.trunk = trunk
        self.branch = branch

    def __call__(self, trunk_inputs, branch_inputs):
        """
        Performs a forward pass of a pretrained DeepONet. Shapes of the inputs should be taken cared of. There are in 
        total 2 scenarios:

        1. [batch_size_trunk, in_dim_trunk] for `trunk_inputs`, [batch_size_branch, in_dim_branch] for 
            `branch_inputs`.
            This happens in deterministic setting, for which batch_branch is generally not equal to batch_trunk.
            In this case, the output of the DeepONet will be [batch_size_trunk, batch_size_branch].

        2. [batch_size, batch_size_trunk, in_dim_trunk] for `trunk_inputs`, [batch_size, in_dim_branch] for 
            `branch_inputs`.
            This happens in the training of Bayesian setting, where the batch_size is 1, or in the inference of
            both Bayesian and deterministic settings, where the batch_size is the number of posterior samples.
            In this case, the output of the DeepONet will be [batch_size, batch_size_trunk].

        Args:
            trunk_inputs (tensor): The inputs to the trunk net. See above for its shape's requirements.
            branch_inputs (tensor): The inputs to the branch net. See above for its shape's requirements.  
        Returns:
            outputs (tensor): The outputs of the DeepONet.
        """
        trunk, branch = self.trunk(trunk_inputs), self.branch(branch_inputs)
        if len(trunk.shape) == 2:
            # trunk is of shape [batch_size_trunk, latent_dim]
            # branch is of shape [batch_size_branch, latent_dim]
            outputs = tf.matmul(branch, tf.transpose(trunk))[..., None]
            # outputs = tf.matmul(trunk, tf.transpose(branch))[..., None]
        elif len(trunk.shape) == 3:
            # trunk is of shape [batch_size, batch_size_trunk, latent_dim]
            # branch is of shape [batch_size, latent_dim]
            # outputs = tf.einsum("K")
            outputs = tf.einsum("Bik,Bk->Bi", trunk, branch)[..., None]
        return outputs
