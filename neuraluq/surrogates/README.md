# General rules for surrogates

This folder contains all surrogates, each one of which is a Python class and acts like a Python callable.

Every surrogate class must have an overwritten `__call__` method, which, in general, takes two arguments, `inputs` and `var_list`. Exception happens for `DeepONet`, which has four arguments, input to the branch net, input to the trunk net, variables of the branch net and variables of the trunk net. `inputs` is the input to the surrogate and `var_list` contains one or multiple realizations of the latent variables, which could be either determinstic or random and are used to define one or multiple surrogates. With `var_list`, `__call__` acts like a regular or an ensemble of functions. The number of functions is determined by the number of realizations in `var_list`. 

`__call__` method in general has two outputs, `inputs` and `outputs`, where `inputs` is the input, possibly reshaped and tiled, to the surrogate and `outputs` is the output of the surrogate. Here, 2 scenarios are considered. One is when `var_list` contains only one realization of the latent variables. For this case, `inputs` keeps as it is and `outputs` is the just the output of one surrogate taking `inputs` as input. The other one is when `var_list` contains multiple realizations. For this case, `inputs` is added one dimension before the first dimension and tiled to a size the same as the number of realizations. Hence, `__call__` returns the same, multiple inputs stacked at the first dimension, and different, multiple outputs stacked at the first dimensions. `DeepONet`, by its definition, returns the input(s) to the trunk net and its output(s).

# Requirements of the arguments of `__call__`

1. For every surrogate, the argument `inputs` must be one of the following two shapes: [batch_size_input, input_dim] or [batch_size_surrogate, batch_size_input, input_dim], where batch_size_input is the size of the input, on which the values of surrogate are computed, input_dim is the dimension of the input, and batch_size_surrogate is the number of considered realizations, which is equal to the number of total considered surrogates.

2. For every surrogate, the argument `var_list` must be a list of tensors, even though there is only one latent variable, e.g. in functional prior examples. When `var_list` contains only one realization, there is no restriction on its elements' shapes. However, when `var_list` contains multiple realizations, its elements must have the same size on their first dimensions, which determines the total number of realizations.

# Requirements of the returns of `__call__`

1. The return `inputs` must be one of the following two shapes: [batch_size_input, input_dim] or [batch_size_surrogate, batch_size_input, input_dim].

2. The return `outputs` could have various shapes, determined the types of surrogates. For example, for `FNN`, `outputs`'s shape is either [batch_size_input, output_dim] or [batch_size_surrogate, batch_size_input, output_dim].
