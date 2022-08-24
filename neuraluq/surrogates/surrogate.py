class Surrogate:
    """Base class for all surrogate modules."""

    def __init__(self):
        self._input_transform = None
        self._output_transform = None

    def __call__(self):
        raise NotImplementedError("__call__ is not implemented.")

    @property
    def input_transform(self):
        return self._input_transform

    @property
    def output_transform(self):
        return self._output_transform


class Identity(Surrogate):
    """An identity function for, e.g. constants."""

    def __init__(self, input_transform=None, output_transform=None):
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, inputs, var_list):
        """Returns the first element of `var_list`."""
        outputs = var_list[0]
        if self.output_transform is not None:
            outputs = self.output_transform(outputs)
        return inputs, outputs
