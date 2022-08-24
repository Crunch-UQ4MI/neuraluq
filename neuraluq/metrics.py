from .config import tf, backend_name
import numpy as np


def NLL(mean, std, true):
    """
    Computes the negative log likelihood metric, based on independent Gaussian distributions with
    predicted `mean` and `std`.

        Args:
            mean (array): The predicted mean, with shape [None, dim].
            std (array): The predicted standard deviation, with shape [None, dim].
            true (array): The true values, based on which the metric is computed, with shape [None, dim].

        Returns:
            nll (scalar): The negative log likelihood; a scalar.
    """
    return np.sum(np.log(np.sqrt(2 * np.pi) * std) + (true - mean) ** 2 / 2 / std ** 2)


def RL2E(predictions, true):
    """
    Computes the relative L2 error.

        Args:
            predictions (array): The predictions, with shape [None, dim].
            true (array): The true values, with shape [None, dim].

        Returns:
            rl2e (scalar): The relative L2 error; a scalar
    """
    return np.sqrt(np.sum((predictions - true) ** 2)) / np.sqrt(np.sum(true ** 2))


def MSE(predictions, true):
    """
    Computes the mean-squared error.

        Args:
            predictions (array): The predictions, with shape [None, dim]
            true (array): The true values, with shape [None, dim].

        Returns:
            mse (array): The mean-squared error.
    """
    return np.mean((predictions - true) ** 2)
