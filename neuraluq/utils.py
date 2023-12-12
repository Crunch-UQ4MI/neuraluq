"""Internal utilities."""

from .config import tf
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwags):
        start_time = time.perf_counter()
        result = func(*args, **kwags)
        end_time = time.perf_counter()
        T = end_time - start_time
        print(
            "Execution time for %r function is: %.3f s, %.3f mins"
            % (func.__name__, T, T / 60)
        )
        return result

    return wrapper


def load_network(file_name, activation=tf.tanh, resnet=False):
    """
    Loads a neural network surrogate from .npy file, such that only tf.Tensor is
    involved, and no tf.Variable is created and no tf.Session is used. This is
    particularly useful for loading pre-trained generators, for downstream tasks
    of generative models, e.g. functional prior.
    Note the data stored in .npy is of type numpy.ndarray, which needs to be
    transformed to a Python list to be compatible with the definition of neural
    network.

        Args:
            file_name (string): The name of the directory that stores the weights
                and biases.
            activation (callable): The activation function for the neural network.
            resnet (bool): The boolean value that indicates if a skip connection is
                added at the final layer.

        Returns:
            _fn (callable): The function of the neural netowrk's forward propagation.
    """
    # setting allow_pickle=True is necessary, because a ragged numpy is loaded
    weights_and_biases = list(np.load(file_name, allow_pickle=True))
    weights_and_biases = [tf.constant(v, dtype=tf.float32) for v in weights_and_biases]
    # weights = weights_and_biases[::2]
    # biases = weights_and_biases[1::2]
    weights = weights_and_biases[: len(weights_and_biases) // 2]
    biases = weights_and_biases[len(weights_and_biases) // 2 :]

    def _fn(inputs):
        y = inputs
        for i in range(len(weights) - 1):
            y = activation(tf.matmul(y, weights[i]) + biases[i])
        y = tf.matmul(y, weights[-1]) + biases[-1]
        return y if resnet is False else y + inputs

    return _fn


def batch_samples(samples):
    """
    Reshapes a batch of lists of unbatched elements, to a list of batched elements.
    This functionality is useful when we obtain multiple networks' weights/biases, which
    are stored in forms of sequential lists, and want to turn them into one list, whose
    elements are batched.
    """
    N = len(samples)  # total number of samples
    L = len(samples[0])  # number of variables in each sample
    batched_samples = [
        np.zeros(shape=[N] + list(v.shape), dtype=v.dtype) for v in samples[0]
    ]
    for j in range(L):
        # go over all variables
        for i in range(N):
            # iterate over all samples
            batched_samples[j][i, ...] = samples[i][j]
    return batched_samples


def to_flat(var_list, batch_size=None):
    """Reshapes a list of (batched) tensors into a (batched) 1-D tensor."""
    if batch_size is None:
        flat = tf.concat([tf.reshape(var, [-1, 1]) for var in var_list], axis=0)
    else:
        flat = tf.concat(
            [tf.reshape(var, [batch_size, -1]) for var in var_list], axis=-1
        )
    return flat


def from_flat(flat, shape_list, batch_size=None):
    """Reshapes a (batched) 1-D tensor into a list of (batched) tensors."""
    var_list = []
    beg, end = 0, 0
    for shape in shape_list:
        end = beg + tf.reduce_prod(shape)
        var_list += (
            [tf.reshape(flat[beg:end], shape)]
            if batch_size is None
            else [tf.reshape(flat[:, beg:end], [batch_size] + shape)]
        )
        beg = end
    return var_list


def hessian(fn, var_list):
    """Computes Hessian matrix with respect to var_list."""
    shape_list = [var.shape for var in var_list]
    x = to_flat(
        var_list
    )  # declare a tensor, with respect to which the Hessian is computed
    y = fn(from_flat(x, shape_list))
    return tf.hessians(y, x)


def plot1d(
    x,
    y,
    x_test,
    y_test,
    y_samples,
    xlim=None,
    ylim=None,
    xlabel="$x$",
    ylabel="$y$",
    title="",
):
    y_mean = np.mean(y_samples, axis=0)
    y_std = np.std(y_samples, axis=0)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x, y, "k.", markersize=10, label="Training data")
    plt.plot(x_test, y_test, "k-", label="Exact")
    plt.plot(x_test, y_mean, "r--", label="Mean")
    plt.fill_between(
        x_test.ravel(),
        y_mean + 2 * y_std,
        y_mean - 2 * y_std,
        alpha=0.3,
        facecolor="c",
        label="2 stds",
    )
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.show()


def plot2d(xx1, xx2, y, xlim=None, ylim=None, clim=None, title=""):
    """
    2-D plot.

        Args:
            x1 (array): The 2-D array representing the grid of the first coordinate,
                    with shape [N1, N2].
            x2 (array): The 2-D array representing the grid of the second coordinate,
                    with shape [N1, N2].
            y (array): The 2-D array representing values of the output on the
                    grid formed by x1, x2, with shape [N1, N2]
    """
    fig, ax = plt.subplots(dpi=100)
    c = ax.pcolormesh(xx1, xx2, y, cmap="RdBu")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    c.set_clim(clim)
    plt.show()


def hist(samples, bins=20, name=""):
    """Plots histogram over samples of 1-D variable."""
    mean = np.mean(samples)
    std = np.std(samples)
    # print('sample mean: ', mean)
    # print('sample std: ', std)
    plt.hist(samples, bins=bins, density=True)
    plt.title("$\mu=${0}, $\sigma=${1}".format(str(mean), str(std)))
    plt.ylabel(name)
    plt.show()
