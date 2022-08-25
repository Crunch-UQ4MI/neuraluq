"""
This file contains configurations of the library. Currently it supports backend setting
(tensorflow & tensorflow.compat.v1), data type, device for training (CPU & GPU).
"""

import os
import tensorflow as tf
import tensorflow_probability as tfp

# Currently, only two backends are supported: Tensorflow 1 and TensorFlow 2.
# It is worth noting that both backends require installing the lastest released
# version of Tensorflow, which is generally called Tensorflow 2. Tensorflow 1 is
# in fact supported through tensorflow.compat.v1, with eager mode shut down. There
# are slight but crucial differences between working in Tensorflow 1 environment
# directly on installed tensorflow 1.15, and indirectly by shutting down the eager
# mode and importing tensorflow.compat.v1 on installed tensorflow 2.x. Our package
# may fail to work on the former. However, based on our past experiments, the latter
# is quite sufficient for people who enjoy good old graph-mode Tensorflow.


def set_backend(backend_name="tensroflow.compat.v1"):
    # TODO 1: load backend_name from command line or other file
    # TODO 2: automatically set default backend.
    # See https://github.com/lululxvi/deepxde/tree/master/deepxde/backend for reference

    if backend_name == "tensorflow.compat.v1":
        import tensorflow.compat.v1 as _tf

        # TODO: disable other v2 behaviors as well
        _tf.disable_eager_execution()
    elif backend_name == "tensorflow":
        import tensorflow as _tf
    else:
        raise ValueError("Backend {} is not supported.".format(backend_name))
    return _tf


def set_device(device_name="cpu"):
    # TODO: support GPU
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        raise ValueError("Device {} is not supported.".format(device_name))


def set_dtype(dtype_name="float32"):
    if dtype_name == "float32":
        dtype = tf.float32
    elif dtype_name == "float64":
        dtype = tf.float64
    else:
        raise ValueError("Data-type {} is not supported.".format(dtype_name))
    return dtype


backend_name = "tensorflow.compat.v1"
# backend_name = "tensorflow"

dtype_name = "float32"
# dtype_name = "float64"

tf = set_backend(backend_name)
tfp = tfp
dtype = set_dtype(dtype_name)
# TODO: support gpu and user-specified device
# From our experience, sampling methods like HMC run much faster in cpu than in gpu.
# For stability, we recommand setting device to be cpu.
set_device("cpu")
