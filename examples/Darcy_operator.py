"""NeuralUQ for Darcy problem usin Deep Ensembles"""

import neuraluq as neuq
import neuraluq.variables as neuq_vars
from neuraluq.config import tf

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_data():
    data = sio.loadmat("../dataset/Darcy_operator_train.mat")
    data_OOD = sio.loadmat("../dataset/Darcy_OOD.mat")
    x_loc, y_loc = data["x_train"], data["y_train"]
    f_train, u_train = data["f_train"], data["u_train"]
    f_test, u_test = data["f_test"], data["u_test"]
    f_ood, u_ood = data_OOD["f_test"], data_OOD["u_test"]
    # preprocessing
    loc = 2 * np.concatenate([x_loc, y_loc], axis=-1) - 1  # normalized to [-1, 1]
    f_train, f_test, f_ood = [np.log(e) for e in [f_train, f_test, f_ood]]
    u_train, u_test, u_ood = [np.log(e + 1e-7) for e in [u_train, u_test, u_ood]]
    return loc, f_train, u_train, f_test, u_test, f_ood, u_ood


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    loc, f_train, u_train, f_test, u_test, f_ood, u_ood = load_data()
    input_mean = np.mean(f_train, axis=0, keepdims=True)
    input_std = np.std(f_train, axis=0, keepdims=True)
    output_mean = np.mean(u_train, axis=0, keepdims=True)
    output_std = np.std(u_train, axis=0, keepdims=True)

    f_train = (f_train - input_mean) / (input_std + 1e-7)
    u_train = (u_train - output_mean) / (output_std + 1e-7)
    f_test = (f_test - input_mean) / (input_std + 1e-7)

    layers_trunk = [2, 128, 128, 128, 100]
    layers_branch = [400, 128, 128, 100]

    ################################ Build processes #################################
    surrogate = neuq.surrogates.DeepONet(
        layers_trunk=layers_trunk, layers_branch=layers_branch,
    )
    trunk_variables = neuq.variables.fnn.Trainable(layers=layers_trunk)
    branch_variables = neuq.variables.fnn.Trainable(layers=layers_branch)

    process_u = neuq.process.Process(
        surrogate=surrogate, posterior=trunk_variables + branch_variables,
    )

    ############################### Build likelihoods ################################
    loss = neuq.likelihoods.MSE_operator(
        inputs=(loc, f_train), targets=u_train, processes=[process_u], batch_size=64,
    )

    ################### Build models, assign methods, and compile ####################
    model = neuq.models.Model(processes=[process_u], likelihoods=[loss])
    method = neuq.inferences.DEns(num_samples=5, num_iterations=50000)
    model.compile(method)

    ############################# Posterior inference ###############################
    samples = model.run()
    samples = neuq.utils.batch_samples(samples)

    ################################# Predictions ###################################
    # for in-distribution estimate
    # The figures presented in Sec. 4.4.2 of the paper are computed using the 87th f as input.
    inputs = loc, f_test
    (u_pred,) = model.predict(inputs, samples, [process_u])
    u_pred = u_pred * (output_std + 1e-7) + output_mean
    mu_u = np.mean(u_pred, axis=0)
    std_u = np.std(u_pred, axis=0)
    L2 = neuq.metrics.RL2E(mu_u, u_test)
    print("\nIn-distribution test L2 relative error: ", L2)
    print("\nIn-distribution test MSE: ", neuq.metrics.MSE(mu_u, u_test))

    # for out-of-distribution estimate
    inputs = loc, f_ood
    (u_pred,) = model.predict(inputs, samples, [process_u])
    u_pred = u_pred * (output_std + 1e-7) + output_mean
    mu_u = np.mean(u_pred, axis=0)
    std_u = np.std(u_pred, axis=0)
    L2 = neuq.metrics.RL2E(mu_u, u_ood)

    print("\nOut-of-distribution test L2 relative error: ", L2)
    print("\nOut-of-distribution test MSE: ", neuq.metrics.MSE(mu_u, u_ood))
