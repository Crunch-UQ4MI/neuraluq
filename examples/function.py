"""NeuralUQ for 1D function approximation."""

import neuraluq as neuq
import neuraluq.variables as neuq_vars
from neuraluq.config import tf

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_data():
    data = sio.loadmat("../dataset/func_train.mat")
    x_u_train = data["x_train"]
    u_train = data["y_train"]
    x_test = data["x_test"]
    u_test = data["y_test"]
    return x_u_train, u_train, x_test, u_test


@neuq.utils.timer
def Samplable(x_train, u_train, layers):
    # build surrogate
    surrogate = neuq.surrogates.FNN(layers=layers)
    # build prior and posterior
    prior = neuq_vars.fnn.Samplable(layers=layers, mean=0, sigma=1)
    # build process
    process = neuq.process.Process(surrogate=surrogate, prior=prior)
    # build likelihood
    likelihood = neuq.likelihoods.Normal(
        inputs=x_train, targets=u_train, processes=[process], sigma=0.1,
    )
    # build model
    model = neuq.models.Model(processes=[process], likelihoods=[likelihood])
    # assign and compile method
    # Note: HMC is a random method. A random seed is required if the users would like to reproduce the results on the same machine.
    # However, different machines with the same seed may also not be able to produce the same results.
    # The optimal acceptance rate in theory for HMC is around 0.6. Users can change the parameters, e.g., time step, burnin step,
    # to achieve a better acceptance rate. NoUTurn which is able to adjust the time step automatically can be used as an more advanced alternative to HMC.
    method = neuq.inferences.HMC(
        num_samples=1000,
        num_burnin=1000,
        init_time_step=0.01,
        leapfrog_step=50,
        seed=6666,
    )
    # method = neuq.inferences.MALA(
    #     num_samples=3000,
    #     num_burnin=3000,
    #     time_step=0.00002,
    # )
    model.compile(method)
    # obtain posterior samples
    samples, results = model.run()
    print("Acceptance rate: %.3f \n" % (np.mean(results)))  # if HMC is used
    return process, samples, model


@neuq.utils.timer
def Variational(x_train, u_train, layers):
    surrogate = neuq.surrogates.FNN(layers=layers)
    prior = neuq.variables.fnn.Variational(layers=layers, mean=0, sigma=1)
    # Variational inference requires a proposed distribution
    posterior = neuq_vars.fnn.Variational(
        layers=layers, mean=0, sigma=0.1, trainable=True
    )
    process = neuq.process.Process(
        surrogate=surrogate, prior=prior, posterior=posterior
    )
    likelihood = neuq.likelihoods.Normal(
        inputs=x_train, targets=u_train, processes=[process], sigma=0.1,
    )
    model = neuq.models.Model(processes=[process], likelihoods=[likelihood])
    method = neuq.inferences.VI(
        batch_size=64,
        num_samples=1000,
        num_iterations=10000,
        optimizer=tf.train.AdamOptimizer(1e-3),
    )
    model.compile(method)
    samples = model.run()
    return process, samples, model


@neuq.utils.timer
def Trainable(x_train, u_train, layers):
    surrogate = neuq.surrogates.FNN(layers=layers)
    # Deterministic training requires only posterior, which could be interpreted as
    # constant random variables. Regularizations are enforced through varibles.

    # Deep ensemble could be realized in either sequential training or parallelized
    # training, depending priority over computational time or space.
    ############# For sequential training #############
    # posterior = neuq_vars.fnn.Trainable(
    #     layers=layers, regularizer=tf.keras.regularizers.l2(1e-5),
    # )
    # method = neuq.inferences.DEns(
    #     num_iterations=20000,
    #     num_samples=10,
    #     optimizer=tf.train.AdamOptimizer(1e-3),
    # )
    ############# For parallelized training #############
    posterior = neuq_vars.pfnn.Trainable(
        layers=layers, num=95, regularizer=tf.keras.regularizers.l2(1e-5),
    )
    method = neuq.inferences.DEns(
        num_iterations=20000,
        optimizer=tf.train.AdamOptimizer(1e-3),
        is_parallelized=True,
    )

    process_u = neuq.process.Process(surrogate=surrogate, posterior=posterior)
    loss = neuq.likelihoods.MSE(inputs=x_train, targets=u_train, processes=[process_u])
    model = neuq.models.Model(processes=[process_u], likelihoods=[loss])

    # method = neuq.inferences.SEns(num_iterations=20000, num_samples=10)
    model.compile(method)
    samples = model.run()
    return process_u, samples, model


@neuq.utils.timer
def MCD(x_train, u_train, layers):
    # similar to Variational, while the only difference is MCD uses Bernoulli distribution
    # to approximate the posterior
    surrogate = neuq.surrogates.FNN(layers=layers)
    prior = neuq_vars.fnn.Variational(layers=layers, mean=0, sigma=5)
    posterior = neuq_vars.fnn.MCD(layers=layers, dropout_rate=0.05, trainable=True)
    process_u = neuq.process.Process(
        surrogate=surrogate, prior=prior, posterior=posterior,
    )
    likelihood = neuq.likelihoods.Normal(
        inputs=x_train, targets=u_train, processes=[process_u], sigma=0.1,
    )
    model = neuq.models.Model(processes=[process_u], likelihoods=[likelihood])
    method = neuq.inferences.VI(
        batch_size=64,
        num_samples=1000,
        num_iterations=10000,
        optimizer=tf.train.AdamOptimizer(1e-3),
    )
    model.compile(method)
    samples = model.run()
    return process_u, samples, model


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    x_train, u_train, x_test, u_test = load_data()
    layers = [1, 50, 50, 1]

    ####################### Build model and perform inference ########################
    # All models share the same general procedure:
    # Step 1: build surrogate, e.g. a fully-connected neural network, using [surrogates]
    # Step 2: build prior and/or posterior using [variables]
    # Step 3: build process, based the surrogate, prior and/or posterior, using [Process]
    # Step 4: build likelihood, given noisy measurements, using [likelihoods]
    # Step 5: build model using [models]
    # Step 6: create an inference method and assign it to the model using [inferences]
    # Step 7: perform posterior sampling using [model.run]

    ############################### Choose framework #################################
    process, samples, model = Samplable(x_train, u_train, layers)
    # process, samples, model = Variational(x_train, u_train, layers)
    # process, samples, model = Trainable(x_train, u_train, layers)
    # process, samples, model = MCD(x_train, u_train, layers)

    ################################# Predictions ####################################
    (u_pred,) = model.predict(x_test, samples, processes=[process])
    ############################### Postprocessing ###################################
    neuq.utils.plot1d(x_train, u_train, x_test, u_test, u_pred[:, :, 0])
