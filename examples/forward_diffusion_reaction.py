"""NeuralUQ for 1-D nonlinear diffusion reaction equation (forward), from UQ review paper."""


# See also this paper for reference:
# Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons

import neuraluq as neuq
from neuraluq.config import tf

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_data():
    data = sio.loadmat("../dataset/diffusion_reaction.mat")
    x_test, u_test, f_test = data["x_test"], data["u_test"], data["f_test"]
    x_u_train, u_train = data["x_u_train"], data["u_train"]
    x_f_train, f_train = data["x_f_train"], data["f_train"]
    return x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test


def pde_fn(x, u):
    D = 0.01
    k = 0.7
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    return D * u_xx - k * u ** 3


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test = load_data()
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

    process = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
    )
    likelihood_u = neuq.likelihoods.Normal(
        inputs=x_u_train, targets=u_train, processes=[process], pde=None, sigma=0.1,
    )
    likelihood_f = neuq.likelihoods.Normal(
        inputs=x_f_train, targets=f_train, processes=[process], pde=pde_fn, sigma=0.1,
    )

    model = neuq.models.Model(
        processes=[process], likelihoods=[likelihood_u, likelihood_f],
    )
    method = neuq.inferences.HMC(
        num_samples=1000,
        num_burnin=800,
        init_time_step=0.1,
        leapfrog_step=50,
        seed=6666,
    )
    model.compile(method)
    samples, results = model.run()
    print("Acceptance rate: %.3f \n" % (np.mean(results)))  # if HMC is used

    ################################# Predictions ####################################
    (u_pred,) = model.predict(x_test, samples, processes=[process])
    (f_pred,) = model.predict(x_test, samples, processes=[process], pde_fn=pde_fn)
    ############################### Postprocessing ###################################
    neuq.utils.plot1d(x_u_train, u_train, x_test, u_test, u_pred[..., 0])
    neuq.utils.plot1d(x_f_train, f_train, x_test, f_test, f_pred[..., 0])
