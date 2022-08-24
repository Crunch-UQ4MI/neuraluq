"""NeuralUQ for inverse KdV problem."""

import neuraluq as neuq
import neuraluq.variables as neuq_vars
from neuraluq.config import tf

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_data(noise_u, noise_f):
    data = sio.loadmat("../dataset/kdv_train.mat")
    x_u_train, t_u_train = data["x_u_train"], data["t_u_train"]
    x_f_train, t_f_train = data["x_f_train"], data["t_f_train"]
    x_test, t_test, u_test = data["x_test"], data["t_test"], data["u_test"]
    x_test, t_test, u_test = (
        x_test.reshape([-1, 1]),
        t_test.reshape([-1, 1]),
        u_test.reshape([-1, 1]),
    )
    u_train, f_train = data["u_train"], data["f_train"]
    train_u = x_u_train, t_u_train, u_train
    train_f = x_f_train, t_f_train, f_train
    test = x_test, t_test, u_test
    return train_u, train_f, test


def pde_fn(x, u, k_1, k_2):
    u_x, u_t = tf.split(tf.gradients(u, x)[0], 2, axis=-1)
    u_xx = tf.gradients(u_x, x)[0][..., 0:1]
    u_xxx = tf.gradients(u_xx, x)[0][..., 0:1]
    f = u_t - tf.exp(k_1) * u * u_x - tf.exp(k_2) * u_xxx
    return f

@neuq.utils.timer
def Samplable(
    x_u_train, t_u_train, u_train, x_f_train, t_f_train, f_train, noise, layers
):
    # build processes
    process_u = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=neuq_vars.fnn.Samplable(layers=layers, mean=0, sigma=1),
    )
    process_logk_1 = neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        prior=neuq_vars.const.Samplable(mean=0, sigma=1),
    )
    process_logk_2 = neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        prior=neuq_vars.const.Samplable(mean=0, sigma=1),
    )
    # build likelihood
    likelihood_u = neuq.likelihoods.Normal(
        inputs=np.concatenate([x_u_train, t_u_train], axis=-1),
        targets=u_train,
        processes=[process_u],
        sigma=noise,
    )
    likelihood_f = neuq.likelihoods.Normal(
        inputs=np.concatenate([x_f_train, t_f_train], axis=-1),
        targets=f_train,
        processes=[process_u, process_logk_1, process_logk_2],
        pde=pde_fn,
        sigma=noise,
    )
    # build model
    model = neuq.models.Model(
        processes=[process_u, process_logk_1, process_logk_2],
        likelihoods=[likelihood_u, likelihood_f],
    )
    # assign and compile method
    # Change the parameters to make the acceptance rate close to 0.6.
    method = neuq.inferences.HMC(
        num_samples=500,
        num_burnin=3000,
        init_time_step=0.01,
        leapfrog_step=50,
        seed=66,
    )
    model.compile(method)
    # obtain posterior samples
    samples, results = model.run()
    print("Acceptance rate: %.3f \n"%(np.mean(results)))

    processes = [process_u, process_logk_1, process_logk_2]
    return processes, samples, model


@neuq.utils.timer
def Trainable(
    x_u_train, t_u_train, u_train, x_f_train, t_f_train, f_train, noise, layers
):
    # build processes
    process_u = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        posterior=neuq_vars.fnn.Trainable(layers=layers),
    )
    process_logk_1 = neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        posterior=neuq_vars.const.Trainable(value=0),
    )
    process_logk_2 = neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        posterior=neuq_vars.const.Trainable(value=0),
    )
    # build losses
    loss_u = neuq.likelihoods.MSE(
        inputs=np.concatenate([x_u_train, t_u_train], axis=-1),
        targets=u_train,
        processes=[process_u],
        multiplier=1.0,
    )
    loss_f = neuq.likelihoods.MSE(
        inputs=np.concatenate([x_f_train, t_f_train], axis=-1),
        targets=f_train,
        processes=[process_u, process_logk_1, process_logk_2],
        pde=pde_fn,
        multiplier=1.0,
    )
    # build model
    model = neuq.models.Model(
        processes=[process_u, process_logk_1, process_logk_2],
        likelihoods=[loss_u, loss_f],
    )
    # assign and compile method
    method = neuq.inferences.DEns(
        num_samples=20, num_iterations=20000, optimizer=tf.train.AdamOptimizer(1e-3),
    )
    model.compile(method)
    # obtain posterior samples
    samples = model.run()
    samples = neuq.utils.batch_samples(samples)  # reshape

    processes = [process_u, process_logk_1, process_logk_2]
    return processes, samples, model


def plots(
    logk_1_pred,
    logk_2_pred,
    u_pred,
    x_test,
    t_test,
    u_test,
    x_u_train,
    t_u_train,
    u_train,
):
    k_1_pred, k_2_pred = np.exp(logk_1_pred), np.exp(logk_2_pred)
    print("Mean & Std of k1 are %.3f, %.3f" % (np.mean(k_1_pred), np.std(k_1_pred)))
    print("Mean & Std of k2 are %.3f, %.3f" % (np.mean(k_2_pred), np.std(k_2_pred)))

    u_pred = np.reshape(u_pred, [-1, NT, NX])
    mu = np.mean(u_pred, axis=0)
    std = np.std(u_pred, axis=0)

    x_test = np.reshape(x_test, [NT, NX])
    t_test = np.reshape(t_test, [NT, NX])
    u_test = np.reshape(u_test, [NT, NX])
    i = 15

    current_t = t_test[i][0]
    current_x = x_u_train[t_u_train == current_t]
    current_u = u_train[t_u_train == current_t]
    # std = np.sqrt(std**2 + 0.1**2)
    plt.plot(np.linspace(-10, 10, 201), mu[i, :], "--", label="mean")
    plt.fill_between(
        np.linspace(-10, 10, 201), (mu + 2 * std)[i, :], (mu - 2 * std)[i, :], alpha=0.3
    )
    plt.plot(np.linspace(-10, 10, 201), u_test[i, :], label="reference")
    plt.plot(current_x, current_u, "o", label="observations")
    plt.legend()
    plt.title("t=" + str(current_t))
    plt.show()


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    NT, NX = 31, 201
    noise = 0.1
    train_u, train_f, test = load_data(noise, noise)
    x_u_train, t_u_train, u_train = train_u
    x_f_train, t_f_train, f_train = train_f
    x_test, t_test, u_test = test

    layers = [2, 50, 50, 1]

    ############################### Choose framework #################################
    processes, samples, model = Samplable(
        x_u_train, t_u_train, u_train, x_f_train, t_f_train, f_train, noise, layers,
    )
    # processes, samples, model = Trainable(
    #     x_u_train, t_u_train, u_train, x_f_train, t_f_train, f_train, noise, layers,
    # )

    ################################# Predictions ####################################
    u_pred, logk_1_pred, logk_2_pred = model.predict(
        np.concatenate([x_test, t_test], axis=-1), samples, processes, pde_fn=None,
    )
    ############################### Postprocessing ###################################
    # TODO: save the results, instead of visualizing them.
    plots(
        logk_1_pred,
        logk_2_pred,
        u_pred,
        x_test,
        t_test,
        u_test,
        x_u_train,
        t_u_train,
        u_train,
    )

    '''
    sio.savemat(
        "./Output/kdv_HMC.mat",
        {
            "x_u_train": x_u_train,
            "t_u_train": t_u_train,
            "u_train": u_train,
            "x_f_train": x_f_train,
            "t_f_train": t_f_train,
            "f_train": f_train,
            "x_test": x_test,
            "t_test": t_test,
            "u_test": u_test,
            "u_pred": u_pred,
            "k_1": samples_k_1.flatten(),
            "k_2": samples_k_2.flatten(),
        },
    )
    '''
