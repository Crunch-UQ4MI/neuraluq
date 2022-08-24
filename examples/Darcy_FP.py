"""NeuralUQ for Darcy problem using PA-GAN-FP"""


import neuraluq as neuq
import neuraluq.variables as neuq_vars
from neuraluq.config import tf

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_data():
    data = sio.loadmat("../dataset/Darcy_train.mat")
    x_u_train, u_train = data["x_u_train"], data["u_train"]
    x_f_train, f_train = data["x_f_train"], data["f_train"]
    x_test, u_test, f_test = data["x_test"], data["u_test"], data["f_test"]
    return x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test

def surrogate_u(inputs, var_list):
    _, f = process_f.surrogate(x_full, var_list)
    f = f[..., 0]  # squeeze out the 1-dimensional tensor
    if len(f.shape) == 1:
        f = f[None, ...]  # make f 2-dimensional
    outputs = deeponet(inputs, f)
    if outputs.shape[0] == 1:
        # in training
        outputs = outputs[0, ...]
    return inputs, outputs

if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test = load_data()
    x_u_train, x_f_train, x_test = [2 * x - 1 for x in [x_u_train, x_f_train, x_test]]

    ################################ Build processes #################################
    # build process_f, which is a generator
    generator_branch = neuq.utils.load_network(
    file_name="../dataset/pretrained_model/KU/generator_branch.npy", activation=tf.tanh,
    )
    generator_trunk = neuq.utils.load_network(
        file_name="../dataset/pretrained_model/KU/generator_trunk.npy", activation=tf.tanh,
    )
    surrogate_f = neuq.surrogates.Generator(
        branch=generator_branch, trunk=generator_trunk, variable_shape=[100]
    )
    prior_shared = neuq_vars.const.Samplable(mean=0, sigma=1, shape=[100])
    process_f = neuq.process.Process(surrogate=surrogate_f, prior=prior_shared)

    # build process_u in 3 steps:
    # 1. load a pre-trained deeponet, by loading branch and trunk
    # 2. build a surrogate, which is a Python callable
    # 3. build a process with the same prior/posterior but brand new surrogate
    # step 1
    deeponet_branch = neuq.utils.load_network(
        file_name="../dataset/pretrained_model/KU/deeponet_branch.npy", activation=tf.tanh,
    )
    deeponet_trunk = neuq.utils.load_network(
        file_name="../dataset/pretrained_model/KU/deeponet_trunk.npy", activation=tf.tanh,
    )
    deeponet = neuq.surrogates.DeepONet_pretrained(
        trunk=deeponet_trunk, branch=deeponet_branch,
    )
    # step 2
    # to obtain input to the branch of the deeponet
    x_full = tf.constant(x_test, tf.float32)

    # step 3
    process_u = neuq.process.Process(surrogate=surrogate_u, prior=prior_shared)

    ############################### Build likelihoods ################################
    likelihood_f = neuq.likelihoods.Normal(
        inputs=x_f_train, targets=f_train, processes=[process_f], sigma=0.1,
    )


    likelihood_u = neuq.likelihoods.Normal(
        inputs=x_u_train, targets=u_train, processes=[process_u], sigma=0.1,
    )

    ################### Build models, assign methods, and compile ####################
    model = neuq.models.Model(
        processes=[process_f, process_u], likelihoods=[likelihood_f, likelihood_u],
    )
    method = neuq.inferences.HMC(
        num_samples=1000, num_burnin=1000, init_time_step=0.01, leapfrog_step=50, seed=666,
    )
    model.compile(method)

    ############################# Posterior inference ###############################
    samples, results = model.run()
    print("Acceptance rate: %.3f \n"%(np.mean(results)))
    f_pred, u_pred = model.predict(x_test, samples, [process_f, process_u])

    # sio.savemat(
    #     "./Output/Darcy_FP.mat",
    #     {
    #         "x_u_train": x_u_train,
    #         "u_train": u_train,
    #         "x_f_train": x_f_train,
    #         "f_train": f_train,
    #         "x_test": x_test,
    #         "u_test": u_test,
    #         "f_test": f_test,
    #         "f_pred": f_pred,
    #         "u_pred": u_pred,
    #     },
    # )
