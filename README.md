# NeuralUQ
Scientific machine learning (SciML) has emerged recently as an effective and powerful tool for data fusion, solving ordinary/partial differential equations (ODEs, PDEs), and learning operator mappings in various scientific and engineering disciplines. Physics-informed neural networks ([PINNs](https://www.sciencedirect.com/science/article/pii/S0021999118307125)) and deep operator networks ([DeepONets](https://www.nature.com/articles/s42256-021-00302-5)) are two such models for solving ODEs/PDEs and learning operator mappings, respectively. Quantifying predictive uncertainties is crucial for risk-sensitive applications as well as for efficient and economical design. **NeuralUQ** is a Python library for uncertainty quantification in various SciML algorithms. In NeuralUQ, each UQ method is decomposed into a surrogate and an inference method for posterior estimation. NeuralUQ has included various surrogates and inference methods, i.e., 
- Surrogates
  - Bayesian Neural Networks (BNNs)
  - Deterministic Neural Networks, e.g., fully-connected neural networks (FNNs)
  - Deep Generative Models, e.g., Generative Adversarial Nets (GANs)
- Inference Methods
  - Sampling methods
    - Hamiltonian Monte Carlo (HMC)
    - Langevin Dynamics (LD)
    - No-U-Turn (NUTS)
    - Metropolis-adjusted Langevin algorithm (MALA)
  - Variational Methods
    - Mean-field Variational Inference (MFVI)
    - Monte Carlo Dropout (MCD)
  - Ensemble Methods
    - Deep ensembles (DEns)
    - Snapshot ensemble (SEns)
    - Laplace approximation (LA)
    
Users can refer to this paper for the design and description, as well as the examples, of the NeuralUQ library:
- [NeuralUQ: A comprehensive library for uncertainty quantification in neural differential equations and operators](https://epubs.siam.org/doi/pdf/10.1137/22M1518189)

Users can refer to the following papers for more details on the algorithms:
- [A comprehensive review on uncertainty quantification in scientific machine learning](https://www.sciencedirect.com/science/article/pii/S0021999122009652)
- UQ for physics-informed neural networks
  - [B-PINNs: Bayesian Physics-informed Networks](https://www.sciencedirect.com/science/article/pii/S0021999120306872)
  - [Learning Functional Priors and Posteriors from Data and Physics](https://www.sciencedirect.com/science/article/pii/S0021999122001358)
  - [Physics-Informed Generative Adversarial Networks for Stochastic Differential Equations](https://epubs.siam.org/doi/abs/10.1137/18M1225409)
  - ...
- UQ for DeepONets
  - [Learning Functional Priors and Posteriors from Data and Physics](https://www.sciencedirect.com/science/article/pii/S0021999122001358)
  - [Bayesian DeepONets](https://arxiv.org/pdf/2111.02484.pdf)
  - [Randomized Priors](https://arxiv.org/pdf/2203.03048.pdf)
  - ...
# Installation
**NeuralUQ** requires the following dependencies to be installed:

- Python 3.7.0
- Tensorflow 2.9.1
- TensorFlow Probability 0.17.0

Then install with `python`:

```
$ python setup.py install
```

For developers, you could clone the folder to your local machine via
```
$ git clone https://github.com/Crunch-UQ4MI/neuraluq.git
```

# Explore more

NeuralUQ for uncertainty quantification in general neural differential equations and operators:
- [NeuralUQ: A comprehensive library for uncertainty quantification in neural differential equations and operators](https://arxiv.org/pdf/2208.11866.pdf)
- [Uncertainty quantification in scientific machine learning: Methods, metrics, and comparisons](https://www.sciencedirect.com/science/article/pii/S0021999122009652)

NeuralUQ for Biomechanical constitutive models with experimental data (inferring model parameters from known model and data; inferring functions from pre-trained GAN and data): 
- [A Generative Modeling Framework for Inferring Families of Biomechanical Constitutive Laws in Data-Sparse Regimes](https://arxiv.org/pdf/2305.03184.pdf)

Extensions of NeuralUQ:
- [L-HYDRA: Multi-Head Physics-Informed Neural Networks](https://arxiv.org/abs/2301.02152)

# Cite NeuralUQ

[@article{zou2024neuraluq,
  title={NeuralUQ: A Comprehensive Library for Uncertainty Quantification in Neural Differential Equations and Operators},
  author={Zou, Zongren and Meng, Xuhui and Psaros, Apostolos F and Karniadakis, George E},
  journal={SIAM Review},
  volume={66},
  number={1},
  pages={161--190},
  year={2024},
  publisher={SIAM}
}](https://epubs.siam.org/doi/abs/10.1137/22M1518189)
