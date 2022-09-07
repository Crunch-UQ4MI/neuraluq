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
- [NeuralUQ: A comprehensive library for uncertainty quantification in neural differential equations and operators](http://arxiv.org/abs/2208.11866)

Users can refer to the following papers for more details on the algorithms:
- [A comprehensive review on uncertainty quantification in scientific machine learning](https://arxiv.org/pdf/2201.07766.pdf)
- UQ for physics-informed neural networks
  - [B-PINNs: Bayesian Physics-informed Networks](https://www.sciencedirect.com/science/article/pii/S0021999120306872)
  - [Learning Functional Priors and Posteriors from Data and Physics](https://www.sciencedirect.com/science/article/pii/S0021999122001358)
  - ...
- UQ for DeepONets
  - [Learning Functional Priors and Posteriors from Data and Physics](https://www.sciencedirect.com/science/article/pii/S0021999122001358)
  - [Bayesian DeepONets](https://arxiv.org/pdf/2111.02484.pdf)
  - [Randomized Priors](https://arxiv.org/pdf/2203.03048.pdf)
  - ...
# Installation
**NeuralUQ** requires the following dependencies to be installed:

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

# Cite NeuralUQ

[@misc{zou2022neuraluq, <br />
    title={NeuralUQ: A comprehensive library for uncertainty quantification in neural differential equations and operators}, <br />
    author={Zongren Zou, Xuhui Meng, Apostolos F Psaros, and George Em Karniadakis}, <br />
    year={2022}, <br />
    eprint={2208.11866}, <br />
    archivePrefix={arXiv}, <br />
    primaryClass={cs.LG} <br />
}](http://arxiv.org/abs/2208.11866)
