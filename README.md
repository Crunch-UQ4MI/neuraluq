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
  - [B-PINNs: Bayesian physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999120306872)
  - [Learning functional priors and posteriors from data and physics](https://www.sciencedirect.com/science/article/pii/S0021999122001358)
  - [Physics-informed generative adversarial networks for stochastic differential equations](https://epubs.siam.org/doi/abs/10.1137/18M1225409)
  - ...
- UQ for DeepONets
  - [Learning functional priors and posteriors from data and physics](https://www.sciencedirect.com/science/article/pii/S0021999122001358)
  - [Bayesian DeepONets](https://arxiv.org/abs/2111.02484)
  - [Randomized priors for DeepONets](https://www.sciencedirect.com/science/article/pii/S0045782522004595)
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
- [NeuralUQ: A comprehensive library for uncertainty quantification in neural differential equations and operators](https://epubs.siam.org/doi/abs/10.1137/22M1518189)
- [Uncertainty quantification in scientific machine learning: Methods, metrics, and comparisons](https://www.sciencedirect.com/science/article/pii/S0021999122009652)
- [Uncertainty quantification for noisy inputs-outputs in physics-informed neural networks and neural operators](https://www.sciencedirect.com/science/article/pii/S0045782524007333)

NeuralUQ for physical model misspecification and uncertainty:
- [Correcting model misspecification in physics-informed neural networks (PINNs)](https://www.sciencedirect.com/science/article/pii/S0021999124001670)

NeuralUQ for physics-informed Kolmogorov-Arnold networks (PIKANs):
- [A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks](https://www.sciencedirect.com/science/article/pii/S0045782524005462)

NeuralUQ for Biomechanical constitutive models with experimental data (inferring model parameters from known model and data; inferring functions from pre-trained GAN and data): 
- [A generative modeling framework for inferring families of biomechanical constitutive laws in data-sparse regimes](https://www.sciencedirect.com/science/article/pii/S0022509623002284?dgcid=rss_sd_all)

NeuralUQ for learning and discovering multiple solutions:
- [Learning and discovering multiple solutions using physics-informed neural networks with random initialization and deep ensemble](https://arxiv.org/abs/2503.06320)

Extensions of NeuralUQ:
- [Multi-head physics-informed neural networks for learning functional priors and uncertainty quantification](https://www.sciencedirect.com/science/article/abs/pii/S002199912500230X)

# Cite NeuralUQ
```
@article{zou2024neuraluq,
  title={NeuralUQ: A Comprehensive Library for Uncertainty Quantification in Neural Differential Equations and Operators},
  author={Zou, Zongren and Meng, Xuhui and Psaros, Apostolos F and Karniadakis, George E},
  journal={SIAM Review},
  volume={66},
  number={1},
  pages={161--190},
  year={2024},
  publisher={SIAM}
}
```


# The Team

NeuralUQ was developed by Zongren Zou and Xuhui Meng under the supervision of [Professor George Em Karniadakis](https://sites.brown.edu/crunch-group/) at [Brown University](https://www.brown.edu/) between 2022 and 2024, with helpful discussion and invaluable support from [Dr. Apostolos F Psaros](https://www.afpsaros.com/) and [Professor Ling Guo](https://scholar.google.com/citations?user=Ys5ZVhEAAAAJ&hl=en). The project is currently maintained by Zongren Zou at California Institute of Technology and Xuhui Meng at Huazhong University of Science and Technology.
