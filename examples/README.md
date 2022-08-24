We strongly recommend starting from `function.py` example to go through and be familiar with the whole UQ procedure and the NeuralUQ library.

# Supported examples

Function approximation for discontinuous function (Samplable/Trainable/Variational): `function.py`

Inverse problem on Kraichnan Orszag system (Samplable/Trainable/Variational): `invserse_ko.py`

Inverse problem on Korteweg-de Vries equation (Samplable/Trainable): `inverse_kdv.py`

Inverse problem on SIRD model with Italy COVID-19 data (first outburst in early 2020) (Samplable/Trainable): `inverse_sird.py`

Function approximation for 100-dimensional Darcy problem with neural network and DeepONet (Samplable): `Darcy_NN.py`

Function approximation for 100-dimensional Darcy problem with functional prior (FP) and DeepONet (Samplable): `Darcy_FP.py`

# Working in progress

Operator learning for Darcy problem, from parameter to solution, with DeepONet and deep ensemble method (tensorflow.compat.v1, Trainable): `Darcy_operator.py`

Sine wave regression using different surrogates: `sine_wave_regression.ipynb`

A demonstration of Monte Carlo Dropout in scientific machine learning: `MCD4SciML.ipynb`
