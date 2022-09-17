We strongly recommend the users to start with the `function.py` example to go through the whole UQ procedure in **NeuralUQ** library.

# Supported examples

- Function approximation for discontinuous function (Samplable/Trainable/Variational): `function.py`

- Inverse problem for Kraichnan Orszag system (Samplable/Trainable/Variational): `invserse_ko.py`

- Inverse problem for Korteweg-de Vries equation (Samplable/Trainable): `inverse_kdv.py`

- Model identification -- SIRD model with Italy COVID-19 data (first outburst in early 2020) (Samplable/Trainable): `inverse_sird.py`

- 100-dimensional Darcy problem using PA-BNN-FP (Samplable): `Darcy_NN.py`

- 100-dimensional Darcy problem using PA-GAN-FP (Samplable): `Darcy_FP.py`

- 100-dimensional Darcy problem using Uncertain DeepONet, i.e., DeepONet + deep ensemble (tensorflow.compat.v1, Trainable): `Darcy_operator.py`

- Sine wave regression using different surrogates: `appendix/sine_wave_regression.ipynb`

- Forward problem for 1-D nonlinear Poisson equation (Samplable), from B-PINN paper: `forward_poisson_bpinn.py`

- Inverse problem for 1-D nonlinear Poisson equation (Samplable), from B-PINN paper: `inverse_poisson_bpinn.py`

- Forward problem for 1-D steady nonlinear diffusion reaction equation (Samplable), from UQ review paper: `forward_diffusion_reaction.py`

- Monte Carlo dropout in variational inference fashion for uncertainty quantification in scientific machine learning: `Monte_Carlo_dropout_for_SciML.ipynb` 
