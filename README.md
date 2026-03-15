# MoE Hyperparameter Tuning — MZM PINN (JAX / Equinox)

> **Branch:** `hyper/moe-jax` — diverges from `hyper/moe`
> PyTorch implementation of this search: `hyper/moe`

Bayesian hyperparameter search for the **Mixture-of-Experts (MoE)** surrogate of Mach-Zehnder Modulators, using **JAX**, **Equinox**, and **Optuna**.

## Overview

MZMs are essential components in optical communication systems. This branch re-implements the MoE hyperparameter search from `hyper/moe` using **JAX / Equinox** with forward-mode automatic differentiation (forward-AD) for the physics-informed loss.

For the base MoE-PINN model see the `master` branch.

## Dataset

The dataset ([Sim_generated_dataset.txt](Sim_generated_dataset.txt)) contains **9,633 electromagnetic simulations** of MZM devices.

### Input Features (Device Geometry)

| Feature | Description |
|---|---|
| `PN_offset` | PN junction offset |
| `Bias_V` | Bias voltage |
| `Core_width` | Waveguide core width |
| `P+_width` | P+ doping region width |
| `N+_width` | N+ doping region width |
| `P_width` | P doping region width |
| `N_width` | N doping region width |
| `Phase_length` | Phase shifter length ($L$) |

### Output Targets (Performance Metrics)

| Target | Description |
|---|---|
| `BW_3dB` | 3 dB electro-optic bandwidth (GHz) |
| `IL` | Insertion loss (dB) |
| `V_pi` | Half-wave voltage (V) |

## Model

A Mixture-of-Experts network implemented in **Equinox**. Hyperparameters searched via Optuna (TPE sampler, median pruner):

- Number of experts
- Hidden dimensions per expert
- Activation function (ReLU, Tanh, Gaussian)
- Learning rate and weight decay (`optax`)
- Physics constraint weights ($\lambda$)

## Physics-Informed Constraints

Physical priors are enforced as penalty terms in the loss via JAX forward-mode AD:

| Constraint | Formulation | Physical Meaning |
|---|---|---|
| BW monotonicity | $\frac{\partial \text{BW}}{\partial L} \leq 0$ | Bandwidth decreases with longer phase length |
| IL monotonicity | $\frac{\partial \text{IL}}{\partial L} \geq 0$ | Insertion loss increases with longer waveguide |
| $V_\pi L$ conservation | $\frac{\partial (V_\pi \cdot L)}{\partial L} \approx 0$ | The $V_\pi L$ product is approximately constant |
| Smoothness | $\left\lVert\frac{\partial^2 \text{BW}}{\partial L^2}\right\rVert$ | Device metrics are smooth functions of geometry |

Each term is weighted by a tunable $\lambda$ optimized via Optuna.

## Project Structure

| File | Description |
|---|---|
| [MZM_Hyperparameter_Tuning_MoE_JAX.ipynb](MZM_Hyperparameter_Tuning_MoE_JAX.ipynb) | Optuna hyperparameter search for MoE (JAX / Equinox) |
| [Sim_generated_dataset.txt](Sim_generated_dataset.txt) | Simulation dataset (9,633 samples) |

## Dependencies

- Python 3.x
- **JAX / Equinox / Optax** — model, training, and forward-mode AD
- **Optuna** — Bayesian hyperparameter optimization
- **scikit-learn** — data preprocessing (`StandardScaler`, `train_test_split`)
- **Matplotlib** — visualization

Install with:

```bash
pip install jax jaxlib equinox optax optuna scikit-learn matplotlib
```

## Usage

1. Place `Sim_generated_dataset.txt` in the working directory.
2. Open `MZM_Hyperparameter_Tuning_MoE_JAX.ipynb` and run all cells.
3. Optuna will run the search and report the best hyperparameter configuration.

## References

Paula, Aldaya, I., Tiago Sutili, Figueiredo, R. C., Pita, J. L., & Bustamante, R. (2023). Design of a silicon Mach–Zehnder modulator via deep learning and evolutionary algorithms. Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-41558-8
