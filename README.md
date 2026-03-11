# Physics-Informed Neural Network Modeling of Mach-Zehnder Modulators

Surrogate neural network models for predicting the performance of silicon photonic **Mach-Zehnder Modulators (MZMs)** from geometric design parameters, incorporating physical constraints via a **Physics-Informed Neural Network (PINN)** loss formulation.

## Overview

MZMs are essential components in optical communication systems. Designing them typically requires expensive electromagnetic simulations. This project trains neural network surrogates — **Multi-Layer Perceptrons (MLPs)** and **Mixture-of-Experts (MoE)** networks — that map device geometry to key performance metrics, enforced by physics-based soft constraints through automatic differentiation.

Two deep learning frameworks are explored: **PyTorch** and **JAX** (Flax / Equinox).

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

## Models

### Multi-Layer Perceptron (MLP)

A 5-layer feedforward network with BatchNorm, dropout, and residual connections.

### Mixture of Experts (MoE)

Multiple expert sub-networks whose outputs are blended by a learned gating network (softmax-weighted sum). Supports configurable number of experts, hidden dimensions, and activation functions (ReLU, Tanh, Gaussian).

## Physics-Informed Constraints

Physical priors are enforced as penalty terms in the loss via autograd:

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
| [MZM_MoE_PINN_Model.ipynb](MZM_MoE_PINN_Model.ipynb) | Main notebook — trains MoE-PINN in PyTorch with data-only baseline comparison |
| [MZM_Hyperparameter_Tuning_MoE.ipynb](MZM_Hyperparameter_Tuning_MoE.ipynb) | Optuna hyperparameter search for MoE (PyTorch) |
| [MZM_Hyperparameter_Tuning_MoE_JAX.ipynb](MZM_Hyperparameter_Tuning_MoE_JAX.ipynb) | Optuna hyperparameter search for MoE (JAX / Equinox) |
| [MZM_Hyperparameter_Tuning_MLP.ipynb](MZM_Hyperparameter_Tuning_MLP.ipynb) | Optuna hyperparameter search for MLP (PyTorch) |
| [MZM_Hyperparameter_Tuning_MLP_JAX.ipynb](MZM_Hyperparameter_Tuning_MLP_JAX.ipynb) | Optuna hyperparameter search for MLP (JAX / Flax) |
| [mzm_moe_pinn_model(1).py](mzm_moe_pinn_model(1).py) | Standalone Python export of the main MoE-PINN notebook |
| [best_hyperparams.json](best_hyperparams.json) | Best configuration found by Optuna |
| [best_model.pt](best_model.pt) | Saved PyTorch model weights (state dict) |
| [Sim_generated_dataset.txt](Sim_generated_dataset.txt) | Simulation dataset (9,633 samples) |

## Results

Best configuration found via Bayesian optimization (TPE sampler, median pruner):

| Hyperparameter | Value |
|---|---|
| $\lambda_{\text{BW\_mon}}$ | 0.0 |
| $\lambda_{\text{IL\_mon}}$ | 0.3 |
| $\lambda_{V_\pi L}$ | 0.005 |
| $\lambda_{\text{smooth}}$ | 0.1 |

| Metric | Value |
|---|---|
| Train MSE | 0.0111 |
| Test MSE | 0.0136 |
| Model parameters | 619,503 |

## Dependencies

- Python 3.x
- **PyTorch** — model training and autograd-based physics constraints
- **JAX / Flax / Equinox / Optax** — alternative implementations with forward-mode AD
- **Optuna** — Bayesian hyperparameter optimization
- **scikit-learn** — data preprocessing (`StandardScaler`, `train_test_split`)
- **Matplotlib** — visualization

Install with:

```bash
pip install torch jax jaxlib flax equinox optax optuna scikit-learn matplotlib
```

## Usage

1. Place `Sim_generated_dataset.txt` in the working directory.
2. Open any notebook to explore the corresponding model/framework combination.
3. Run all cells to train the model and visualize results.
4. To use the pre-trained MoE-PINN, load `best_model.pt` with the architecture defined in the main notebook.
