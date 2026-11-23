# Physics-Informed Latent PDE World Model

A modular PyTorch prototype exploring Physics-Informed Latent World Models, replacing unconstrained recurrent or Transformer latent transitions with a structured spatial latent state governed by a discrete PDE residual operator and a residual neural correction, paired with latent Model Predictive Control (Cross-Entropy Method) for planning.

---

## 1. Motivation & Theoretical Overview

Standard world models (such as RSSM / Dreamer-style architectures) typically compress high-dimensional observations into flat or token-based latent vectors and evolve them using black-box sequence models (GRUs, MLPs, or Transformers). While expressive, unconstrained latent transitions can drift over long imagination horizons and lack physical interpretability.

In **Physics-Informed Latent World Models**, we address this by:
1. **Structuring the latent space spatially** on a coarse 2D grid $(H_z, W_z)$ with physically typed channels:
   $$z_t = \big(u_t,\; v_t,\; p_t,\; r_t\big) \in \mathbb{R}^{4 \times H_z \times W_z}$$
   - $(u_t, v_t)$: latent 2D velocity components.
   - $p_t$: latent scalar / pressure-like field.
   - $r_t$: latent residual / action-modulated forcing channel.
2. **Coupling explicit PDE tendencies with a neural corrector**:
   $$z_{t+1} = \operatorname{clip}\Big(z_t + \Delta t \big(\mathcal{F}_{\text{PDE}}(z_t, a_t) + f_{\theta}(z_t, a_t)\big),\; -5,\; 5\Big)$$
   where $\mathcal{F}_{\text{PDE}}$ computes coupled 2D advection-diffusion and forcing tendencies using differentiable finite-difference operators (`SpatialDifferentialOperators`), with learnable diffusion coefficients ($\nu, \kappa > 0$ via `softplus`) and damping ($\gamma$).
3. **Training with a multi-term physics-aware objective**:
   $$\mathcal{L} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{dyn}} \mathcal{L}_{\text{dyn}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{rollout}} \mathcal{L}_{\text{rollout}} + \mathcal{L}_{\text{reg}}$$
   - $\mathcal{L}_{\text{rec}}$: observation reconstruction MSE across sequence steps.
   - $\mathcal{L}_{\text{dyn}}$: one-step latent transition alignment against posterior latents $z_{t+1}^{\text{post}}$.
   - $\mathcal{L}_{\text{pde}}$: Smooth $L_1$ penalty on the discrete PDE residual $(\hat{z}_{t+1} - \hat{z}_t)/\Delta t - \mathcal{F}_{\text{PDE}}(\hat{z}_{t+1}, a_t)$.
   - $\mathcal{L}_{\text{rollout}}$: multi-step open-loop imagination loss (`imagine_rollout`) without future observations.
   - $\mathcal{L}_{\text{reg}}$: mild Gaussian prior regularization on latent mean and log-variance.

> **Research Caveat (Identifiability & Gauge Freedom):** Without strong inductive bias or physical supervision, latent fields do not automatically carry metric units and may exhibit coordinate/scaling symmetries. Channel typing, spatial grid structure, and PDE residual penalties help anchor the dynamics, though careful ablation is recommended when transferring to real-world visual or sensor streams.

---

## 2. Repository Structure

```text
physics_dreamer/
├── README.md
├── pyproject.toml
├── configs/
│   └── navier_stokes_2d.yaml
├── src/
│   └── pde_world_model/
│       ├── __init__.py
│       ├── config.py # Typed dataclasses: WorldModelConfig, PlannerConfig
│       ├── synthetic_env.py # SyntheticFluidEnv: 2D nonlinear advection-diffusion simulator
│       ├── pde_operators.py # SpatialDifferentialOperators & LatentPDEResidual
│       ├── networks.py # SpatialEncoder, SpatialDecoder, ResidualPDEBlock, NeuralTransitionOperator
│       ├── world_model.py # PhysicsInformedLatentWorldModel (transition, rollout, loss)
│       ├── planner.py # LatentPDEPlannerCEM (latent Cross-Entropy Method planner)
│       └── trainer.py # Training loop & metric logging
└── scripts/
    └── train_and_eval.py 
```
