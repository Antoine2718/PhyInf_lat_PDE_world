# Physics-Informed Latent World Model (`pde_world_model`)

A prototype combining spatial latent state representations with explicit 2D differential operators (PDE residuals) and a residual neural transition block for short-horizon rollouts and Cross-Entropy Method (CEM) planning.

## Overview

Standard latent world models (such as RSSM / Dreamer-style architectures) typically model latent dynamics with generic recurrent or attention blocks (`GRU`, `Transformer`, or MLPs). While expressive, unconstrained latent transitions can drift out-of-distribution on long horizons or learn arbitrary gauge symmetries with no physical meaning.

This repository explores a structured alternative:
1. **Spatially Typed Latent Grid:** Observations $x_t \in \mathbb{R}^{C \times H \times W}$ are encoded onto a coarser 2D latent grid $z_t \in \mathbb{R}^{4 \times H_z \times W_z}$ with channels assigned to `u`, `v` (2D velocity-like components), `p` (scalar/pressure-like field), and `r` (latent forcing/residual modulation).
2. **Hybrid Latent Transition:** Each step advances via an explicit finite-difference PDE right-hand side (`LatentPDEResidual`) plus a residual convolutional transition block (`NeuralTransitionOperator`) conditioned on the action $a_t$.
3. **PDE Residual Loss:** Penalizes discrepancies between discrete time derivatives and the explicit PDE tendencies, with learnable diffusion ($\nu, \kappa$) and damping ($\gamma$) parameters.
4. **Latent CEM Planning:** Rolls out candidate action sequences in latent space and optimizes an energy/scalar-weighted target objective via Cross-Entropy Method.

## Honest Scope & Limitations

This codebase is a **minimal research prototype**, not a full DreamerV3 reproduction or a high-accuracy CFD solver:

- **No Guaranteed Physical Identifiability:** A latent space trained with pixel/observation reconstruction and PDE residuals can still exhibit latent gauge symmetries (scaling, shifting, or mixing channels). Learned $\nu$, $\kappa$, and $\gamma$ reflect the *latent* dynamics rather than guaranteed ground-truth physical constants unless anchored by direct state supervision or stronger structural constraints.
- **Explicit Euler Discretization:** Both the synthetic test environment (`SyntheticFluidEnv`) and the latent transition use explicit Euler steps (`dt`), which require small time steps for numerical stability and can become stiff under high diffusivity or strong advection.
- **No Incompressibility Projection Yet:** Despite fluid-inspired notation (`u`, `v`, `p`), there is currently no Helmholtz–Hodge projection or divergence-free constraint enforced on `(u, v)`.
- **Synthetic Benchmark Only:** Included data comes from a fast 2D synthetic advection-diffusion environment designed for end-to-end debugging and rapid iteration, not real-world video or external physics benchmarks.
- **Planning Objective:** `LatentPDEPlannerCEM` illustrates trajectory optimization over imagined latent rollouts using a heuristic spatial moment cost; it does not yet include an actor-critic policy or learned reward/value head.

## Repository Structure

```text
physics_dreamer/
├── README.md
├── pyproject.toml
├── configs/
│   └── navier_stokes_2d.yaml
├── src/
│   └── pde_world_model/
│       ├── __init__.py
│       ├── config.py
│       ├── synthetic_env.py
│       ├── pde_operators.py
│       ├── networks.py
│       ├── world_model.py
│       ├── planner.py
│       └── trainer.py
└── scripts/
    └── train_and_eval.py
```
## Quickstart
Requirements : 
- Python 3.10+
- PyTorch
### Run End-to-End Training & CEM Evaluation
From the repository root:
```bash 
python scripts/train_and_eval.py
````
The script generates synthetic 2D fluid-like trajectories, trains PhysicsInformedLatentWorldModel, logs loss components (rec, dyn, pde, roll) and learned parameters (nu, kappa), and runs a single CEM planning step on a sample observation.

### Key Configuration Knobs (src/pde_world_model/config.py)
- latent_grid: Spatial resolution (H_z, W_z) of the structured latent state.
- Bullet dt, dx, dy: Temporal and spatial grid spacing used in discrete differential operators.
- Bullet init_nu, init_kappa, init_damping: Initial values for learnable PDE coefficients.
- Bullet w_rec, w_dyn, w_pde, w_rollout: Loss weights balancing observation reconstruction, one-step latent dynamics, PDE residual minimization, and multi-step imagined rollouts.
- Bullet rollout_horizon: Number of open-loop latent steps rolled out during training.
