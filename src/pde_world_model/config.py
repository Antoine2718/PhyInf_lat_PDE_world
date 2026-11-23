from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class WorldModelConfig:
    # Input and structured latent grid dimensions / Dimensions d'entrée et de grille latente structurée
    in_channels: int = 3
    img_size: Tuple[int, int] = (64, 64)
    latent_grid: Tuple[int, int] = (16, 16)
    action_dim: int = 2
    hidden_dim: int = 64

    # No initial time and physical parameters / Pas de temps et conditions initiales
    dt: float = 0.05
    dx: float = 1.0 / 16.0
    dy: float = 1.0 / 16.0
    init_nu: float = 0.02 # Visocsity / Viscosité / diffusion vitesse
    init_kappa: float = 0.02 # Scalar field diffusion / Diffusion champ scalaire p
    init_damping: float = -0.05

    # Weight of loss terms / Poids des termes de perte
    w_rec: float = 1.0
    w_dyn: float = 1.0
    w_pde: float = 0.5
    w_rollout: float = 0.5
    rollout_horizon: int = 6

    # Training settings / Entraînement
    batch_size: int = 32
    seq_len: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    epochs: int = 25
    device: str = "cpu"
    seed: int = 42


@dataclass
class PlannerConfig:
    horizon: int = 8
    num_samples: int = 128
    num_elites: int = 16
    cem_iters: int = 4
    action_min: float = -1.0
    action_max: float = 1.0
    target_pos: Tuple[float, float] = (0.25, 0.25)  # Normalized coordinates in [-1,1] / coordonnées normalisées dans [-1, 1]
