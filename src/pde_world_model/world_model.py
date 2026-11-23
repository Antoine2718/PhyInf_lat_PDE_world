from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WorldModelConfig
from .networks import NeuralTransitionOperator, SpatialDecoder, SpatialEncoder
from .pde_operators import LatentPDEResidual

class PhysicsInformedLatentWorldModel(nn.Module): 
    """ PDE-constrained latent world model:
    - Encodes x_t -> z_t (channel 0: u, channel 1: v, channel 2: p, channel 3: r)
    - Evolves z_t -> z_{t+1}^{pred} via a hybrid scheme [physics-based PDE + neural correction]
    - Explicitly penalizes the PDE residual, reconstruction error, and multi-step rollouts.
    
    World Model latent contraint par EDP :
    - Encode x_t -> z_t (canal 0: u, canal 1: v, canal 2: p, canal 3: r)
    - Fait évoluer z_t -> z_{t+1}^{pred} via un schéma hybride [EDP physique + correction neuronale]
    - Pénalise explicitement le résidu d'EDP, la reconstruction et les rollouts multi-pas. """
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpatialEncoder(
            in_channels=cfg.in_channels,
            img_size=cfg.img_size,
            latent_grid=cfg.latent_grid,
            hidden_dim=cfg.hidden_dim,
            latent_channels=4,
        )
        self.decoder = SpatialDecoder(
            out_channels=cfg.in_channels,
            img_size=cfg.img_size,
            latent_grid=cfg.latent_grid,
            hidden_dim=cfg.hidden_dim,
            latent_channels=4,
        )
        self.pde_module = LatentPDEResidual(
            action_dim=cfg.action_dim,
            dt=cfg.dt,
            dx=cfg.dx,
            dy=cfg.dy,
            init_nu=cfg.init_nu,
            init_kappa=cfg.init_kappa,
            init_damping=cfg.init_damping,
        )
        self.neural_transition = NeuralTransitionOperator(
            action_dim=cfg.action_dim,
            latent_channels=4,
            hidden_channels=32,
        )

    def transition_step(self, z: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Advances by one time step in the latent space:
        z_{t+1} = z_t + dt * rhs_pde(z_t, a_t) + dt * nn_corr(z_t, a_t)
        Returns (z_next, rhs_pde).
        
        Avance d'un pas de temps dans l'espace latent :
        z_{t+1} = z_t + dt * rhs_pde(z_t, a_t) + dt * nn_corr(z_t, a_t)
        Retourne (z_next, rhs_pde).
        """
        rhs_pde = self.pde_module.pde_rhs(z, action)
        nn_corr = self.neural_transition(z, action)
        z_next = z + self.cfg.dt * (rhs_pde + nn_corr)
        z_next = torch.clamp(z_next, -5.0, 5.0)
        return z_next, rhs_pde

    def imagine_rollout(
        self, z_init: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Rolls out a latent trajectory without observations.
        z_init: (B, 4, H_z, W_z)
        actions: (B, H_roll, action_dim)
        Returns z_rollout of shape (B, H_roll + 1, 4, H_z, W_z) and obs_rollout of shape (B, H_roll + 1, C, H, W).

        Déroule une trajectoire latente sans observations.
        z_init: (B, 4, H_z, W_z)
        actions: (B, H_roll, action_dim)
        Retourne z_rollout de forme (B, H_roll + 1, 4, H_z, W_z) et obs_rollout (B, H_roll + 1, C, H, W).
        """
        horizon = actions.size(1)
        z_curr = z_init
        z_traj = [z_curr]
        obs_traj = [self.decoder(z_curr)]

        for t in range(horizon):
            z_curr, _ = self.transition_step(z_curr, actions[:, t])
            z_traj.append(z_curr)
            obs_traj.append(self.decoder(z_curr))

        return torch.stack(z_traj, dim=1), torch.stack(obs_traj, dim=1)

    def compute_loss(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        obs_seq: (B, T, C, H, W)
        act_seq: (B, T, action_dim)
        """
        b, seq_len, c, h, w = obs_seq.shape
        flat_obs = obs_seq.reshape(b * seq_len, c, h, w)
        z_post, mean, logvar = self.encoder(flat_obs)
        z_post = z_post.reshape(b, seq_len, 4, *self.cfg.latent_grid)
        mean = mean.reshape(b, seq_len, 4, *self.cfg.latent_grid)
        logvar = logvar.reshape(b, seq_len, 4, *self.cfg.latent_grid)

        # 1. Immediate reconstruction
        rec_obs = self.decoder(z_post.reshape(b * seq_len, 4, *self.cfg.latent_grid)).reshape(b, seq_len, c, h, w)
        loss_rec = F.mse_loss(rec_obs, obs_seq)

        # 2. One-step transition & PDE residual over t = 0 .. T-2 / Transition un pas & résidu EDP sur t = 0 .. T-2
        z_curr = z_post[:, :-1].reshape(b * (seq_len - 1), 4, *self.cfg.latent_grid)
        z_target = z_post[:, 1:].reshape(b * (seq_len - 1), 4, *self.cfg.latent_grid)
        act_step = act_seq[:, 1:].reshape(b * (seq_len - 1), self.cfg.action_dim)

        z_pred, _ = self.transition_step(z_curr, act_step)
        loss_dyn = F.mse_loss(z_pred, z_target.detach()) + 0.25 * F.mse_loss(z_pred, z_target)

        ''' PDE residual evaluated on the latent prediction z_pred with respect to z_curr 
        Résidu d'EDP évalué sur la prédiction latente z_pred relativement à z_curr '''
        loss_pde = self.pde_module.residual(z_pred, z_curr, act_step)

        # Soft variance regularization to prevent representation collapse / Régularisation douce de variance pour éviter l'effondrement
        loss_reg = 1e-3 * (mean.pow(2).mean() + (logvar.exp() - 1.0 - logvar).mean())

        # 3. Multi-step rollout starting from t = 0 / Rollout multi-pas depuis t = 0
        h_roll = min(self.cfg.rollout_horizon, seq_len - 1)
        if h_roll > 0:
            _, obs_imag = self.imagine_rollout(z_post[:, 0], act_seq[:, 1 : 1 + h_roll])
            loss_roll = F.mse_loss(obs_imag[:, 1:], obs_seq[:, 1 : 1 + h_roll])
        else:
            loss_roll = torch.tensor(0.0, device=obs_seq.device)

        total_loss = (
            self.cfg.w_rec * loss_rec
            + self.cfg.w_dyn * loss_dyn
            + self.cfg.w_pde * loss_pde
            + self.cfg.w_rollout * loss_roll
            + loss_reg
        )

        metrics = {
            "loss": float(total_loss.item()),
            "rec": float(loss_rec.item()),
            "dyn": float(loss_dyn.item()),
            "pde": float(loss_pde.item()),
            "roll": float(loss_roll.item()),
            "nu": float(self.pde_module.nu.item()),
            "kappa": float(self.pde_module.kappa.item()),
        }
        return total_loss, metrics
