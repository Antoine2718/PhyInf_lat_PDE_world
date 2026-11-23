from typing import Tuple
import torch

from .config import PlannerConfig
from .world_model import PhysicsInformedLatentWorldModel


class LatentPDEPlannerCEM:
    """ MPC planner via Cross-Entropy Method (CEM) in the physical latent space.
    Example objective: move the latent center of mass/energy toward a target (tx, ty)
    within the normalized latent grid [-1, 1]^2.
    
    Planificateur MPC par Cross-Entropy Method (CEM) dans l'espace latent physique.
    Exemple d'objectif : déplacer le centre de masse/énergie latente vers une cible (tx, ty)
    dans la grille latente normalisée [-1, 1]^2.
    """
    def __init__(self, world_model: PhysicsInformedLatentWorldModel, cfg: PlannerConfig):
        self.wm = world_model
        self.cfg = cfg

    @torch.no_grad()
    def plan(self, obs_curr: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """ obs_curr: (1, C, H, W) Returns the first optimal action (1, action_dim) and the minimum cost.
        Retourne la première action optimale (1, action_dim) et le coût minimal.
        """
        self.wm.eval()
        device = obs_curr.device
        z0, _, _ = self.wm.encoder(obs_curr)
        _, _, hz, wz = z0.shape

        # Latent spatial coordinates in [-1, 1] / Coordonnées spatiales latentes dans [-1, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, hz, device=device),
            torch.linspace(-1.0, 1.0, wz, device=device),
            indexing="ij",
        )
        tx, ty = self.cfg.target_pos

        mean = torch.zeros((self.cfg.horizon, self.wm.cfg.action_dim), device=device)
        std = torch.ones((self.cfg.horizon, self.wm.cfg.action_dim), device=device) * 0.5

        best_cost = float("inf")
        best_seq = mean.clone()

        for _ in range(self.cfg.cem_iters):
            eps = torch.randn((self.cfg.num_samples, self.cfg.horizon, self.wm.cfg.action_dim), device=device)
            actions = torch.clamp(
                mean.unsqueeze(0) + std.unsqueeze(0) * eps,
                self.cfg.action_min,
                self.cfg.action_max,
            )

            z_init_batch = z0.expand(self.cfg.num_samples, -1, -1, -1)
            z_traj, _ = self.wm.imagine_rollout(z_init_batch, actions)

            # Kinetic + scalar energy as spatial weights: (N, H+1, Hz, Wz) / Énergie cinétique + scalaire comme poids spatial : (N, H+1, Hz, Wz)
            u = z_traj[:, :, 0]
            v = z_traj[:, :, 1]
            p = z_traj[:, :, 2]
            weights = torch.relu(p) + 0.5 * (u ** 2 + v ** 2) + 1e-4
            w_sum = weights.sum(dim=(-2, -1), keepdim=True)
            norm_w = weights / w_sum

            cx = (norm_w * xx.view(1, 1, hz, wz)).sum(dim=(-2, -1))
            cy = (norm_w * yy.view(1, 1, hz, wz)).sum(dim=(-2, -1))

            # Terminal horizon cost + action effort / Coût sur la fin de l'horizon + effort d'action
            pos_err = ((cx[:, -1] - tx) ** 2 + (cy[:, -1] - ty) ** 2)
            act_reg = 0.02 * (actions ** 2).mean(dim=(1, 2))
            costs = pos_err + act_reg

            elite_idxs = torch.topk(costs, k=self.cfg.num_elites, largest=False).indices
            elites = actions[elite_idxs]

            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=0.05)

            iter_best_cost = float(costs[elite_idxs[0]].item())
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_seq = elites[0]

        return best_seq[0:1], best_cost
