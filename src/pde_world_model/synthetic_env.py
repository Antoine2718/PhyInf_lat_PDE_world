import math
from typing import Tuple
import torch


class SyntheticFluidEnv:
    """
    Fast 2D nonlinear advection-diffusion simulator forced by an action \((u_{\text{act}}, v_{\text{act}})\) 
    injecting a Gaussian impulse into the velocity field. Enables end-to-end testing of 
    the world model without external dependencies.
    
    Simulateur 2D rapide d'advection-diffusion non linéaire forcée par une action (u_act, v_act)
    injectant une impulsion gaussienne dans le champ de vitesse.
    Permet de tester le world model de bout en bout sans dépendance externe.
    """ 
    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        dt: float = 0.05,
        nu: float = 0.015,
        kappa: float = 0.015,
        device: str = "cpu",
    ):
        self.height = height
        self.width = width
        self.dt = dt
        self.nu = nu
        self.kappa = kappa
        self.device = torch.device(device)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=self.device),
            torch.linspace(-1.0, 1.0, width, device=self.device),
            indexing="ij",
        )
        self.xx = xx
        self.yy = yy
        self.dx = 2.0 / width
        self.dy = 2.0 / height

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        # Periodic boundary conditions implemented via roll (suitable for spectral tests and periodic fluid simulations).
        # Conditions aux limites périodiques via roll (adapté aux tests spectraux/fluides périodiques)
        d2x = (
            torch.roll(f, shifts=-1, dims=-1)
            - 2.0 * f
            + torch.roll(f, shifts=1, dims=-1)
        ) / (self.dx ** 2)
        d2y = (
            torch.roll(f, shifts=-1, dims=-2)
            - 2.0 * f
            + torch.roll(f, shifts=1, dims=-2)
        ) / (self.dy ** 2)
        return d2x + d2y

    def _grad_x(self, f: torch.Tensor) -> torch.Tensor:
        return (torch.roll(f, shifts=-1, dims=-1) - torch.roll(f, shifts=1, dims=-1)) / (2.0 * self.dx)

    def _grad_y(self, f: torch.Tensor) -> torch.Tensor:
        return (torch.roll(f, shifts=-1, dims=-2) - torch.roll(f, shifts=1, dims=-2)) / (2.0 * self.dy)

    def reset(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne u, v, p de forme (B, H, W)."""
        u = torch.zeros((batch_size, self.height, self.width), device=self.device)
        v = torch.zeros((batch_size, self.height, self.width), device=self.device)
        p = torch.zeros((batch_size, self.height, self.width), device=self.device)

        for b in range(batch_size):
            cx = torch.empty(1, device=self.device).uniform_(-0.4, 0.4).item()
            cy = torch.empty(1, device=self.device).uniform_(-0.4, 0.4).item()
            sigma = torch.empty(1, device=self.device).uniform_(0.18, 0.35).item()
            amp = torch.empty(1, device=self.device).uniform_(0.6, 1.2).item()

            blob = amp * torch.exp(-((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2.0 * sigma ** 2))
            # Vortical field + associated scalar p / Champ tourbillonnaire + scalaire p associé
            u[b] = -blob * self.yy
            v[b] = blob * self.xx
            p[b] = blob

        return u, v, p

    def step(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        state: (u, v, p) chacun (B, H, W)
        action: (B, 2) dans [-1, 1]
        Retourne next_state et obs_rgb de forme (B, 3, H, W) dans [0, 1].
        """
        u, v, p = state
        ax = action[:, 0].view(-1, 1, 1)
        ay = action[:, 1].view(-1, 1, 1)

        # Localized forcing around the origin / Forçage localisé autour de l'origine
        forcing_mask = torch.exp(-(self.xx ** 2 + self.yy ** 2) / (2.0 * 0.35 ** 2)).unsqueeze(0)
        fx = 0.5 * ax * forcing_mask
        fy = 0.5 * ay * forcing_mask

        du_dt = (
            -(u * self._grad_x(u) + v * self._grad_y(u))
            - 0.25 * self._grad_x(p)
            + self.nu * self._laplacian(u)
            + fx
        )
        dv_dt = (
            -(u * self._grad_x(v) + v * self._grad_y(v))
            - 0.25 * self._grad_y(p)
            + self.nu * self._laplacian(v)
            + fy
        )
        dp_dt = (
            -(self._grad_x(p * u) + self._grad_y(p * v))
            + self.kappa * self._laplacian(p)
        )

        u_next = torch.clamp(u + self.dt * du_dt, -3.0, 3.0)
        v_next = torch.clamp(v + self.dt * dv_dt, -3.0, 3.0)
        p_next = torch.clamp(p + self.dt * dp_dt, -3.0, 3.0)

        obs = self.render(u_next, v_next, p_next)
        return (u_next, v_next, p_next), obs

    def render(self, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Converts (u, v, p) into a normalized RGB image / Convertit (u, v, p) en image RGB normalisée 
        c0 = torch.sigmoid(2.0 * u)
        c1 = torch.sigmoid(2.0 * v)
        c2 = torch.sigmoid(2.0 * p)
        return torch.stack([c0, c1, c2], dim=1)

    def rollout_dataset(self, num_trajectories: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Generates (obs_seq, act_seq) with shapes (N, T, 3, H, W) and (N, T, 2), respectively.
           Génère (obs_seq, act_seq) de formes (N, T, 3, H, W) et (N, T, 2). '''
        obs_list = []
        act_list = []

        state = self.reset(num_trajectories)
        obs_0 = self.render(*state)

        # Initial action
        prev_act = torch.zeros((num_trajectories, 2), device=self.device)
        for t in range(seq_len):
            # Temporally smoothed actions (OU-like) / Actions lissées temporellement 
            noise = torch.randn((num_trajectories, 2), device=self.device) * 0.35
            act = torch.clamp(0.75 * prev_act + noise, -1.0, 1.0)
            if t == 0:
                obs_t = obs_0
            else:
                state, obs_t = self.step(state, act)
            obs_list.append(obs_t)
            act_list.append(act)
            prev_act = act

        obs_seq = torch.stack(obs_list, dim=1)
        act_seq = torch.stack(act_list, dim=1)
        return obs_seq, act_seq
