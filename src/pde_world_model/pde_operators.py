import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDifferentialOperators(nn.Module):
    """
    Differentiable 2D discrete differential operators on a latent grid (B, C, H_z, W_z). 
    Uses replication padding to avoid boundary artifacts.
    
    Opérateurs différentiels discrets 2D différentiables sur une grille latente (B, C, H_z, W_z).
    Utilise un padding par réplication pour éviter les artefacts de bord.
    """
    def __init__(self, dx: float = 1.0 / 16.0, dy: float = 1.0 / 16.0):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def grad_x(self, f: torch.Tensor) -> torch.Tensor:
        # f: (B, C, H, W)
        padded = F.pad(f, (1, 1, 0, 0), mode="replicate")
        return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * self.dx)

    def grad_y(self, f: torch.Tensor) -> torch.Tensor:
        padded = F.pad(f, (0, 0, 1, 1), mode="replicate")
        return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * self.dy)

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        px = F.pad(f, (1, 1, 0, 0), mode="replicate")
        py = F.pad(f, (0, 0, 1, 1), mode="replicate")
        d2x = (px[:, :, :, 2:] - 2.0 * f + px[:, :, :, :-2]) / (self.dx ** 2)
        d2y = (py[:, :, 2:, :] - 2.0 * f + py[:, :, :-2, :]) / (self.dy ** 2)
        return d2x + d2y

    def divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.grad_x(u) + self.grad_y(v)


class LatentPDEResidual(nn.Module):
    """
    Computes the physical tendency of a structured latent state z = [u, v, p, r] (4 channels) 
    and evaluates the PDE residual between z_next and z_curr under action a. 
    The diffusion (nu, kappa) and damping (gamma) coefficients are learnable.
    
    Calcule la tendance physique d'un état latent structuré z = [u, v, p, r] (4 canaux)
    et évalue le résidu d'EDP entre z_next et z_curr sous l'action a.
    Les coefficients de diffusion (nu, kappa) et d'amortissement (gamma) sont apprenables.
    """

    def __init__(
        self,
        action_dim: int,
        dt: float = 0.05,
        dx: float = 1.0 / 16.0,
        dy: float = 1.0 / 16.0,
        init_nu: float = 0.02,
        init_kappa: float = 0.02,
        init_damping: float = -0.05,
    ):
        super().__init__()
        self.dt = dt
        self.ops = SpatialDifferentialOperators(dx=dx, dy=dy)

        # Positive diffusion parameterization / Paramétrisation positive pour diffusion
        self._raw_nu = nn.Parameter(torch.tensor(float(init_nu)).clamp(min=1e-4).log())
        self._raw_kappa = nn.Parameter(torch.tensor(float(init_kappa)).clamp(min=1e-4).log())
        self.gamma = nn.Parameter(torch.tensor(float(init_damping)))

        # Action projection to spatial forcing (3 channels: u, v, p) / Projection d'action vers un forçage spatial (3 canaux : u, v, p)
        self.action_to_forcing = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )

    @property
    def nu(self) -> torch.Tensor:
        return F.softplus(self._raw_nu) + 1e-4

    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self._raw_kappa) + 1e-4

    def pde_rhs(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns physical dz/dt of shape (B, 4, H_z, W_z).
        """
        u = z[:, 0:1]
        v = z[:, 1:2]
        p = z[:, 2:3]
        r = z[:, 3:4]

        du_dx, du_dy = self.ops.grad_x(u), self.ops.grad_y(u)
        dv_dx, dv_dy = self.ops.grad_x(v), self.ops.grad_y(v)
        dp_dx, dp_dy = self.ops.grad_x(p), self.ops.grad_y(p)

        # Uniform action forcing modulated by the residual channel r / Forçage d'action uniforme modulé par le canal résiduel r
        act_f = self.action_to_forcing(action).view(z.size(0), 3, 1, 1)
        mod = 1.0 + 0.25 * torch.tanh(r)

        rhs_u = (
            -(u * du_dx + v * du_dy)
            - 0.25 * dp_dx
            + self.nu * self.ops.laplacian(u)
            + self.gamma * u
            + act_f[:, 0:1] * mod
        )
        rhs_v = (
            -(u * dv_dx + v * dv_dy)
            - 0.25 * dp_dy
            + self.nu * self.ops.laplacian(v)
            + self.gamma * v
            + act_f[:, 1:2] * mod
        )
        rhs_p = (
            -self.ops.divergence(p * u, p * v)
            + self.kappa * self.ops.laplacian(p)
            + act_f[:, 2:3]
        )
        rhs_r = -0.2 * r

        return torch.cat([rhs_u, rhs_v, rhs_p, rhs_r], dim=1)

    def residual(
        self,
        z_next: torch.Tensor,
        z_curr: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        PDE residual R = (z_next - z_curr) / dt - F_PDE(z_next, action). 
        Returns the mean scalar loss.
        
        Résidu d'EDP R = (z_next - z_curr)/dt - F_PDE(z_next, action). Retourne une perte scalaire moyenne.
        """
        time_deriv = (z_next - z_curr) / self.dt
        rhs = self.pde_rhs(z_next, action)
        res = time_deriv - rhs
        return F.smooth_l1_loss(res, torch.zeros_like(res), beta=0.5)
