from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    """
    Encodes a high-dimensional observation (B, C_in, H, W) into a diagonal Gaussian distribution 
    over the structured latent grid (B, 4, H_z, W_z).
    
    Encode une observation haute dimension (B, C_in, H, W) en une distribution
    gaussienne diagonale sur la grille latente structurée (B, 4, H_z, W_z).
    """
    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int],
        latent_grid: Tuple[int, int],
        hidden_dim: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.latent_grid = latent_grid
        self.latent_channels = latent_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
        )

        self.to_stats = nn.Conv2d(hidden_dim * 2, latent_channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, self.latent_grid)
        stats = self.to_stats(h)
        mean, logvar = torch.chunk(stats, chunks=2, dim=1)
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar


class SpatialDecoder(nn.Module):
    """ Decodes a structured latent state z (B, 4, H_z, W_z) back into the observation space (B, C_out, H, W).
        Décode un état latent structuré z (B, 4, H_z, W_z) vers l'espace d'observation (B, C_out, H, W). """
    def __init__(
        self,
        out_channels: int,
        img_size: Tuple[int, int],
        latent_grid: Tuple[int, int],
        hidden_dim: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_grid = latent_grid

        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        if out.shape[-2:] != self.img_size:
            out = F.interpolate(out, size=self.img_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


class ResidualPDEBlock(nn.Module):
    """ Spatial residual block with dilated convolutions to capture non-local couplings.
        Bloc résiduel spatial avec connexions dilatées pour capturer les couplages non locaux."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.silu(x + h)


class NeuralTransitionOperator(nn.Module):
    """ Predicts the neural-network-corrected latent increment \Delta z_nn(z_t, a_t) to complement or refine the latent PDE integration.
        Prédit l'incrément latent corrigé par neural network \Delta z_nn(z_t, a_t) afin de compléter ou raffiner l'intégration de l'EDP latente. """
    def __init__(self, action_dim: int, latent_channels: int = 4, hidden_channels: int = 32):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.in_proj = nn.Conv2d(latent_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.block1 = ResidualPDEBlock(hidden_channels)
        self.block2 = ResidualPDEBlock(hidden_channels)
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        # Conservative near-zero initialization for initial stability / Initialisation prudente proche de zéro pour stabilité initiale
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, _, h, w = z.shape
        a_emb = self.action_mlp(action).view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([z, a_emb], dim=1)
        feat = self.block2(self.block1(F.silu(self.in_proj(x))))
        return self.out_proj(feat)
