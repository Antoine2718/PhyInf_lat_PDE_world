from typing import Dict
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import WorldModelConfig
from .world_model import PhysicsInformedLatentWorldModel


def train_world_model(
    model: PhysicsInformedLatentWorldModel,
    obs_data: torch.Tensor,
    act_data: torch.Tensor,
    cfg: WorldModelConfig,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.to(device)

    dataset = TensorDataset(obs_data, act_data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    last_metrics: Dict[str, float] = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        agg: Dict[str, float] = {}
        num_batches = 0

        for obs_b, act_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.compute_loss(obs_b, act_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            num_batches += 1

        last_metrics = {k: v / max(num_batches, 1) for k, v in agg.items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
                f"loss={last_metrics['loss']:.4f} | "
                f"rec={last_metrics['rec']:.4f} | "
                f"dyn={last_metrics['dyn']:.4f} | "
                f"pde={last_metrics['pde']:.4f} | "
                f"roll={last_metrics['roll']:.4f} | "
                f"nu={last_metrics['nu']:.4f} | "
                f"kappa={last_metrics['kappa']:.4f}"
            )

    return last_metrics
