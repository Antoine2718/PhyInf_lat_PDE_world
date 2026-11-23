import os
import sys
import torch

''' Allows running directly from the repository root without prior installation.
Permet d'exécuter directement depuis la racine du dépôt sans installation préalable '''
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pde_world_model.config import PlannerConfig, WorldModelConfig
from pde_world_model.planner import LatentPDEPlannerCEM
from pde_world_model.synthetic_env import SyntheticFluidEnv
from pde_world_model.trainer import train_world_model
from pde_world_model.world_model import PhysicsInformedLatentWorldModel


def main() -> None:
    wm_cfg = WorldModelConfig(
        img_size=(64, 64),
        latent_grid=(16, 16),
        seq_len=16,
        rollout_horizon=6,
        batch_size=32,
        epochs=25,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )
    torch.manual_seed(wm_cfg.seed)

    env = SyntheticFluidEnv(
        height=wm_cfg.img_size[0],
        width=wm_cfg.img_size[1],
        dt=wm_cfg.dt,
        nu=0.015,
        kappa=0.015,
        device=wm_cfg.device,
    )

    print("Génération des trajectoires d'entraînement...")
    obs_train, act_train = env.rollout_dataset(num_trajectories=256, seq_len=wm_cfg.seq_len)

    model = PhysicsInformedLatentWorldModel(wm_cfg)
    metrics = train_world_model(model, obs_train, act_train, wm_cfg)
    print("\nMétriques finales :", metrics)

    # Evaluation of Cross-Entropy Method (CEM) planning from an initial observation / Test de planification CEM sur une observation initiale
    state_eval = env.reset(batch_size=1)
    obs_eval = env.render(*state_eval)
    planner = LatentPDEPlannerCEM(model, PlannerConfig(horizon=8, num_samples=128))
    best_act, best_cost = planner.plan(obs_eval)
    print(f"Action planifiée CEM : {best_act.squeeze(0).tolist()} | Coût latent : {best_cost:.4f}")


if __name__ == "__main__":
    main()
