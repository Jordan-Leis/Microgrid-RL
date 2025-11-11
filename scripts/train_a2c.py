import argparse
import os
import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.train_common import build_env
from scripts.metrics_logger import MetricsLogger, make_run_id
from scripts.callbacks import StructuredMetricsCallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--days', type=int, default=180)
    ap.add_argument('--cfg', type=str, default='configs/default.yaml')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--total_timesteps', type=int, default=500_000)
    ap.add_argument('--scenario', type=str, default=None, help='Scenario name (default: auto-generated)')
    ap.add_argument('--log_freq', type=int, default=1000, help='Log metrics every N timesteps')
    ap.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    ap.add_argument('--wandb-project', type=str, default='microgrid-rl', help='W&B project name')
    ap.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username/team)')
    args = ap.parse_args()

    # Load config for metrics
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    
    # Create scenario name if not provided
    scenario = args.scenario or f"lat{args.lat:.2f}_lon{args.lon:.2f}_{args.days}d"
    scenario = scenario.replace('.', '_').replace('-', 'm')  # sanitize filename
    
    # Create run ID and metrics logger
    run_id = make_run_id('a2c', scenario, args.seed)

    metrics_logger = MetricsLogger(
        run_id=run_id,
        algorithm='a2c',
        scenario=scenario,
        seed=args.seed,
        config_path=args.cfg
    )

    def make_env():
        env = build_env(args.cfg, args.lat, args.lon, args.days)
        return Monitor(env)

    env = DummyVecEnv([make_env])
    model = A2C('MlpPolicy', env, verbose=1, seed=args.seed)
    
    # Create callback with environment parameters from config
    callback = StructuredMetricsCallback(
        metrics_logger=metrics_logger,
        log_freq=args.log_freq,
        diesel_cost_per_liter=cfg['diesel']['fuel_cost_per_liter'],
        co2_kg_per_liter=cfg['diesel']['co2_kg_per_liter'],
        step_hours=cfg['time']['step_hours'],
        battery_capacity_kwh=cfg['battery']['capacity_kwh'],
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=['a2c', scenario],
        verbose=1
    )
    
    # Train with structured logging
    os.makedirs('outputs/models', exist_ok=True)
    model_path = f'outputs/models/{run_id}.zip'
    
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(model_path)
    
    # Save summary
    metrics_logger.save_summary({'model_path': model_path})
    
    print(f'Model saved to {model_path}')
    print(f'Metrics logged to {metrics_logger.run_dir}')

if __name__ == '__main__':
    main()
