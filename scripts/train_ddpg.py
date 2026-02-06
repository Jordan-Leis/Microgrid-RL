"""
DDPG (Deep Deterministic Policy Gradient) training script for Microgrid RL.

DDPG is an off-policy actor-critic algorithm for continuous action spaces.
It uses a deterministic policy and adds exploration noise during training.
"""
import argparse, os, yaml
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from scripts.train_common import build_env
from scripts.metrics_logger import MetricsLogger, make_run_id
from scripts.callbacks import StructuredMetricsCallback

def main():
    ap = argparse.ArgumentParser(description="Train DDPG agent for microgrid control")
    ap.add_argument('--lat', type=float, required=True, help='Latitude for solar data')
    ap.add_argument('--lon', type=float, required=True, help='Longitude for solar data')
    ap.add_argument('--days', type=int, default=180, help='Number of days to simulate')
    ap.add_argument('--cfg', type=str, default='configs/default.yaml', help='Path to config file')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--total_timesteps', type=int, default=200_000, help='Total training timesteps')
    ap.add_argument('--scenario', type=str, default=None, help='Scenario name (default: auto-generated)')
    ap.add_argument('--log_freq', type=int, default=1000, help='Log metrics every N timesteps')
    ap.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    ap.add_argument('--wandb-project', type=str, default='microgrid-rl', help='W&B project name')
    ap.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username/team)')
    # DDPG-specific arguments
    ap.add_argument('--noise_type', type=str, default='normal', choices=['normal', 'ou'],
                    help='Action noise type: normal (Gaussian) or ou (Ornstein-Uhlenbeck)')
    ap.add_argument('--noise_std', type=float, default=0.1, help='Action noise standard deviation')
    ap.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    ap.add_argument('--buffer_size', type=int, default=1_000_000, help='Replay buffer size')
    ap.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    ap.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target networks')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    args = ap.parse_args()

    # Load config for metrics
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    
    # Create scenario name if not provided
    scenario = args.scenario or f"lat{args.lat:.2f}_lon{args.lon:.2f}_{args.days}d"
    scenario = scenario.replace('.', '_').replace('-', 'm')  # sanitize filename
    
    # Create run ID and metrics logger
    run_id = make_run_id('ddpg', scenario, args.seed)
    metrics_logger = MetricsLogger(
        run_id=run_id,
        algorithm='ddpg',
        scenario=scenario,
        seed=args.seed,
        config_path=args.cfg
    )

    def make_env():
        env = build_env(args.cfg, args.lat, args.lon, args.days)
        return Monitor(env)

    env = DummyVecEnv([make_env])
    
    # Get action space dimensions for noise
    n_actions = env.action_space.shape[-1]
    
    # Create action noise (DDPG uses deterministic policy, noise is added for exploration)
    if args.noise_type == 'ou':
        # Ornstein-Uhlenbeck noise - temporally correlated, good for physical control
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.noise_std * np.ones(n_actions)
        )
    else:
        # Normal (Gaussian) noise - simpler, often works well
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.noise_std * np.ones(n_actions)
        )
    
    # Create DDPG model
    model = DDPG(
        'MlpPolicy',
        env,
        action_noise=action_noise,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        verbose=1,
        seed=args.seed
    )
    
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
        wandb_tags=['ddpg', scenario],
        verbose=1
    )
    
    # Train with structured logging
    os.makedirs('outputs/models', exist_ok=True)
    model_path = f'outputs/models/{run_id}.zip'
    
    print(f"Training DDPG agent with {args.noise_type} noise (std={args.noise_std})")
    print(f"Learning rate: {args.learning_rate}, Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}, Tau: {args.tau}, Gamma: {args.gamma}")
    
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(model_path)
    
    # Save summary
    metrics_logger.save_summary({
        'model_path': model_path,
        'noise_type': args.noise_type,
        'noise_std': args.noise_std,
        'learning_rate': args.learning_rate,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'tau': args.tau,
        'gamma': args.gamma
    })
    
    print(f'Model saved to {model_path}')
    print(f'Metrics logged to {metrics_logger.run_dir}')

if __name__ == '__main__':
    main()
