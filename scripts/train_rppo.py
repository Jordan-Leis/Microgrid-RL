import argparse
import os
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.train_common import build_env
from scripts.metrics_logger import MetricsLogger, make_run_id
from scripts.callbacks import StructuredMetricsCallback

# Lazy imports for algorithms to allow optional sb3-contrib
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except Exception:
    RecurrentPPO = None
    SB3_CONTRIB_AVAILABLE = False


def main():
    ap = argparse.ArgumentParser()
    # Scenario/environment
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--days', type=int, default=180)
    ap.add_argument('--cfg', type=str, default='configs/rppo.yaml')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--total_timesteps', type=int, default=500_000)
    ap.add_argument('--scenario', type=str, default=None, help='Scenario name (default: auto-generated)')
    ap.add_argument('--log_freq', type=int, default=1000, help='Log metrics every N timesteps')
    ap.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    ap.add_argument('--wandb-project', type=str, default='microgrid-rl', help='W&B project name')
    ap.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username/team)')

    # Model hyperparameters / architecture
    ap.add_argument('--ff', action='store_true', help='Use feedforward PPO baseline instead of recurrent')
    ap.add_argument('--rnn-type', type=str, choices=['lstm', 'gru'], default='lstm', help='Recurrent cell type')
    ap.add_argument('--hidden-size', type=int, default=128, help='RNN hidden size')
    ap.add_argument('--n-layers', type=int, default=1, help='Number of RNN layers')
    ap.add_argument('--n-steps', type=int, default=256, help='Rollout steps per env')
    ap.add_argument('--batch-size', type=int, default=64, help='Batch size for optimization')
    ap.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    ap.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    ap.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    ap.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    ap.add_argument('--ent-coef', type=float, default=0.0, help='Entropy coefficient')
    ap.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    ap.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    args = ap.parse_args()

    # Load config for metrics/env parameters
    cfg = yaml.safe_load(open(args.cfg, 'r'))

    # Create scenario name if not provided
    scenario = args.scenario or f"lat{args.lat:.2f}_lon{args.lon:.2f}_{args.days}d"
    scenario = scenario.replace('.', '_').replace('-', 'm')  # sanitize filename

    algo_name = 'ppo' if args.ff else 'rppo'
    run_id = make_run_id(algo_name, scenario, args.seed)
    metrics_logger = MetricsLogger(
        run_id=run_id,
        algorithm=algo_name,
        scenario=scenario,
        seed=args.seed,
        config_path=args.cfg
    )

    def make_env():
        env = build_env(args.cfg, args.lat, args.lon, args.days)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # Build policy kwargs
    policy_kwargs = {}
    if not args.ff:
        # Recurrent policy kwargs differ by cell type; fall back to defaults if unknown
        if args.rnn_type == 'lstm':
            policy_kwargs.update({
                'lstm_hidden_size': args.hidden_size,
                'n_lstm_layers': args.n_layers,
                'shared_lstm': True,
            })
            policy = 'MlpLstmPolicy'
        else:
            policy_kwargs.update({
                'gru_hidden_size': args.hidden_size,
                'n_gru_layers': args.n_layers,
                'shared_gru': True,
            })
            policy = 'MlpGruPolicy'
    else:
        policy = 'MlpPolicy'

    # Instantiate model
    if args.ff:
        model = PPO(
            policy,
            env,
            seed=args.seed,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        if not SB3_CONTRIB_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for Recurrent PPO. Install with: pip install sb3-contrib>=2.3.0"
            )
        model = RecurrentPPO(
            policy,
            env,
            seed=args.seed,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
        )

    # Callback for structured logging (no eval_env to keep it generic and RNN-safe)
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
        wandb_tags=[algo_name, scenario],
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

