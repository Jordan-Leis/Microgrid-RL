#!/usr/bin/env python3
"""
Train a TQC (Truncated Quantile Critics) agent on the Microgrid-RL environment.

Environment setup is adapted from test_env.py but extended for RL training:
- Supports arbitrary days/hours
- Supports lat/lon (future extension)
- Supports seed for reproducibility
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TQC

from envs.microgrid_env import MicrogridEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--log-dir", type=str, default="runs/tqc")
    p.add_argument("--config", type=str, default="configs/tqc.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Create run directory
    run_dir = Path(args.log_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Environment factory
    # ---------------------------
    def make_env(lat, lon, days, seed):
        hours = days * 24
        # Simple synthetic solar and temperature profiles (can be replaced by real data)
        dt = pd.date_range("2024-01-01", periods=hours, freq="h")
        df = pd.DataFrame({
            "datetime": dt,
            "ALLSKY_SFC_SW_DWN": np.clip(np.sin((dt.hour - 6) / 12 * np.pi), 0, None),
            "T2M": 25.0,  # constant temperature for now
        })
        load = np.ones(hours) * cfg["load"]["base_kwh_per_hour"]

        env = MicrogridEnv(cfg, df, load)
        # Optional: set a seed if your environment supports it
        if hasattr(env, "seed"):
            env.seed(seed)
        return Monitor(env)

    # Create vectorized environments
    env = DummyVecEnv([lambda: make_env(args.lat, args.lon, args.days, args.seed)])
    eval_env = DummyVecEnv([lambda: make_env(args.lat, args.lon, args.days, args.seed + 10000)])

    # ---------------------------
    # Model initialization
    # ---------------------------
    model = TQC(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        n_quantiles=cfg["n_quantiles"],
        top_quantiles_to_drop_per_net=cfg["top_quantiles_to_drop_per_net"],
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
    )

    # ---------------------------
    # Evaluation callback
    # ---------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
    )

    # ---------------------------
    # Training
    # ---------------------------
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_cb,
    )

    # Save final checkpoint
    model.save(str(run_dir / "final_model.zip"))


if __name__ == "__main__":
    main()
