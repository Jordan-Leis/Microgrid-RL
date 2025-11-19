import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


import gym
import numpy as np


# Import Stable Baselines3 (assumed installed)
try:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3 import A2C, SAC
except Exception as e:
    raise ImportError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3[extra]"
    ) from e




def detect_success(info: Dict[str, Any]) -> int:
    """Return 1 if env signals success in `info` dict, else 0 or -1 if not present."""
    if info is None:
        return -1
    for key in ("is_success", "success", "done"): # common names
        if key in info:
            val = info[key]
            try:
                return int(bool(val))
            except Exception:
                return -1
    return -1




def evaluate_model(model_path: str, env_id: str, num_episodes: int, seed: int, out_dir: str, max_episode_steps: int = None):
    os.makedirs(out_dir, exist_ok=True)


    # Load model
    model = None
    try:
        model = BaseAlgorithm.load(model_path)
    except Exception:
        # Fallback: try specific classes
        for cls in (A2C, SAC):
            try:
                model = cls.load(model_path)
                break
            except Exception:
                continue
    if model is None:
        raise RuntimeError(f"Failed to load model from {model_path}")


    # Build env
    env = gym.make(env_id)
    if max_episode_steps is not None and hasattr(env, 'spec'):
        env._max_episode_steps = max_episode_steps


    env.seed(seed)
    np.random.seed(seed)


    results = []


    obs = env.reset()
    for ep in range(1, num_episodes + 1):
        done = False
        ep_reward = 0.0
        ep_len = 0
        infos = None


        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            infos = info
            if done:
                break


        mean_r = ep_reward / max(1, ep_len)
        success_flag = detect_success(infos)


        results.append({
        "episode": ep,
        "total_reward": ep_reward,
        "episode_length": ep_len,
        "mean_reward_per_step": mean_r,
        "success": success_flag,
        })
        
        # Write CSV
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem
    out_path = Path(out_dir) / f"{model_name}_eval_{ts}.csv"
    with out_path.open("w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)


    print(f"Wrote per-episode CSV to: {out_path}")
    return str(out_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help='Path to saved SB3 model (.zip)')
    parser.add_argument('--env-id', required=True, help='Gym env id, e.g. CartPole-v1')
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--max-episode-steps', type=int, default=None)
    args = parser.parse_args()


evaluate_model(
    model_path=args.model_path,
    env_id=args.env_id,
    num_episodes=args.num_episodes,
    seed=args.seed,
    out_dir=args.out_dir,
    max_episode_steps=args.max_episode_steps,
)