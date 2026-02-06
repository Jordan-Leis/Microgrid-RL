#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
from stable_baselines3 import SAC, A2C, DDPG, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from scripts.train_common import build_env

# Optional sb3-contrib for RecurrentPPO and TQC
try:
    from sb3_contrib import RecurrentPPO, TQC
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    RecurrentPPO = None
    TQC = None
    SB3_CONTRIB_AVAILABLE = False


def parse_args():
   p = argparse.ArgumentParser()
   p.add_argument("--model", type=str, required=True)
   p.add_argument("--algo", type=str, choices=["sac", "a2c", "ddpg", "ppo", "rppo", "tqc"], required=True)
   p.add_argument("--lat", type=float, required=True)
   p.add_argument("--lon", type=float, required=True)
   p.add_argument("--days", type=int, default=60) # default 180 in stub?
   p.add_argument("--cfg", type=str, default="configs/default.yaml")
   p.add_argument("--episodes", type=int, default=50)
   p.add_argument("--seed", type=int, default=0)
   p.add_argument("--out-dir", type=str, required=True)
   p.add_argument("--deterministic", action="store_true", default=True) # or hardcode True...yes prob don't want stochastic?
   return p.parse_args()


# compute summary
def compute_summary(df: pd.DataFrame):
   return {c: {"mean": float(np.nanmean(df[c])), "std": float(np.nanstd(df[c]))}
           for c in df.columns if c != "episode"}

def main():
   args = parse_args()
   out_dir = Path(args.out_dir)
   out_dir.mkdir(parents=True, exist_ok=True)
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)


   env = build_env(args.cfg, args.lat, args.lon, args.days)


   if args.algo == "sac":
       model: BaseAlgorithm = SAC.load(args.model, env=env)
   elif args.algo == "a2c":
       model: BaseAlgorithm = A2C.load(args.model, env=env)
   elif args.algo == "ddpg":
       model: BaseAlgorithm = DDPG.load(args.model, env=env)
   elif args.algo == "ppo":
       model: BaseAlgorithm = PPO.load(args.model, env=env)
   elif args.algo == "rppo":
       if not SB3_CONTRIB_AVAILABLE:
           raise ImportError("sb3-contrib required for rppo. Install with: pip install sb3-contrib>=2.3.0")
       model: BaseAlgorithm = RecurrentPPO.load(args.model, env=env)
   elif args.algo == "tqc":
       if not SB3_CONTRIB_AVAILABLE:
           raise ImportError("sb3-contrib required for tqc. Install with: pip install sb3-contrib>=2.3.0")
       model: BaseAlgorithm = TQC.load(args.model, env=env)
   else:
       raise ValueError(f"Unsupported algo {args.algo}")

   # Track if model is recurrent (needs state handling)
   is_recurrent = args.algo == "rppo"


   rows = []
   for ep in range(args.episodes):
       obs, info = env.reset(seed=args.seed + ep)
       done = False
       # For recurrent models, reset hidden state at episode start
       state = None
       # RecurrentPPO needs episode_start flag
       episode_start = True

       ep_return = 0.0
       ep_unmet = 0.0
       ep_litres = 0.0
       ep_len = 0

       while not done:
           if is_recurrent:
               action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=args.deterministic)
               episode_start = False
           else:
               action, _ = model.predict(obs, deterministic=args.deterministic)
           obs, reward, terminated, truncated, info = env.step(action)
           done = bool(terminated or truncated)

           ep_return += float(reward)
           ep_unmet += info.get("unmet_kwh", 0.0)
           ep_litres += info.get("litres_used", 0.0)
           ep_len += 1

       rows.append({"episode": ep,
                    "return": ep_return,
                    "unmet_kwh": ep_unmet,
                    "litres_used": ep_litres,
                    "length": ep_len
                    })
      
   # save to CSV
   df = pd.DataFrame(rows)
   df.to_csv(out_dir / "episodes.csv", index=False)


   # save summary JSON
   summary = compute_summary(df)
   (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))


   print(f"Saved evaluation results to {out_dir}")

if __name__ == "__main__":
   main()

