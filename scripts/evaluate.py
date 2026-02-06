#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
from stable_baselines3 import SAC, A2C, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from scripts.train_common import build_env


def parse_args():
   p = argparse.ArgumentParser()
   p.add_argument("--model", type=str, required=True)
   p.add_argument("--algo", type=str, choices=["sac", "a2c", "ddpg"], required=True)
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
   else:
       raise ValueError(f"Unsupported algo {args.algo}")


   rows = []
   for ep in range(args.episodes):
       obs, info = env.reset(seed=args.seed + ep) # yes?
       done = False


       ep_return = 0.0
       ep_unmet = 0.0
       ep_litres = 0.0
       ep_len = 0


       while not done:
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

