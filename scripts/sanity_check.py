import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from scripts.train_common import build_env


@dataclass
class EpisodeStats:
	reward_sum: float
	unmet_kwh_sum: float
	liters_sum: float


def evaluate_episode(model, cfg_path: str, lat: float, lon: float, days: int, seed: int, save_trajectory_path: str | None = None, progress_sec: float | None = None) -> EpisodeStats:
	"""
	Runs a single evaluation episode with the trained model on a fresh env.
	Optionally saves a per-step trajectory CSV.
	"""
	env = build_env(cfg_path, lat, lon, days)
	obs, _ = env.reset(seed=seed)

	r_total = 0.0
	unmet_total = 0.0
	liters_total = 0.0

	trajectory_rows: List[List[float]] = []
	header = [
		"t",
		"a_batt",
		"a_diesel",
		"r_step",
		"unmet_kwh",
		"liters_used",
		"soc",
		"fuel",
		"e_solar",
		"e_load",
		"e_batt",
		"e_diesel",
	]

	# Gymnasium returns (terminated, truncated). Stop when either is True.
	terminated = False
	truncated = False
	t = 0
	last_beat = time.time()
	while not (terminated or truncated):
		action, _ = model.predict(obs, deterministic=True)
		obs, r, terminated, truncated, info = env.step(action)
		r_total += float(r)
		unmet_total += float(info.get("unmet_kwh", 0.0))
		liters_total += float(info.get("liters_used", 0.0))

		# Optional evaluation heartbeat
		if progress_sec is not None:
			now = time.time()
			if now - last_beat >= progress_sec:
				print(f"[EVAL] t={t}  reward_sum={r_total:.2f}  unmet_kWh={unmet_total:.2f}  diesel_L={liters_total:.2f}")
				last_beat = now

		if save_trajectory_path is not None:
			row = [
				t,
				float(action[0]),
				float(action[1]),
				float(r),
				float(info.get("unmet_kwh", 0.0)),
				float(info.get("liters_used", 0.0)),
				float(info.get("soc", np.nan)),
				float(info.get("fuel", np.nan)),
				float(info.get("e_solar", np.nan)),
				float(info.get("e_load", np.nan)),
				float(info.get("e_batt", np.nan)),
				float(info.get("e_diesel", np.nan)),
			]
			trajectory_rows.append(row)
		t += 1

	if save_trajectory_path is not None:
		os.makedirs(os.path.dirname(save_trajectory_path), exist_ok=True)
		with open(save_trajectory_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(header)
			writer.writerows(trajectory_rows)

	return EpisodeStats(reward_sum=r_total, unmet_kwh_sum=unmet_total, liters_sum=liters_total)


class ProgressCallback(BaseCallback):
	"""
	Simple console progress: prints percent, speed (steps/s), elapsed, and ETA every N seconds.
	"""
	def __init__(self, total_timesteps: int, label: str, print_every_sec: float = 30.0):
		super().__init__()
		self.total_timesteps = int(total_timesteps)
		self.label = label
		self.print_every_sec = float(print_every_sec)
		self._t0 = None
		self._last_print = None
		self._t0_steps = 0

	def _on_training_start(self) -> None:
		self._t0 = time.time()
		self._last_print = self._t0
		self._t0_steps = int(self.num_timesteps)
		print(f"[{self.label}] Training started...")

	def _on_step(self) -> bool:
		now = time.time()
		if now - self._last_print >= self.print_every_sec:
			done = min(int(self.num_timesteps), self.total_timesteps)
			elapsed = now - self._t0
			delta_steps = max(1, done - self._t0_steps)
			steps_per_sec = delta_steps / max(1e-6, elapsed)
			pct = 100.0 * done / max(1, self.total_timesteps)
			remain_steps = max(0, self.total_timesteps - done)
			eta_sec = remain_steps / max(1e-6, steps_per_sec)
			def fmt(sec: float) -> str:
				m = int(sec // 60); s = int(sec % 60)
				h = m // 60; m = m % 60
				return f"{h:02d}:{m:02d}:{s:02d}"
			print(f"[{self.label}] {pct:6.2f}%  {done}/{self.total_timesteps} steps  "
			      f"{steps_per_sec:6.2f} steps/s  elapsed {fmt(elapsed)}  ETA {fmt(eta_sec)}")
			self._last_print = now
		return True

	def _on_training_end(self) -> None:
		total_elapsed = time.time() - self._t0 if self._t0 is not None else 0.0
		print(f"[{self.label}] Training finished in {total_elapsed:.1f}s")


def train_once(algo: str, seed: int, cfg_path: str, lat: float, lon: float, days: int, total_timesteps: int):
	"""
	Trains either A2C or SAC for a small budget on a single DummyVecEnv.
	Returns the trained model.
	"""
	def make_env():
		env = build_env(cfg_path, lat, lon, days)
		return Monitor(env)

	env = DummyVecEnv([make_env])
	if algo == "a2c":
		model = A2C("MlpPolicy", env, verbose=0, seed=seed)
	elif algo == "sac":
		model = SAC("MlpPolicy", env, verbose=0, seed=seed)
	else:
		raise ValueError(f"Unknown algo: {algo}")

	cb = ProgressCallback(total_timesteps=total_timesteps, label=f"{algo.upper()} seed={seed}")
	model.learn(total_timesteps=total_timesteps, callback=cb)
	return model


def summarize(results: Dict[str, List[EpisodeStats]]) -> Dict[str, Tuple[float, float, float]]:
	"""
	Aggregates per-algorithm episode stats across seeds.
	Returns dict algo -> (mean_reward, mean_unmet_kwh, mean_liters)
	"""
	summary: Dict[str, Tuple[float, float, float]] = {}
	for algo, stats_list in results.items():
		if not stats_list:
			summary[algo] = (float("nan"), float("nan"), float("nan"))
			continue
		mean_reward = float(np.mean([s.reward_sum for s in stats_list]))
		mean_unmet = float(np.mean([s.unmet_kwh_sum for s in stats_list]))
		mean_liters = float(np.mean([s.liters_sum for s in stats_list]))
		summary[algo] = (mean_reward, mean_unmet, mean_liters)
	return summary


def emit_sanity_warnings(episode: EpisodeStats, label: str):
	"""
	Prints simple sanity warnings for obviously broken behavior.
	We can add more checks as needed.
	"""
	if not np.isfinite(episode.reward_sum):
		print(f"[WARN] {label}: non-finite episode reward")
	if episode.liters_sum < -1e-6:
		print(f"[WARN] {label}: negative diesel consumption")
	if episode.unmet_kwh_sum < -1e-6:
		print(f"[WARN] {label}: negative unmet demand")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--cfg", type=str, default="configs/default.yaml")
	ap.add_argument("--lat", type=float, default=-1.2921)
	ap.add_argument("--lon", type=float, default=36.8219)
	ap.add_argument("--days", type=int, default=60)
	ap.add_argument("--steps", type=int, default=20_000)
	ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
	ap.add_argument("--out_dir", type=str, default="outputs/sanity")
	ap.add_argument("--progress_sec", type=float, default=30.0, help="Seconds between progress prints during training")
	args = ap.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)

	algos = ["a2c", "sac"]
	all_results: Dict[str, List[EpisodeStats]] = {a: [] for a in algos}

	for algo in algos:
		for i, seed in enumerate(args.seeds):
			print(f"[{algo.upper()}] Training seed={seed} for {args.steps} steps...")
			# Progress frequency is configured within the callback; ensure it uses the CLI arg
			model = train_once(
				algo=algo,
				seed=seed,
				cfg_path=args.cfg,
				lat=args.lat,
				lon=args.lon,
				days=args.days,
				total_timesteps=args.steps,
			)

			# Evaluate one episode and optionally save a trajectory for the first seed per algo
			trajectory_path = None
			if i == 0:
				trajectory_path = os.path.join(args.out_dir, f"{algo}_seed{seed}_trajectory.csv")
			print(f"[{algo.upper()} seed={seed}] Evaluating episode...")
			_eval_t0 = time.time()
			stats = evaluate_episode(
				model=model,
				cfg_path=args.cfg,
				lat=args.lat,
				lon=args.lon,
				days=args.days,
				seed=seed,
				save_trajectory_path=trajectory_path,
				progress_sec=args.progress_sec,
			)
			print(f"[{algo.upper()} seed={seed}] Evaluation finished in {time.time()-_eval_t0:.1f}s")
			all_results[algo].append(stats)
			emit_sanity_warnings(stats, f"{algo.upper()} seed {seed}")

	# Aggregate and print short summary
	summary = summarize(all_results)
	print("\n=== Sanity Check Summary (3 seeds × 1 scenario × 20k steps) ===")
	for algo in algos:
		mr, mu, ml = summary[algo]
		print(f"{algo.upper():>4} → mean_reward={mr:.2f}  unmet_kWh={mu:.2f}  diesel_L={ml:.2f}")

	# Save summary CSV
	summary_csv = os.path.join(args.out_dir, "summary.csv")
	with open(summary_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["algo", "mean_reward", "mean_unmet_kWh", "mean_diesel_L"])
		for algo in algos:
			mr, mu, ml = summary[algo]
			writer.writerow([algo, f"{mr:.6f}", f"{mu:.6f}", f"{ml:.6f}"])
	print(f"\nWrote summary → {summary_csv}")


if __name__ == "__main__":
	main()


