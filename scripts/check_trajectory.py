import argparse
import sys
import numpy as np
import pandas as pd
import yaml


def load_cfg(cfg_path: str) -> dict:
	with open(cfg_path, 'r') as f:
		return yaml.safe_load(f)


def pct(frac: float) -> float:
	return 100.0 * float(frac)


def is_finite_series(s: pd.Series) -> bool:
	return np.isfinite(s.to_numpy(dtype=float)).all()


def analyze(csv_path: str, cfg_path: str) -> int:
	cfg = load_cfg(cfg_path)
	soc_min = float(cfg['battery']['soc_min_pct'])
	soc_max = float(cfg['battery']['soc_max_pct'])

	df = pd.read_csv(csv_path)

	required_cols = [
		't','a_batt','a_diesel','r_step','unmet_kwh','liters_used',
		'soc','fuel','e_solar','e_load','e_batt','e_diesel'
	]
	for c in required_cols:
		if c not in df.columns:
			print(f"[ERROR] Missing column in CSV: {c}")
			return 2

	# Finite checks
	has_nans = df.isna().any().any()
	finite_ok = all(is_finite_series(df[c]) for c in required_cols)

	steps = len(df)
	reward_sum = float(df['r_step'].sum())
	unmet_total = float(df['unmet_kwh'].sum())
	liters_total = float(df['liters_used'].sum())

	soc_min_obs = float(df['soc'].min())
	soc_max_obs = float(df['soc'].max())
	soc_viol_lo = float((df['soc'] < soc_min - 1e-6).mean())
	soc_viol_hi = float((df['soc'] > soc_max + 1e-6).mean())

	unmet_frac = float((df['unmet_kwh'] > 0).mean())
	solar_frac = float((df['e_solar'] > 0).mean())

	neg_liters = bool((df['liters_used'] < 0).any())
	neg_diesel_e = bool((df['e_diesel'] < 0).any())
	neg_solar = bool((df['e_solar'] < 0).any())

	# Simple correlations for plausibility
	mean_batt_when_solar = float(df.loc[df['e_solar'] > 0, 'e_batt'].mean()) if (df['e_solar'] > 0).any() else float('nan')
	mean_batt_when_dark = float(df.loc[df['e_solar'] <= 1e-9, 'e_batt'].mean()) if (df['e_solar'] <= 1e-9).any() else float('nan')

	print("=== Trajectory Sanity Report ===")
	print(f"File: {csv_path}")
	print(f"Steps: {steps}")
	print(f"Reward sum: {reward_sum:.2f}")
	print(f"Unmet total (kWh): {unmet_total:.2f}  | hours unmet>0: {pct(unmet_frac):.1f}%")
	print(f"Diesel total (L): {liters_total:.2f}")
	print(f"SOC min/max: {soc_min_obs:.3f} / {soc_max_obs:.3f}  (expected within [{soc_min:.2f}, {soc_max:.2f}])")
	print(f"SOC violations: below_min {pct(soc_viol_lo):.2f}% | above_max {pct(soc_viol_hi):.2f}%")
	print(f"Solar>0 steps: {pct(solar_frac):.1f}%")
	print(f"Any NaNs: {has_nans} | All finite: {finite_ok}")
	print(f"Any negative liters_used: {neg_liters} | Any negative e_diesel: {neg_diesel_e} | Any negative e_solar: {neg_solar}")
	print(f"Mean e_batt when solar>0 (charge expected, negative): {mean_batt_when_solar:.3f}")
	print(f"Mean e_batt when solar≈0 (discharge expected, positive): {mean_batt_when_dark:.3f}")

	# Simple pass/fail summary
	ok = True
	if has_nans or not finite_ok:
		print("[FAIL] Non-finite values detected.")
		ok = False
	if neg_liters or neg_diesel_e or neg_solar:
		print("[FAIL] Negative physical quantities detected.")
		ok = False
	if soc_viol_lo > 0.01 or soc_viol_hi > 0.01:
		print("[WARN] >1% of steps violate SOC bounds.")
	# Unmet being nonzero is not a failure by itself; it's penalized but can be present.

	if ok:
		print("[PASS] Basic sanity checks OK.")
	else:
		print("[NOTE] Some checks failed; review details above.")

	return 0 if ok else 1


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--csv', type=str, required=True, help='Path to trajectory CSV (from sanity_check.py)')
	ap.add_argument('--cfg', type=str, default='configs/default.yaml', help='Config to read SOC bounds')
	args = ap.parse_args()
	sys.exit(analyze(args.csv, args.cfg))


if __name__ == '__main__':
	main()


