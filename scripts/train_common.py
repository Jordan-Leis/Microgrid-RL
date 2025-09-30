from __future__ import annotations
import numpy as np, yaml
from data_sources.nasa_power import fetch_power_hourly
from envs.microgrid_env import MicrogridEnv

def synthetic_load(hours: int, base_kwh: float, evening_mult: float, weekend_mult: float, noise_std: float, seed: int = None):
    rng = np.random.default_rng(seed)
    load = np.zeros(hours, dtype=float)
    for t in range(hours):
        hour = t % 24
        val = base_kwh
        if 18 <= hour <= 22: val *= evening_mult
        day = (t // 24) % 7
        if day in (5, 6): val *= weekend_mult
        val *= max(0.0, rng.normal(1.0, noise_std))
        load[t] = val
    return load

def build_env(cfg_path: str, lat: float, lon: float, days: int, year: int = 2024, load_scenario: str = 'base', seed: int = None):
    cfg = yaml.safe_load(open(cfg_path, 'r'))

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    df_all = fetch_power_hourly(lat, lon, start, end)

    end_hours = days * 24
    df = df_all.iloc[:end_hours].copy()

    l = cfg['load']
    if load_scenario == 'high':
        base = l['base_kwh_per_hour'] * l['peak_multiplier_evening']
    elif load_scenario == 'low':
        base = l['base_kwh_per_hour'] * l['weekend_multiplier']
    else:  # 'base'
        base = l['base_kwh_per_hour']

    load = synthetic_load(
        end_hours,
        base,
        l['peak_multiplier_evening'],
        l['weekend_multiplier'],
        l['noise_std_pct'],
        seed
    )

    return MicrogridEnv(cfg, df, load)

