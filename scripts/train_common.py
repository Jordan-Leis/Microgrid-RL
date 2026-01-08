from __future__ import annotations
import numpy as np, yaml
from data_sources.nasa_power import fetch_power_hourly
from envs.microgrid_env import MicrogridEnv

def synthetic_load(hours: int, base_kwh: float, evening_mult: float, weekend_mult: float, noise_std: float):
    rng = np.random.default_rng(42)
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

def build_env(cfg_path: str, lat: float, lon: float, days: int):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    tz = cfg.get("location", {}).get("tz", "UTC") # if no timezone found, fall back to UTC
    parameters = cfg.get("location", {}).get("parameters", "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M")
    df_all = fetch_power_hourly(lat, lon, '2024-01-01', '2024-12-31', tz=tz, parameters=parameters) # data localized before being used to train agent
    end_hours = days * 24
    df = df_all.iloc[:end_hours].copy()

    # added new var to df - environment has access to new column
    if "ALLSKY_SFC_SW_DWN" in df.columns and "CLRSKY_SFC_SW_DWN" in df.columns:
        df["cloudiness_index"] = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]
        df["cloudiness_index"] = df["cloudiness_index"].clip(0, 1) # < 0: very cloudy, > 1: clear
        corr = df["cloudiness_index"].corr(df["ALLSKY_SFC_SW_DWN"]) # correlation between cloudiness index and actual sunlight
        cfg["analysis"] = {"cloudiness_corr": float(corr)}

    l = cfg['load']
    load = synthetic_load(end_hours, l['base_kwh_per_hour'], l['peak_multiplier_evening'], l['weekend_multiplier'], l['noise_std_pct'])
    return MicrogridEnv(cfg, df, load)
