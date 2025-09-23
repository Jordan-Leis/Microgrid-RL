from __future__ import annotations
import os, io, hashlib
import pandas as pd
import requests

CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(lat: float, lon: float, start: str, end: str, params: str) -> str:
    key = f"{lat:.4f}_{lon:.4f}_{start}_{end}_{params}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"nasa_{h}.csv")

def fetch_power_hourly(lat: float, lon: float, start: str, end: str,
                       parameters: str = "ALLSKY_SFC_SW_DWN,T2M") -> pd.DataFrame:
    cache = _cache_path(lat, lon, start, end, parameters)
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["datetime"])

    base = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    payload = {
        "parameters": parameters,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "CSV",
    }
    r = requests.get(base, params=payload, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), skiprows=10)  # skip header comments
    df["DATE"] = df["DATE"].astype(str)
    hour_col = "HOUR" if "HOUR" in df.columns else "Hour"
    df[hour_col] = df[hour_col].astype(int)
    df["datetime"] = pd.to_datetime(df["DATE"], format="%Y%m%d") + pd.to_timedelta(df[hour_col], unit="h")
    keep = ["datetime"] + [c for c in df.columns if c in parameters.split(",")]
    out = df[keep].copy()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_csv(cache, index=False)
    return out
