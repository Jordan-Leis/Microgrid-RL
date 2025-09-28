# data_sources/nasa_power.py
from __future__ import annotations

import os
import io
import hashlib
from typing import Optional, List

import pandas as pd
import requests


# Where we cache downloaded CSVs so runs are deterministic/offline after first fetch
CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(
    lat: float,
    lon: float,
    start: str,
    end: str,
    parameters: str,
    tz: Optional[str],
) -> str:
    """
    Build a deterministic cache filename keyed on query args.
    """
    key = f"{lat:.5f}_{lon:.5f}_{start}_{end}_{parameters}_{tz or 'none'}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"nasa_power_hourly_{h}.csv")


def _find_hour_col(columns: List[str]) -> str:
    """
    NASA POWER hourly CSVs can label the hour in different ways.
    Try a set of common options and return the first match.
    """
    for c in ("HOUR", "HR", "Hour", "hour", "hr"):
        if c in columns:
            return c
    # Some experimental dumps have "UTC" as 0..23
    if "UTC" in columns:
        return "UTC"
    raise KeyError(f"No hour column found. Got columns: {columns[:10]}...")


def _build_datetime(df: pd.DataFrame) -> pd.Series:
    """
    Construct a pandas datetime Series from the various date/hour encodings POWER uses.

    Supported schemas:
    - DATE (YYYYMMDD) + HOUR/HR
    - YEAR, MO, DY + HR/HOUR
    - YYYY, MM, DD + HR/HOUR
    """
    cols = set(df.columns)

    # DATE present: typically a string/int like 20240101
    if "DATE" in cols:
        date_str = df["DATE"].astype(str).str.slice(0, 8)

    # YEAR/MO/DY present
    elif {"YEAR", "MO", "DY"}.issubset(cols):
        y = df["YEAR"].astype(int).astype(str).str.zfill(4)
        m = df["MO"].astype(int).astype(str).str.zfill(2)
        d = df["DY"].astype(int).astype(str).str.zfill(2)
        date_str = y + m + d

    # Alternate field names
    elif {"YYYY", "MM", "DD"}.issubset(cols):
        y = df["YYYY"].astype(int).astype(str).str.zfill(4)
        m = df["MM"].astype(int).astype(str).str.zfill(2)
        d = df["DD"].astype(int).astype(str).str.zfill(2)
        date_str = y + m + d

    else:
        raise KeyError(f"Could not find date fields. Columns: {list(df.columns)[:12]}")

    hour_col = _find_hour_col(list(df.columns))
    hours = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)

    # Occasionally hours are 1..24; normalize to 0..23
    hours = hours % 24

    dt = pd.to_datetime(date_str, format="%Y%m%d") + pd.to_timedelta(hours, unit="h")
    return dt


def fetch_power_hourly(
    lat: float,
    lon: float,
    start: str,
    end: str,
    parameters: str = "ALLSKY_SFC_SW_DWN,T2M",
    tz: Optional[str] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch hourly NASA POWER data for a point (lat, lon) and return a tidy DataFrame:

        columns: ['datetime', <requested parameters that exist>]

    Notes
    -----
    - By default, timestamps are **UTC** and timezone-naive.
    - If `tz` is provided (e.g., 'Africa/Nairobi'), timestamps are converted to that
      zone and made naive (tz info dropped) so downstream code doesn't need timezone-aware dt.
    - CSV schema from POWER can vary; this function handles the common variants.
    - Results are cached under data/cache/ to support offline/fast re-runs.
    """
    cache = _cache_path(lat, lon, start, end, parameters, tz)
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["datetime"])

    base = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    payload = {
        "parameters": parameters,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        # POWER expects yyyymmdd (no dashes)
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "CSV",
        # Keep timestamps in UTC; we'll convert after parse if tz is requested.
        # "time-standard": "UTC",
    }

    # Issue request
    r = requests.get(base, params=payload, timeout=timeout)
    r.raise_for_status()

    # POWER CSV has many comment/header lines starting with '#'
    df = pd.read_csv(io.StringIO(r.text), comment="#")

    # Build a clean datetime from whatever fields we got
    dt = _build_datetime(df)

    # Keep only parameters that actually exist in this CSV
    requested = [p.strip() for p in parameters.split(",") if p.strip()]
    present = [c for c in requested if c in df.columns]
    if not present:
        raise ValueError(
            f"None of requested parameters {requested} found in CSV. "
            f"Columns: {df.columns.tolist()[:12]}"
        )

    out = pd.DataFrame({"datetime": dt})
    for c in present:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional timezone conversion (assumes input times are UTC)
    if tz:
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)

    # Sort & drop duplicates just in case
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # Cache and return
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_csv(cache, index=False)
    return out
