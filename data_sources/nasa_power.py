# data_sources/nasa_power.py
from __future__ import annotations

import os
import io
import hashlib
from typing import Optional, List

import pandas as pd
import requests


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
    key = f"{lat:.5f}_{lon:.5f}_{start}_{end}_{parameters}_{tz or 'none'}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"nasa_power_hourly_{h}.csv")


def _find_header_index(lines: List[str]) -> int:
    """
    POWER CSVs include a preamble with comments and sometimes plain-text lines
    not prefixed by '#'. Find the first line that looks like a CSV header.
    """
    for i, line in enumerate(lines):
        if "," not in line:
            continue
        u = line.upper()
        has_date = "DATE" in u
        has_ymd = ("YEAR" in u or "YYYY" in u) and ("MO" in u or "MM" in u) and ("DY" in u or "DD" in u)
        has_hour = any(tok in u for tok in ["HOUR", "HR"])
        if (has_date and has_hour) or (has_ymd and has_hour):
            return i
    # As a last resort, return the first comma line
    for i, line in enumerate(lines):
        if "," in line:
            return i
    raise ValueError("Could not find a CSV header line in NASA POWER response.")


def _find_hour_col(columns: List[str]) -> str:
    for c in ("HOUR", "HR", "Hour", "hour", "hr", "UTC"):
        if c in columns:
            return c
    raise KeyError(f"No hour column found. Got columns: {columns[:10]}...")


def _build_datetime(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)

    if "DATE" in cols:
        date_str = df["DATE"].astype(str).str.slice(0, 8)
    elif {"YEAR", "MO", "DY"}.issubset(cols):
        y = df["YEAR"].astype(int).astype(str).str.zfill(4)
        m = df["MO"].astype(int).astype(str).str.zfill(2)
        d = df["DY"].astype(int).astype(str).str.zfill(2)
        date_str = y + m + d
    elif {"YYYY", "MM", "DD"}.issubset(cols):
        y = df["YYYY"].astype(int).astype(str).str.zfill(4)
        m = df["MM"].astype(int).astype(str).str.zfill(2)
        d = df["DD"].astype(int).astype(str).str.zfill(2)
        date_str = y + m + d
    else:
        raise KeyError(f"Could not find date fields. Columns: {list(df.columns)[:12]}")

    hour_col = _find_hour_col(list(df.columns))
    hours = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
    hours = hours % 24  # normalize 1..24 → 0..23
    return pd.to_datetime(date_str, format="%Y%m%d") + pd.to_timedelta(hours, unit="h")


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
    Fetch hourly NASA POWER data and return a tidy DataFrame with:
        columns: ['datetime', <requested parameters that exist>]

    Notes
    -----
    - Timestamps are UTC (naive) unless `tz` is provided (e.g., 'Africa/Nairobi'),
      in which case they're converted and tz info dropped.
    - Handles multiple CSV schemas and weird preambles.
    - Caches results under data/cache/.
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
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "CSV",
    }
    r = requests.get(base, params=payload, timeout=timeout)
    r.raise_for_status()

    text = r.text.replace("\r\n", "\n")
    lines = text.splitlines()
    hdr_idx = _find_header_index(lines)
    clean_csv = "\n".join(lines[hdr_idx:])

    # Read the cleaned CSV
    df = pd.read_csv(io.StringIO(clean_csv))
    # Trim whitespace in column names for safety
    df.columns = [c.strip() for c in df.columns]

    # Build a proper datetime index
    dt = _build_datetime(df)

    # Keep only requested params that actually exist in this file
    requested = [p.strip() for p in parameters.split(",") if p.strip()]
    present = [c for c in requested if c in df.columns]
    if not present:
        raise ValueError(
            f"None of requested parameters {requested} found. "
            f"Columns: {df.columns.tolist()[:12]}"
        )

    out = pd.DataFrame({"datetime": dt})
    for c in present:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional timezone conversion (assumes NASA times are UTC)
    if tz:
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)

    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_csv(cache, index=False)
    return out
