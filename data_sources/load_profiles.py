# data_sources/load_profiles.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union


def get_load_profiles(path=None):
    """
    Load household and commercial load profiles from CSV.
    If no path provided, uses default profiles in data/raw/load_profiles.csv
    Expected columns: hour, high_income_wh, medium_income_wh, low_income_wh, average_wh, commercial_wh
    """
    if path is None:
        path = Path(__file__).parent.parent / 'data' / 'raw' / 'load_profiles.csv'
    df = pd.read_csv(path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def scale_household_loads(hourly_series, num_households=1):
    """Scale load profile by specified number of households/buildings."""
    return hourly_series * num_households


def apply_weekday_weekend_pattern(load_series, is_weekend=False, weekday_factor=1.0, weekend_factor=1.2):
    """Apply different scaling for weekdays and weekends."""
    factor = weekend_factor if is_weekend else weekday_factor
    return load_series * factor


def add_random_noise(load_series, noise_std=0.05, seed=None, rng=None):
    """
    Add Gaussian noise to the load profile for synthetic variability.
    noise_std is in the same units as load_series (absolute).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=len(load_series))
    noisy_load = load_series + noise
    noisy_load = np.clip(noisy_load, 0, None)  # Ensure non-negative loads
    return noisy_load


def synthetic_daily_temp(day_of_year, t_mean=27.0, amp=3.0):
    """Simple sinusoidal annual temperature model (°C)."""
    phase_shift = 80
    return t_mean + amp * np.sin(2*np.pi*(day_of_year - phase_shift)/365)


def weather_scaler_hourly(T_day, kind="household"):
    """Cooling-driven hourly multiplier based on daily temperature."""
    T_0 = 25.0  # cooling baseline
    cdd = max(0.0, T_day - T_0)
    if kind == "household":
        hour_weights = np.array([
            0.02,0.02,0.02,0.02,0.02,0.03,0.04,0.05,
            0.05,0.05,0.05,0.06,0.05,0.05,0.05,0.06,
            0.07,0.08,0.09,0.09,0.08,0.06,0.04,0.03
        ])
        beta = 0.015   # ~1.5% per °C above 25°C, spread by hour
    else:  # commercial
        hour_weights = np.array([
            0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.07,
            0.10,0.10,0.10,0.10,0.10,0.09,0.08,0.06,
            0.05,0.03,0.01,0.01,0.01,0.01,0.01,0.01
        ])
        beta = 0.012
    return 1.0 + beta * cdd * hour_weights


def sample_day_with_weather(b, day_of_year, kind="household", rng=np.random.default_rng()):
    """
    Generate a single day's load (per-connection) from a 24h baseline b (kW),
    with lognormal day scaling, AR(1) shape noise, and weather hour multipliers.
    Returns (np.array length 24 in kW, info_dict).
    """
    # Day-to-day energy scalar
    alpha = rng.lognormal(mean=0.0, sigma=0.20)

    # Correlated hour noise
    rho, sigma = (0.85, 0.08) if kind == "household" else (0.85, 0.05)
    eps = np.zeros_like(b, dtype=float)
    for t in range(1, len(b)):
        eps[t] = rho*eps[t-1] + rng.normal(0, sigma)

    # Weather
    T_day = synthetic_daily_temp(day_of_year)
    s_hour = weather_scaler_hourly(T_day, kind=kind)  # length 24

    # Construct per-connection load
    L = alpha * b * (1 + eps) * s_hour
    L = np.clip(L, 0, None)
    return L, {"T_day": T_day, "alpha": alpha}


def get_daily_load(
    day_of_year: int,
    n_households: int = 1,
    n_commercial: int = 0,
    weekend: bool = False,
    path: Union[str, Path, None] = None,
    seed: Union[int, None] = None,
    weekday_factor_house: float = 1.0,
    weekend_factor_house: float = 1.05,
    weekday_factor_comm: float = 1.0,
    weekend_factor_comm: float = 0.90,
    add_measurement_noise: bool = True,
    noise_frac: float = 0.01,
) -> pd.DataFrame:
    """
    Build a realistic 24-hour load profile for a single day.

    Parameters
    ----------
    day_of_year : int
        1->365 (or 366) since Jan 1.
    n_households : int
        Number of households to scale the residential baseline by.
    n_commercial : int
        Number of commercial connections to scale the commercial baseline by.
    weekend : bool
        Whether this day is a weekend (affects scaling).
    path : str | Path | None
        Optional CSV path for load profiles. If None, uses default in data/raw.
    seed : int | None
        Seed for reproducibility.
    weekday_factor_house, weekend_factor_house : float
        Multipliers for residential loads on weekday/weekend.
    weekday_factor_comm, weekend_factor_comm : float
        Multipliers for commercial loads on weekday/weekend.
    add_measurement_noise : bool
        Adds small iid noise after synthesis to avoid identical repeats.
    noise_frac : float
        Noise standard deviation as a fraction of series mean (unitless).

    Returns
    -------
    pd.DataFrame with columns:
        hour, household_kw, commercial_kw, total_kw, temp_C, weekend
    """
    rng = np.random.default_rng(seed)
    df_profiles = get_load_profiles(path)

    # Baselines per connection (kW) from your CSV (Wh per hour -> kW)
    b_house = (df_profiles["average_wh"].to_numpy() / 1000.0).astype(float)
    b_comm  = (df_profiles["commercial_wh"].to_numpy() / 1000.0).astype(float)

    # Sample per-connection synthetic days with weather + correlated noise
    Lh_per, info_h = sample_day_with_weather(b_house, day_of_year, kind="household", rng=rng)
    Lc_per, info_c = sample_day_with_weather(b_comm,  day_of_year, kind="commercial", rng=rng)

    # Apply weekday/weekend patterns (multiplicative)
    Lh_per = apply_weekday_weekend_pattern(
        pd.Series(Lh_per), is_weekend=weekend,
        weekday_factor=weekday_factor_house, weekend_factor=weekend_factor_house
    ).to_numpy()

    Lc_per = apply_weekday_weekend_pattern(
        pd.Series(Lc_per), is_weekend=weekend,
        weekday_factor=weekday_factor_comm, weekend_factor=weekend_factor_comm
    ).to_numpy()

    # Scale by number of connections
    Lh = scale_household_loads(pd.Series(Lh_per), num_households=n_households).to_numpy()
    Lc = scale_household_loads(pd.Series(Lc_per), num_households=n_commercial).to_numpy()

    # Optional small iid "measurement" noise on top (post-aggregation)
    if add_measurement_noise:
        h_std = max(1e-6, noise_frac * max(1e-6, np.mean(Lh)))
        c_std = max(1e-6, noise_frac * max(1e-6, np.mean(Lc)))
        Lh = add_random_noise(pd.Series(Lh), noise_std=h_std, rng=rng).to_numpy()
        Lc = add_random_noise(pd.Series(Lc), noise_std=c_std, rng=rng).to_numpy()

    total = Lh + Lc
    hours = df_profiles["hour"].to_numpy() if "hour" in df_profiles.columns else np.arange(24)

    out = pd.DataFrame({
        "hour": hours.astype(int),
        "household_kw": Lh,
        "commercial_kw": Lc,
        "total_kw": total,
        "temp_C": info_h["T_day"],  # same T used for both streams
        "weekend": bool(weekend),
    })
    return out
