# data_sources/load_profiles.py
import pandas as pd
import numpy as np


def load_nrel_data():
    """
    Loads NREL dataset from the NREL AWS S3 Bucket using the NREL 'buildstock-query' library.
    for acessing the load profile data.
    """
    # TODO - use their shitty ahh library to get the data somehow...
    return None
    

def resample_to_hourly(df, time_col='timestamp', value_col='load'):   # change col default names based off what's actually in the dataset
    """
    Resample load profile time series data to average hourly load.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    hourly_df = df[[value_col]].resample('H').mean().reset_index()
    hourly_df.columns = ['hour', 'load']
    return hourly_df


def scale_household_loads(hourly_series, num_households=1):
    """
    Scale load profile by specified number of households.
    """
    return hourly_series * num_households


def apply_weekday_weekend_pattern(hourly_series, weekday_factor=1.0, weekend_factor=1.2):
    """
    Apply different scaling for weekdays and weekends.
    """
    df = hourly_series.to_frame('load')
    df['weekday'] = df.index.weekday
    df['factor'] = np.where(df['weekday'] < 5, weekday_factor, weekend_factor)
    df['load'] = df['load'] * df['factor']
    return df['load']


def add_random_noise(load_series, noise_std=0.05, seed=None):
    """
    (Optional) Add Gaussian noise to the load profile for synthetic variability.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_std, size=len(load_series))
    noisy_load = load_series + noise
    noisy_load = np.clip(noisy_load, 0, None)  # Ensure non-negative loads
    return noisy_load


def get_load_profile(url, num_households=1, seed=None, weekday_factor=1.0, weekend_factor=1.2, add_noise=False, noise_std=0.05):
    """
    Full pipeline for loading, resampling, scaling, and applying weekend/weekday patterns
    for load profile data, with optional random noise.
    """
    df = load_nrel_data(url, seed=seed)
    hourly_df = resample_to_hourly(df)
    scaled_df = scale_household_loads(hourly_df, num_households)
    patterned_df = apply_weekday_weekend_pattern(scaled_df, weekday_factor, weekend_factor)
    if add_noise:
        patterned_df = add_random_noise(patterned_df, noise_std=noise_std, seed=seed)
    return patterned_df