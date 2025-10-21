# used for testing load profile functions
# python -m pytest tests/test_load_profiles.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_sources.load_profiles import (
    resample_to_hourly,
    scale_household_loads,
    apply_weekday_weekend_pattern,
    add_random_noise
)

@pytest.fixture
def sample_load_df():
    """
    Create sample 15-min load data for testing
    """

    dates = pd.date_range(
        start='2024-01-01',
        periods=96,  # 24 hours * 4 (15-min intervals)
        freq='15T'
    )
    loads = np.ones(len(dates)) * 0.5  # 0.5 kWh base load
    return pd.DataFrame({'timestamp': dates, 'load': loads})

def test_resample_to_hourly(sample_load_df):
    """
    Test resampling 15-min data to hourly.
    """
    hourly = resample_to_hourly(sample_load_df)
    assert len(hourly) == 24  # One day of hourly data
    assert all(hourly['load'] == 0.5)  # Should preserve constant load
    assert list(hourly.columns) == ['hour', 'load']

def test_scale_household_loads():
    """
    Test scaling loads by number of households.
    """
    loads = pd.Series([1.0, 2.0, 3.0])
    scaled = scale_household_loads(loads, num_households=3)
    assert all(scaled == [3.0, 6.0, 9.0])

def test_weekday_weekend_pattern():
    """
    Test weekday/weekend pattern application.
    """
    # Create a week of hourly data
    dates = pd.date_range('2024-01-01', '2024-01-07', freq='H')
    loads = pd.Series(np.ones(len(dates)), index=dates)
    
    result = apply_weekday_weekend_pattern(loads, weekday_factor=1.0, weekend_factor=2.0)
    
    # Check weekday values (Mon-Fri)
    assert all(result[dates.weekday < 5] == 1.0)
    # Check weekend values (Sat-Sun)
    assert all(result[dates.weekday >= 5] == 2.0)

def test_random_noise():
    """
    Test random noise addition.
    """
    loads = pd.Series([1.0] * 100)
    
    # Test reproducibility with seed
    noisy1 = add_random_noise(loads, noise_std=0.1, seed=42)
    noisy2 = add_random_noise(loads, noise_std=0.1, seed=42)
    assert all(noisy1 == noisy2)
    
    # Test non-negative constraint
    assert all(noisy1 >= 0)
    
    # Test noise scale
    assert abs(noisy1.mean() - 1.0) < 0.1  # Mean should be close to original