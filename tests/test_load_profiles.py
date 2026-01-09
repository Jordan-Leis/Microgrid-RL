# used for testing load profile functions
# python -m pytest tests/test_load_profiles.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from data_sources.load_profiles import (
    get_load_profiles,
    scale_household_loads,
    apply_weekday_weekend_pattern,
    add_random_noise,
    synthetic_daily_temp,
    weather_scaler_hourly,
    sample_day_with_weather,
    get_daily_load
)

@pytest.fixture
def sample_load_profiles():
    """Create sample load profile data for testing"""
    data = {
        'hour': range(24),
        'high_income_wh': [10.0] * 24,
        'medium_income_wh': [5.0] * 24,
        'low_income_wh': [2.0] * 24,
        'average_wh': [5.0] * 24,
        'commercial_wh': [100.0] * 24
    }
    df = pd.DataFrame(data)
    # Ensure column names are clean
    df.columns = df.columns.str.strip()
    return df

def test_get_load_profiles(tmp_path, sample_load_profiles):
    """Test loading profiles from CSV"""
    # Save sample data to temp CSV
    test_csv = tmp_path / "test_profiles.csv"
    sample_load_profiles.to_csv(test_csv, index=False)
    
    # Test loading from specific path
    df = get_load_profiles(test_csv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['hour', 'high_income_wh', 'medium_income_wh', 
                               'low_income_wh', 'average_wh', 'commercial_wh']
    assert len(df) == 24

def test_scale_household_loads():
    """Test load scaling by number of households"""
    series = pd.Series([1.0, 2.0, 3.0])
    
    # Test single household (no scaling)
    result = scale_household_loads(series, num_households=1)
    assert all(result == series)
    
    # Test multiple households
    result = scale_household_loads(series, num_households=3)
    assert all(result == [3.0, 6.0, 9.0])

def test_apply_weekday_weekend_pattern():
    """Test weekday/weekend pattern application"""
    series = pd.Series([1.0, 2.0, 3.0])
    
    # Test weekday
    result = apply_weekday_weekend_pattern(series, is_weekend=False, 
                                         weekday_factor=1.0, weekend_factor=1.2)
    assert all(result == series)
    
    # Test weekend
    result = apply_weekday_weekend_pattern(series, is_weekend=True, 
                                         weekday_factor=1.0, weekend_factor=1.2)
    assert all(result == series * 1.2)

def test_add_random_noise():
    """Test random noise addition"""
    series = pd.Series([1.0] * 100)
    noise_std = 0.1
    seed = 42
    
    # Test reproducibility with seed
    result1 = add_random_noise(series, noise_std=noise_std, seed=seed)
    result2 = add_random_noise(series, noise_std=noise_std, seed=seed)
    assert all(result1 == result2)
    
    # Test non-negative constraint
    assert all(result1 >= 0)
    
    # Test noise scale
    assert abs(result1.mean() - 1.0) < 3 * noise_std  # Within 3 sigma

def test_synthetic_daily_temp():
    """Test synthetic temperature generation"""
    # Test summer day (day 200)
    summer_temp = synthetic_daily_temp(200, t_mean=27.0, amp=3.0)
    assert isinstance(summer_temp, float)
    
    # Test winter day (day 20)
    winter_temp = synthetic_daily_temp(20, t_mean=27.0, amp=3.0)
    assert isinstance(winter_temp, float)
    
    # Test full year pattern
    temps = [synthetic_daily_temp(d) for d in range(1, 366)]
    assert len(temps) == 365
    assert min(temps) < 27.0  # Should have temps below mean
    assert max(temps) > 27.0  # Should have temps above mean

def test_weather_scaler_hourly():
    """Test weather-based load scaling"""
    # Test household pattern
    h_scale = weather_scaler_hourly(30.0, kind="household")  # 5°C above baseline
    assert len(h_scale) == 24
    assert all(h_scale >= 1.0)  # All hours should scale up
    
    # Test commercial pattern
    c_scale = weather_scaler_hourly(30.0, kind="commercial")
    assert len(c_scale) == 24
    assert all(c_scale >= 1.0)
    
    # Test below baseline temperature
    cool_scale = weather_scaler_hourly(20.0, kind="household")
    assert all(cool_scale == 1.0)  # No scaling below baseline

def test_get_daily_load(tmp_path, sample_load_profiles):
    """Test daily load profile generation"""

    # Create the CSV that get_daily_load expects
    test_csv = tmp_path / "load_profiles.csv"
    sample_load_profiles.to_csv(test_csv, index=False)

    # Test basic weekday profile
    df = get_daily_load(
        day_of_year=180,  # Summer day
        n_households=10,
        n_commercial=1,
        weekend=False,
        seed=42,
        path=test_csv,  # <-- key fix: use temp CSV instead of repo file
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["hour", "household_kw", "commercial_kw", "total_kw", "temp_C", "weekend"]
    assert len(df) == 24
    assert all(df["total_kw"] == df["household_kw"] + df["commercial_kw"])
    assert not df["weekend"].iloc[0]

    # Test weekend profile
    df_weekend = get_daily_load(
        day_of_year=180,
        n_households=10,
        n_commercial=1,
        weekend=True,
        seed=42,
        path=test_csv,  # <-- same temp CSV
    )
    assert df_weekend["weekend"].iloc[0]

    # Test reproducibility
    df1 = get_daily_load(day_of_year=180, seed=42, path=test_csv)
    df2 = get_daily_load(day_of_year=180, seed=42, path=test_csv)
    assert all(df1["total_kw"] == df2["total_kw"])
