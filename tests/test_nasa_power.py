# tests/test_nasa_power.py
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import json
from pathlib import Path

import data_sources.nasa_power as npower

@pytest.fixture
def sample_hourly_utc_df():
    # create hourly UTC data for 6 hours, but drop one hour to simulate missing hour
    base = datetime(2021, 3, 13, 20, tzinfo=timezone.utc)  # around US DST (Mar 14 2021)
    times = [base + timedelta(hours=i) for i in range(6)]
    # drop the 3rd hour to simulate a missing hour
    del times[2]
    vals = [10, 12, 15, 14, 13]  # correspond to times[0],1,3,4,5
    df = pd.DataFrame(index=pd.DatetimeIndex(times), data={"T2M": vals})
    return df

def test_reindex_and_impute(sample_hourly_utc_df):
    # ensure interpolation fills missing hour
    # localize as UTC index with tz
    df = sample_hourly_utc_df.copy()
    # df index already tz-aware UTC
    # convert to local tz (UTC for simplicity here)
    df_local = df.tz_convert(ZoneInfo("UTC"))
    df_filled = npower._reindex_and_impute(df_local)
    # should have original 6 hours now
    assert len(df_filled) == 6
    # check that the previously-missing index exists and value is interpolated (between 12 and 15)
    # find the interpolated value
    vals = df_filled["T2M"].values
    assert not pd.isna(vals).any()
    # Interpolated value should be >12 and <15
    assert any((v > 12 and v < 15) for v in vals)

def test_localize_and_hour_feature_dst():
    # Create a simple 3-hour sequence in UTC that spans US DST start: 2021-03-14 01:00 UTC -> local conversion to America/New_York (EST->EDT)
    times = [datetime(2021,3,14,h,tzinfo=timezone.utc) for h in (4,5,6)]  # corresponds to local 00:00,01:00,02:00? depends
    df = pd.DataFrame(index=pd.DatetimeIndex(times), data={"T2M":[1,2,3]})
    df_local = npower._localize_and_fix_hours(df, tz_name="America/New_York")
    # hour_of_day column should exist and be consistent with local timezone
    assert "hour_of_day" in df_local.columns
    # ensure tz is the requested zone
    assert str(df_local.index.tz) == "America/New_York"
    # check a couple of hour_of_day values are in 0-23
    assert all(0 <= h <= 23 for h in df_local["hour_of_day"].unique())

def test_cache_save_and_reuse(tmp_path, monkeypatch):
    # monkeypatch the API call to return a small df
    def fake_api(lat, lon, start, end, parameters, api_base=None):
        times = pd.date_range(start="2022-01-01", periods=3, freq="H", tz=timezone.utc)
        return pd.DataFrame(index=times, data={"T2M":[1,2,3]})
    monkeypatch.setattr(npower, "_call_power_api", fake_api)

    cache_dir = str(tmp_path / "cache")
    # first call should create cache
    df1 = npower.get_power(10.0, 20.0, "2022-01-01", "2022-01-01", location_tz="UTC", parameters="T2M", cache_dir=cache_dir, use_cache=True)
    # check that cache files exist
    # compute hash to find cache file
    hsh = npower._params_hash(10.0, 20.0, "2022-01-01", "2022-01-01", "T2M")
    cache_path, meta_path = npower._cache_path(cache_dir, hsh)
    assert Path(meta_path).exists()
    # second call should reuse cache; monkeypatch _call_power_api to raise if used
    def failing_api(*args, **kwargs):
        raise RuntimeError("API should not be called when cache available")
    monkeypatch.setattr(npower, "_call_power_api", failing_api)
    df2 = npower.get_power(10.0, 20.0, "2022-01-01", "2022-01-01", location_tz="UTC", parameters="T2M", cache_dir=cache_dir, use_cache=True)
    # results should match
    pd.testing.assert_frame_equal(df1, df2)

