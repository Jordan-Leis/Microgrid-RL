# tests/test_nasa_power.py
import os
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import data_sources.nasa_power as npower


@pytest.fixture
def sample_hourly_utc_df():
    # 6 hours in UTC, remove one hour to simulate a gap
    base = datetime(2021, 3, 13, 20, tzinfo=timezone.utc)
    times = [base + timedelta(hours=i) for i in range(6)]
    del times[2]  # remove base+2h
    vals = [10, 12, 15, 14, 13]
    return pd.DataFrame(index=pd.DatetimeIndex(times), data={"T2M": vals})


def test_fill_hourly_gaps(sample_hourly_utc_df):
    df = sample_hourly_utc_df.copy()
    df_local = df.tz_convert(ZoneInfo("UTC"))

    df_filled = npower._fill_hourly_gaps(df_local)

    # should now have continuous 6 hours
    assert len(df_filled) == 6

    # ensure missing hour exists and is interpolated
    missing_time = datetime(2021, 3, 13, 22, tzinfo=timezone.utc)
    assert missing_time in df_filled.index

    v = float(df_filled.loc[missing_time, "T2M"])
    assert 12 < v < 15


def _fake_power_csv(date_yyyymmdd: str, hours, values_by_param):
    """
    Build a minimal NASA POWER-like CSV response.
    Expected by your parser: DATE + HOUR + parameter columns.
    """
    header = "DATE,HOUR," + ",".join(values_by_param.keys())
    rows = [header]
    for h in hours:
        row = [date_yyyymmdd, str(h)]
        for p in values_by_param.keys():
            row.append(str(values_by_param[p][h]))
        rows.append(",".join(row))
    # add a comment line to mimic POWER preamble
    return "# NASA POWER\n" + "\n".join(rows) + "\n"


class _Resp:
    def __init__(self, text):
        self.text = text
        self.headers = {"Content-Type": "text/csv"}

    def raise_for_status(self):
        return None


def test_fetch_power_hourly_localize_and_hour_feature_dst(tmp_path, monkeypatch):
    # Send cache into temp dir so tests don't touch real cache
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(npower, "CACHE_DIR", str(cache_dir))

    # UTC hours that map into NY around DST start date (we don't assert exact hours,
    # just that hour_of_day matches localized index hour)
    csv_text = _fake_power_csv(
        "20210314",
        hours=[4, 5, 6],
        values_by_param={"T2M": {4: 1, 5: 2, 6: 3}},
    )

    def fake_get(*args, **kwargs):
        return _Resp(csv_text)

    monkeypatch.setattr(npower.requests, "get", fake_get)

    df_local = npower.fetch_power_hourly(
        lat=10.0,
        lon=20.0,
        start="2021-03-14",
        end="2021-03-14",
        parameters="T2M",
        tz="America/New_York",
    )

    assert "hour_of_day" in df_local.columns
    assert str(df_local.index.tz) == "America/New_York"
    assert (df_local["hour_of_day"].values == df_local.index.hour.values).all()


def test_cache_save_and_reuse(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(npower, "CACHE_DIR", str(cache_dir))

    csv_text = _fake_power_csv(
        "20220101",
        hours=[0, 1, 2],
        values_by_param={"T2M": {0: 1, 1: 2, 2: 3}},
    )

    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return _Resp(csv_text)

    monkeypatch.setattr(npower.requests, "get", fake_get)

    # First call: should hit "network" and create cache
    df1 = npower.fetch_power_hourly(
        lat=10.0,
        lon=20.0,
        start="2022-01-01",
        end="2022-01-01",
        parameters="T2M",
        tz=None,
        timeout=60,
    )
    assert call_count["n"] == 1

    expected_cache = npower._cache_path(10.0, 20.0, "2022-01-01", "2022-01-01", "T2M", None)
    assert os.path.exists(expected_cache)

    # Second call: should reuse cache (no new network calls)
    df2 = npower.fetch_power_hourly(
        lat=10.0,
        lon=20.0,
        start="2022-01-01",
        end="2022-01-01",
        parameters="T2M",
        tz=None,
        timeout=60,
    )
    assert call_count["n"] == 1

    pd.testing.assert_frame_equal(df1, df2)
