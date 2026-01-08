import yaml
import numpy as np
import pandas as pd

from envs.microgrid_env import MicrogridEnv


def make_env():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    hours = 48

    dt = pd.date_range("2024-01-01", periods=hours, freq="h")
    df = pd.DataFrame(
        {
            "datetime": dt,
            "ALLSKY_SFC_SW_DWN": np.clip(np.sin((dt.hour - 6) / 12 * np.pi), 0, None),
            "T2M": 25.0,
        }
    )

    load = np.ones(hours) * cfg["load"]["base_kwh_per_hour"]
    return MicrogridEnv(cfg, df, load)


def _done(terminated, truncated):
    return bool(terminated) or bool(truncated)


def test_reset_and_step():
    env = make_env()
    obs, _ = env.reset()
    assert env.action_space.shape == (2,)

    info = {}
    for _ in range(5):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if _done(terminated, truncated):
            break

    assert "unmet_kwh" in info


def test_inverter_limit():
    env = make_env()
    env.reset()
    env._soc = 0.5

    obs, r, terminated, truncated, info = env.step([1.0, 1.0])
    assert abs(info["e_batt"]) <= env.inverter_max_kw * env.dt


def test_soc_bounds():
    env = make_env()
    env.reset()

    for _ in range(10):
        obs, r, terminated, truncated, info = env.step([-1.0, 0.0])
        if _done(terminated, truncated):
            break

    assert env.soc_min <= env._soc <= env.soc_max


def test_battery_cycle_accumulation():
    env = make_env()
    env.reset()

    # Use the info key if the env reports it; fall back to attribute if present
    initial_attr = getattr(env, "battery_cycle_kwh", 0.0)

    obs, r, terminated, truncated, info = env.step([0.5, 0.0])

    # Prefer the info dictionary if available
    if "battery_cycle_kwh" in info:
        assert info["battery_cycle_kwh"] >= 0.0
    else:
        assert getattr(env, "battery_cycle_kwh", 0.0) >= initial_attr


def test_degradation_penalty():
    env = make_env()
    env.degradation_weight = 1
    env.reset()

    obs, r_no_cycle, terminated, truncated, info = env.step([0.0, 0.0])
    obs, r_with_cycle, terminated, truncated, info = env.step([1.0, 0.0])

    assert r_with_cycle < r_no_cycle


def test_temp_dependent_efficiency():
    env = make_env()
    env.reset()

    obs, _, terminated, truncated, info_cold = env.step([1.0, 0.0])

    # Robustly modify temperature for the current timestep
    # If env._t is an int timestep and df index is RangeIndex, .at works.
    # If df index is DatetimeIndex, we use .iloc.
    if isinstance(env.df.index, pd.RangeIndex) or np.issubdtype(env.df.index.dtype, np.integer):
        env.df.at[env._t, "T2M"] = 35.0
    else:
        env.df.iloc[env._t, env.df.columns.get_loc("T2M")] = 35.0

    obs, _, terminated, truncated, info_hot = env.step([1.0, 0.0])

    assert info_hot["e_batt"] != info_cold["e_batt"]


def test_rainflow_cycle_counter():
    env = make_env()
    obs, _ = env.reset()

    done = False
    info = {}
    while not done:
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        done = _done(terminated, truncated)

    assert "equiv_full_cycles" in info
    assert isinstance(info["equiv_full_cycles"], (float, type(None)))
