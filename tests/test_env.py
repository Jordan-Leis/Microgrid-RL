import yaml, numpy as np, pandas as pd
from envs.microgrid_env import MicrogridEnv

def make_env():
    cfg = yaml.safe_load(open('configs/default.yaml'))
    hours = 48
    dt = pd.date_range('2024-01-01', periods=hours, freq='H')
    df = pd.DataFrame({'datetime': dt,
                       'ALLSKY_SFC_SW_DWN': np.clip(np.sin((dt.hour-6)/12*np.pi), 0, None),
                       'T2M': 25.0})
    load = np.ones(hours) * cfg['load']['base_kwh_per_hour']
    return MicrogridEnv(cfg, df, load)

def test_reset_and_step():
    env = make_env()
    obs, _ = env.reset()
    assert env.action_space.shape == (2,)
    for _ in range(5):
        obs, r, done, _, info = env.step(env.action_space.sample())
    assert 'unmet_kwh' in info

# ------------------ Additional tests ------------------ #
def test_inverter_limit():
    env = make_env()
    env.reset()
    env._soc = 0.5
    obs, r, done, _, info = env.step([1.0, 1.0])
    assert abs(info["e_batt"]) <= env.inverter_max_kw * env.dt

def test_soc_bounds():
    env = make_env()
    env.reset()
    for _ in range(10):
        obs, r, done, _, info = env.step([-1.0, 0.0])
    assert env._soc >= env.soc_min and env._soc <= env.soc_max

def test_battery_cycle_accumulation():
    env = make_env()
    env.reset()
    initial_throughput = getattr(env, "battery_cycle_kwh", 0.0)
    obs, r, done, _, info = env.step([0.5, 0.0])
    assert info["battery_cycle_kwh"] >= initial_throughput

def test_degradation_penalty():
    env = make_env()
    env.degradation_weight = 1 # choose
    env.reset()
    obs, r_no_cycle, _, _, _ = env.step([0.0, 0.0])
    obs, r_with_cycle, _, _, _ = env.step([1.0, 0.0])
    assert r_with_cycle < r_no_cycle

def test_temp_dependent_efficiency():
    env = make_env()
    env.reset()
    obs, _, _, _, info_cold = env.step([1.0, 0.0])
    env.df.loc[env._t, "T2M"] = 35.0
    obs, _, _, _, info_hot = env.step([1.0, 0.0])
    assert info_hot["e_batt"] != info_cold["e_batt"]

def test_rainflow_cycle_counter():
    env = make_env()
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert "equiv_full_cycles" in info
    assert isinstance(info["equiv_full_cycles"], (float, type(None)))

def test_rainflow_cycle_counter():
    env = make_env()
    obs, - = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert "equiv_full_cycles" in info assert isinstance(infol"equiv_full_cycles"], (float, type(None)))
