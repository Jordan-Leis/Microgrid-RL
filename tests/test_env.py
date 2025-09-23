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
