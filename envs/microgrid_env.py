from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class MicrogridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, solar_df: pd.DataFrame, load_series: np.ndarray):
        super().__init__()
        self.cfg = config
        self.df = solar_df.reset_index(drop=True)
        self.load = load_series
        assert len(self.df) >= len(self.load), "solar/load length mismatch"

        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.5, 2.0, 1.0, 5.0], dtype=np.float32)
        )

        b = self.cfg["battery"]
        d = self.cfg["diesel"]
        s = self.cfg["solar"]
        self.batt_cap = b["capacity_kwh"]
        self.batt_eta_rt = b["roundtrip_efficiency"]
        self.eta_ch = np.sqrt(self.batt_eta_rt)
        self.eta_dis = np.sqrt(self.batt_eta_rt)
        self.batt_p_max_ch = b["max_charge_kw"]
        self.batt_p_max_dis = b["max_discharge_kw"]
        self.soc_min = b["soc_min_pct"]
        self.soc_max = b["soc_max_pct"]
        self.diesel_kw_rated = d["rated_kw"]
        self.diesel_kwh_per_l = d["kwh_per_liter"]
        self.diesel_min_frac = d["min_loading_pct"]
        self.fuel_tank_l = d["fuel_tank_liters"]
        self.fuel_cost = d["fuel_cost_per_liter"]
        self.co2_kg_per_l = d["co2_kg_per_liter"]
        self.panel_area = s["panel_area_m2"]
        self.pv_eff = s["efficiency"]
        self.temp_coeff = s["temp_coeff_pct_per_C"]
        self.derate_soiling = s["derate_soiling"]
        self.rw = self.cfg["reward"]
        self._t = 0
        self._soc = None
        self._fuel = None
        self._done_steps = 0
        self._episode_steps = int(self.cfg["time"]["episode_days"] * 24 / self.cfg["time"]["step_hours"])

    def _obs(self):
        ts = self.df.loc[self._t, "datetime"]
        hour = ts.hour / 23.0
        ghi = float(self.df.loc[self._t, "ALLSKY_SFC_SW_DWN"])
        temp = float(self.df.loc[self._t, "T2M"]) / 50.0
        load_kwh = float(self.load[self._t])
        return np.array([float(self._soc), float(self._fuel), ghi, temp, hour, min(load_kwh, 5.0)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._done_steps = 0
        self._soc = self.cfg["battery"]["soc_init_pct"]
        self._fuel = self.cfg["diesel"]["fuel_init_pct"]
        return self._obs(), {}

    def step(self, action):
        a_batt = float(np.clip(action[0], -1.0, 1.0))
        a_diesel = float(np.clip(action[1], 0.0, 1.0))

        dt = self.cfg["time"]["step_hours"]
        p_batt = (a_batt * (self.batt_p_max_dis if a_batt >= 0 else self.batt_p_max_ch))
        e_batt = p_batt * dt

        p_diesel = a_diesel * self.diesel_kw_rated
        if a_diesel > 0.0:
            p_diesel = max(p_diesel, self.diesel_kw_rated * self.diesel_min_frac)
        e_diesel = p_diesel * dt

        ghi = float(self.df.loc[self._t, "ALLSKY_SFC_SW_DWN"])
        temp_C = float(self.df.loc[self._t, "T2M"])
        pv_eff_temp = self.pv_eff * (1.0 + self.temp_coeff * (temp_C - 25.0))
        e_solar = max(0.0, ghi * self.panel_area * pv_eff_temp * self.derate_soiling)

        e_load = float(self.load[self._t])

        soc_kwh = self._soc * self.batt_cap
        if e_batt >= 0:
            e_batt_delivered = min(e_batt, soc_kwh - self.soc_min * self.batt_cap) * self.eta_dis
            soc_kwh -= e_batt_delivered / self.eta_dis
        else:
            e_batt_store_req = -e_batt
            cap_room = self.soc_max * self.batt_cap - soc_kwh
            e_storable = min(e_batt_store_req * self.eta_ch, cap_room)
            soc_kwh += e_storable
            e_batt_delivered = -e_storable / self.eta_ch

        self._soc = np.clip(soc_kwh / self.batt_cap, self.soc_min, self.soc_max)

        liters_used = e_diesel / self.diesel_kwh_per_l if e_diesel > 0 else 0.0
        fuel_abs = self._fuel * self.fuel_tank_l - liters_used
        fuel_abs = max(0.0, fuel_abs)
        self._fuel = fuel_abs / self.fuel_tank_l

        supply = e_solar + max(0.0, e_batt_delivered) + e_diesel
        sinks  = e_load + max(0.0, -e_batt_delivered)
        residual = supply - sinks
        curtailment = residual if residual >= 0 else 0.0
        unmet = -residual if residual < 0 else 0.0

        r = 0.0
        r -= self.rw["blackout_penalty_per_kwh"] * unmet
        r -= self.rw["diesel_cost_weight"] * liters_used * self.fuel_cost
        r -= self.rw["battery_cycle_weight"] * abs(e_batt_delivered)
        r -= self.rw["curtailment_weight"] * curtailment
        if self.soc_min + 0.05 <= self._soc <= self.soc_max - 0.05:
            r += self.rw["keep_soc_in_band_bonus"]

        self._t += 1
        terminated = (self._t >= len(self.load)) or (self._t >= self._episode_steps)
        info = {"e_solar": e_solar, "e_load": e_load, "e_batt": e_batt_delivered, "e_diesel": e_diesel,
                "unmet_kwh": unmet, "curtail_kwh": curtailment, "liters_used": liters_used,
                "soc": self._soc, "fuel": self._fuel}
        return self._obs(), float(r), terminated, False, info

    def render(self):
        pass
