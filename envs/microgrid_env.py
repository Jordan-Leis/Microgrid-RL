# envs/microgrid_env.py
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class MicrogridEnv(gym.Env):
    """
    Solar + Battery + Diesel microgrid environment (Gymnasium).

    Action space (continuous):
        [0] a_batt  in [-1, 1]   -> negative: charge, positive: discharge
        [1] a_diesel in [0, 1]   -> throttle fraction of rated kW
            (if >0, a minimum loading constraint is enforced)

    Observation (6-dim):
        [soc, fuel, ghi, temp_norm, hour_norm, load_capped]

        soc, fuel            : [0..1]
        ghi                  : kWh/m^2 over the step (kept as-is, typical 0..1.5)
        temp_norm            : ambient temperature / 50 (rough normalization)
        hour_norm            : hour_of_day / 23
        load_capped          : kWh demand this step, capped at 5 for scaling

    Episode termination:
        - We use Gymnasium's (terminated, truncated) convention.
        - `terminated=False` unless you add a true terminal condition.
        - `truncated=True` when we hit the episode/horizon or run out of data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, solar_df: pd.DataFrame, load_series: np.ndarray):
        super().__init__()

        # --- Inputs ---
        self.cfg = config
        # Ensure monotonically increasing, positional indexing
        self.df = solar_df.reset_index(drop=True)
        self.load = np.asarray(load_series, dtype=float)
        assert len(self.df) >= len(self.load), "solar/load length mismatch"

        # --- Spaces ---
        # a_batt in [-1..1], a_diesel in [0..1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        # obs = [soc, fuel, ghi, temp_norm, hour_norm, load_capped]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.5, 2.0, 1.0, 5.0], dtype=np.float32),
        )

        # --- Unpack config (short names for performance) ---
        b = self.cfg["battery"]
        d = self.cfg["diesel"]
        s = self.cfg["solar"]

        # Battery
        self.batt_cap = float(b["capacity_kwh"])
        self.batt_eta_rt = float(b["roundtrip_efficiency"])
        # Split roundtrip into charge/discharge legs
        self.eta_ch = float(np.sqrt(self.batt_eta_rt))
        self.eta_dis = float(np.sqrt(self.batt_eta_rt))
        self.batt_p_max_ch = float(b["max_charge_kw"])
        self.batt_p_max_dis = float(b["max_discharge_kw"])
        self.soc_min = float(b["soc_min_pct"])
        self.soc_max = float(b["soc_max_pct"])

        # Diesel
        self.diesel_kw_rated = float(d["rated_kw"])
        self.diesel_kwh_per_l = float(d["kwh_per_liter"])
        self.diesel_min_frac = float(d["min_loading_pct"])
        self.fuel_tank_l = float(d["fuel_tank_liters"])
        self.fuel_cost = float(d["fuel_cost_per_liter"])
        self.co2_kg_per_l = float(d["co2_kg_per_liter"])

        # PV
        self.panel_area = float(s["panel_area_m2"])
        self.pv_eff = float(s["efficiency"])
        self.temp_coeff = float(s["temp_coeff_pct_per_C"])
        self.derate_soiling = float(s["derate_soiling"])

        # Reward weights
        self.rw = self.cfg["reward"]

        # Timebase
        self.dt = float(self.cfg["time"]["step_hours"])
        self._episode_steps = int(
            self.cfg["time"]["episode_days"] * 24 / self.cfg["time"]["step_hours"]
        )

        # Runtime state
        self._t = 0               # current time index (position in data)
        self._soc = None          # state of charge [0..1]
        self._fuel = None         # fuel level [0..1]

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _obs(self, t: int | None = None) -> np.ndarray:
        """
        Build observation safely. If t is past the last index, we clamp to the last
        valid row so Pandas is never indexed out of range.
        """
        if t is None:
            t = self._t
        t = int(min(t, len(self.df) - 1))  # clamp for safety

        row = self.df.iloc[t]
        # Feature engineering
        hour_norm = (pd.to_datetime(row["datetime"]).hour) / 23.0
        ghi = float(row["ALLSKY_SFC_SW_DWN"])  # kWh/m^2 over the step
        temp_norm = float(row["T2M"]) / 50.0
        load_kwh = float(self.load[t])

        obs = np.array(
            [
                float(self._soc),
                float(self._fuel),
                ghi,
                temp_norm,
                hour_norm,
                min(load_kwh, 5.0),
            ],
            dtype=np.float32,
        )
        return obs

    # --------------------------------------------------------------------- #
    # Gym API
    # --------------------------------------------------------------------- #
    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._soc = float(self.cfg["battery"]["soc_init_pct"])
        self._fuel = float(self.cfg["diesel"]["fuel_init_pct"])
        return self._obs(), {}

    def step(self, action):
        # ------------------ Parse action ------------------ #
        a_batt = float(np.clip(action[0], -1.0, 1.0))
        a_diesel = float(np.clip(action[1], 0.0, 1.0))

        # ------------------ Battery setpoint --------------- #
        # Positive a_batt => discharge; negative => charge
        p_batt = a_batt * (self.batt_p_max_dis if a_batt >= 0 else self.batt_p_max_ch)
        e_batt_req = p_batt * self.dt  # requested AC-side energy (+discharge, -charge)

        # ------------------ Diesel setpoint ---------------- #
        p_diesel = a_diesel * self.diesel_kw_rated
        if a_diesel > 0.0:
            # Enforce minimum loading if running
            p_diesel = max(p_diesel, self.diesel_kw_rated * self.diesel_min_frac)
        e_diesel = p_diesel * self.dt  # kWh over the step
        liters_used = e_diesel / self.diesel_kwh_per_l if e_diesel > 0 else 0.0

        # ------------------ PV generation ------------------ #
        # NASA POWER hourly ALLSKY_SFC_SW_DWN is kWh/m^2 over the hour.
        # Scale by area and efficiency (and dt for non-1h steps).
        ghi = float(self.df.iloc[self._t]["ALLSKY_SFC_SW_DWN"])  # kWh/m^2
        temp_C = float(self.df.iloc[self._t]["T2M"])
        pv_eff_temp = self.pv_eff * (1.0 + self.temp_coeff * (temp_C - 25.0))
        e_solar = max(0.0, ghi * self.panel_area * pv_eff_temp * self.derate_soiling * self.dt)

        # ------------------ Load demand -------------------- #
        e_load = float(self.load[self._t])  # kWh this step

        # ------------------ Battery physics ---------------- #
        soc_kwh = self._soc * self.batt_cap

        if e_batt_req >= 0:
            # Discharge: deliver energy to the AC bus, limited by SOC and power
            e_deliverable = min(e_batt_req, max(0.0, soc_kwh - self.soc_min * self.batt_cap)) * self.eta_dis
            soc_kwh -= e_deliverable / self.eta_dis
            e_batt_ac = e_deliverable  # positive to AC bus
        else:
            # Charge: draw energy from AC bus, store with charge efficiency
            e_store_request = -e_batt_req  # positive energy to store (AC side)
            cap_room = max(0.0, self.soc_max * self.batt_cap - soc_kwh)
            # Energy that actually becomes chemical energy after efficiency
            e_storable = min(e_store_request * self.eta_ch, cap_room)
            soc_kwh += e_storable
            # AC side sees a draw (negative delivered)
            e_batt_ac = -e_storable / self.eta_ch  # negative to AC bus

        # Update normalized SOC (clamped)
        self._soc = float(np.clip(soc_kwh / self.batt_cap, self.soc_min, self.soc_max))

        # ------------------ Fuel bookkeeping --------------- #
        # Consume fuel based on e_diesel produced
        fuel_abs = self._fuel * self.fuel_tank_l - liters_used
        fuel_abs = max(0.0, fuel_abs)
        self._fuel = fuel_abs / self.fuel_tank_l

        # ------------------ Power balance ------------------ #
        supply = e_solar + max(0.0, e_batt_ac) + e_diesel
        sinks = e_load + max(0.0, -e_batt_ac)
        residual = supply - sinks
        curtailment = max(0.0, residual)
        unmet = max(0.0, -residual)

        # ------------------ Reward ------------------------- #
        r = 0.0
        r -= float(self.rw["blackout_penalty_per_kwh"]) * unmet
        r -= float(self.rw["diesel_cost_weight"]) * liters_used * self.fuel_cost
        r -= float(self.rw["battery_cycle_weight"]) * abs(e_batt_ac)
        r -= float(self.rw["curtailment_weight"]) * curtailment
        # Small shaping bonus for keeping SOC away from hard limits
        if self.soc_min + 0.05 <= self._soc <= self.soc_max - 0.05:
            r += float(self.rw["keep_soc_in_band_bonus"])

        # ------------------ Time advance ------------------- #
        next_t = self._t + 1
        # Horizon reached or we ran out of data
        truncated = (next_t >= self._episode_steps) or (next_t >= len(self.load))
        terminated = False  # set True only for real terminal faults if you add them

        # Clamp internal index so _obs() never reads past the dataframe
        self._t = min(next_t, len(self.df) - 1)

        info = {
            "e_solar": e_solar,
            "e_load": e_load,
            "e_batt": e_batt_ac,
            "e_diesel": e_diesel,
            "unmet_kwh": unmet,
            "curtail_kwh": curtailment,
            "liters_used": liters_used,
            "soc": self._soc,
            "fuel": self._fuel,
        }

        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        # No-op placeholder (hook up a pygame/plotly view if desired)
        pass
