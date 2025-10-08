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
        # Optional diesel advanced settings (backward compatible defaults)
        # Efficiency vs load (specific fuel consumption curve): list of [load_frac, liters_per_kwh]
        self.diesel_sfc_curve = d.get("sfc_liters_per_kwh_curve")  # e.g., [[0.3, 0.33], [0.5, 0.30], [1.0, 0.28]]
        # Startup penalty in liters when transitioning OFF->ON
        self.diesel_startup_liters = float(d.get("startup_liters_penalty", 0.0))
        # Minimum on/off durations in hours
        self.min_on_hours = float(d.get("min_on_hours", 0.0))
        self.min_off_hours = float(d.get("min_off_hours", 0.0))
        # Auto-refuel options
        self.auto_refuel = bool(d.get("auto_refuel", False))
        self.refuel_threshold_pct = float(d.get("refuel_threshold_pct", 0.0))  # trigger when fuel% < threshold
        self.refuel_target_pct = float(d.get("refuel_target_pct", 1.0))        # top up to this level
        self.refuel_cost_per_liter = float(d.get("refuel_cost_per_liter", self.fuel_cost))
        self.refuel_delivery_fee = float(d.get("refuel_delivery_fee", 0.0))

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
        # Compute min on/off steps after dt is known
        self.min_on_steps = int(round(self.min_on_hours / self.dt)) if self.min_on_hours > 0 else 0
        self.min_off_steps = int(round(self.min_off_hours / self.dt)) if self.min_off_hours > 0 else 0

        # Runtime state
        self._t = 0               # current time index (position in data)
        self._soc = None          # state of charge [0..1]
        self._fuel = None         # fuel level [0..1]
        # Genset state for min on/off and startup penalties
        self._genset_on = False
        self._on_steps = 0
        # Pretend it's been off long enough so we can start immediately unless min_off > 0
        self._off_steps = 0

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
        # Reset genset state; allow immediate start if desired (honor min_off if configured)
        self._genset_on = False
        self._on_steps = 0
        # Initialize off counter to at least min_off to allow start at t=0
        self._off_steps = self.min_off_steps
        return self._obs(), {}

    def step(self, action):
        # ------------------ Parse action ------------------ #
        a_batt = float(np.clip(action[0], -1.0, 1.0))
        a_diesel = float(np.clip(action[1], 0.0, 1.0))

        # ------------------ Battery setpoint --------------- #
        # Positive a_batt => discharge; negative => charge
        p_batt = a_batt * (self.batt_p_max_dis if a_batt >= 0 else self.batt_p_max_ch)
        e_batt_req = p_batt * self.dt  # requested AC-side energy (+discharge, -charge)

        # ------------------ Diesel setpoint & fuel use ----- #
        desired_on = a_diesel > 0.0
        started = False
        stopped = False
        # Enforce min on/off gating
        if self._genset_on:
            if (not desired_on) and (self._on_steps >= self.min_on_steps):
                self._genset_on = False
                self._on_steps = 0
                stopped = True
            else:
                # Stay on
                pass
        else:
            if desired_on and (self._off_steps >= self.min_off_steps):
                # Start allowed
                if self.diesel_startup_liters <= self._fuel * self.fuel_tank_l:
                    self._genset_on = True
                    self._off_steps = 0
                    started = True
                else:
                    # Not enough fuel to even start
                    desired_on = False
                    self._genset_on = False

        # Determine setpoint when ON
        if self._genset_on:
            # Enforce minimum loading
            frac = max(a_diesel, self.diesel_min_frac)
            p_diesel = frac * self.diesel_kw_rated
        else:
            p_diesel = 0.0
            frac = 0.0

        # Compute diesel energy produced and liters used, possibly fuel-limited
        e_diesel_req = p_diesel * self.dt  # requested kWh over the step
        liters_available = self._fuel * self.fuel_tank_l

        # Specific fuel consumption (liters per kWh)
        def _diesel_liters_per_kwh(load_frac: float) -> float:
            if self.diesel_sfc_curve and isinstance(self.diesel_sfc_curve, (list, tuple)) and len(self.diesel_sfc_curve) > 0:
                pts = sorted([(float(x), float(y)) for x, y in self.diesel_sfc_curve])
                xs = np.array([p[0] for p in pts], dtype=float)
                ys = np.array([p[1] for p in pts], dtype=float)
                lf = float(np.clip(load_frac, xs[0], xs[-1]))
                return float(np.interp(lf, xs, ys))
            # Fallback to constant efficiency from kWh per liter
            base = self.diesel_kwh_per_l
            return (1.0 / base) if base > 0 else 1e9

        lpkwh = _diesel_liters_per_kwh(frac) if self._genset_on else 0.0
        liters_start = self.diesel_startup_liters if started else 0.0
        liters_need_for_energy = e_diesel_req * lpkwh
        total_liters_needed = liters_start + liters_need_for_energy

        if self._genset_on and total_liters_needed > liters_available:
            # Fuel-limited: reduce energy output
            rem = max(0.0, liters_available - liters_start)
            e_diesel = max(0.0, rem / max(lpkwh, 1e-12))
            liters_used = liters_available  # consume everything
            # Adjust effective fraction for info only
            frac = (e_diesel / max(self.dt, 1e-12)) / max(self.diesel_kw_rated, 1e-12)
            p_diesel = e_diesel / max(self.dt, 1e-12)
        else:
            e_diesel = e_diesel_req if self._genset_on else 0.0
            liters_used = liters_start + (e_diesel * lpkwh if self._genset_on else 0.0)

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

        # Optional auto-refuel
        refueled_liters = 0.0
        refuel_cost = 0.0
        if self.auto_refuel and self.refuel_threshold_pct > 0.0:
            fuel_pct = fuel_abs / self.fuel_tank_l if self.fuel_tank_l > 0 else 0.0
            if fuel_pct < self.refuel_threshold_pct:
                target_l = np.clip(self.refuel_target_pct, 0.0, 1.0) * self.fuel_tank_l
                add = max(0.0, target_l - fuel_abs)
                if add > 0:
                    refueled_liters = add
                    refuel_cost = add * self.refuel_cost_per_liter + self.refuel_delivery_fee
                    fuel_abs = min(self.fuel_tank_l, fuel_abs + add)

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
        # Charge refuel cost similarly
        if refuel_cost > 0:
            r -= float(self.rw["diesel_cost_weight"]) * refuel_cost
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

        # Update genset counters for min on/off tracking
        if self._genset_on:
            self._on_steps += 1
            self._off_steps = 0
        else:
            self._off_steps += 1
            self._on_steps = 0

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
            "genset_on": bool(self._genset_on),
            "diesel_load_frac": float(frac),
            "diesel_liters_per_kwh": float(lpkwh) if self._genset_on else 0.0,
            "startup_liters": float(liters_start),
            "refueled_liters": float(refueled_liters),
            "refuel_cost": float(refuel_cost),
        }

        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        # No-op placeholder (hook up a pygame/plotly view if desired)
        pass
