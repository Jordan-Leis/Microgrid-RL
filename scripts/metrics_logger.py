"""
Centralized logging utilities for microgrid RL training and evaluation.

Provides consistent, structured metric logging across all algorithms and scenarios.
"""
from __future__ import annotations
import os
import csv
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class MetricsLogger:
    """
    Structured metrics logger for microgrid RL experiments.
    
    Creates a run directory with:
    - metrics.csv: Timestep-by-timestep metrics
    - config.yaml: Configuration snapshot
    - summary.json: Final summary statistics
    """
    
    def __init__(
        self,
        run_id: str,
        algorithm: str,
        scenario: str,
        seed: int,
        config_path: Optional[str] = None,
        base_dir: str = "outputs/metrics"
    ):
        """
        Initialize metrics logger.
        
        Args:
            run_id: Unique run identifier (e.g., 'sac_nairobi_180d_0')
            algorithm: Algorithm name ('sac', 'a2c', 'ppo', etc.)
            scenario: Scenario description (e.g., 'nairobi_180d')
            seed: Random seed used
            config_path: Path to config file to copy
            base_dir: Base directory for metrics output
        """
        self.run_id = run_id
        self.algorithm = algorithm
        self.scenario = scenario
        self.seed = seed
        
        # Create run directory
        self.run_dir = Path(base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.yaml"
        self.summary_path = self.run_dir / "summary.json"
        
        # Copy config if provided
        if config_path and os.path.exists(config_path):
            shutil.copy(config_path, self.config_path)
        
        # CSV writer
        self.csv_file = None
        self.csv_writer = None
        self.csv_header = None
        
        # Tracking
        self.start_time = time.time()
        self.metrics_history = []
    
    def _get_csv_header(self) -> list[str]:
        """Define CSV header columns."""
        return [
            # Core identifiers
            'timestep',
            'episode',
            'wallclock_time',
            
            # Episode metrics
            'total_reward',
            'mean_reward',
            'std_reward',
            'episode_length',
            
            # Energy metrics
            'unmet_kwh',
            'diesel_liters',
            'curtailment_kwh',
            'solar_total_kwh',
            
            # Battery metrics
            'soc_mean',
            'soc_min',
            'soc_max',
            'battery_cycles',
            
            # Fuel metrics
            'fuel_mean',
            'fuel_end',
            
            # Derived metrics
            'blackout_hours',
            'diesel_cost',
            'co2_emissions_kg',
            
            # Training metrics (common)
            'learning_rate',
            'entropy',
            
            # Algorithm-specific (SAC)
            'actor_loss',
            'critic_loss',
            'ent_coef',
            'ent_coef_loss',
            
            # Algorithm-specific (A2C/PPO)
            'value_loss',
            'policy_gradient_loss',
            'explained_variance',
            'clip_fraction',
            'approx_kl',
        ]
    
    def open_csv(self):
        """Open CSV file and write header."""
        if self.csv_file is None:
            self.csv_header = self._get_csv_header()
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_header)
            self.csv_writer.writeheader()
            self.csv_file.flush()
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log a single row of metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
                    Missing values will be filled with empty strings
        """
        if self.csv_writer is None:
            self.open_csv()
        
        # Add wallclock time
        metrics['wallclock_time'] = time.time() - self.start_time
        
        # Create row with all columns (fill missing with '')
        row = {col: metrics.get(col, '') for col in self.csv_header}
        
        # Write to CSV
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        # Store for summary
        self.metrics_history.append(metrics)
    
    def save_summary(self, additional_info: Optional[Dict[str, Any]] = None):
        """
        Save final summary JSON.
        
        Args:
            additional_info: Additional fields to include in summary
        """
        summary = {
            'run_id': self.run_id,
            'algorithm': self.algorithm,
            'scenario': self.scenario,
            'seed': self.seed,
            'training_time_seconds': time.time() - self.start_time,
        }
        
        # Add final metrics if available
        if self.metrics_history:
            final = self.metrics_history[-1]
            summary.update({
                'total_timesteps': final.get('timestep', 0),
                'total_episodes': final.get('episode', 0),
                'final_mean_reward': final.get('mean_reward', None),
                'final_unmet_kwh': final.get('unmet_kwh', None),
                'final_diesel_liters': final.get('diesel_liters', None),
                'final_co2_kg': final.get('co2_emissions_kg', None),
            })
        
        # Add any additional info
        if additional_info:
            summary.update(additional_info)
        
        # Write JSON
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def close(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None


class EpisodeMetricsAccumulator:
    """
    Accumulates metrics from environment info dicts during episode rollout.
    
    Designed to work with MicrogridEnv's info dict.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulators for new episode."""
        self.total_reward = 0.0
        self.steps = 0
        
        # Energy accumulators
        self.unmet_kwh = 0.0
        self.diesel_liters = 0.0
        self.curtailment_kwh = 0.0
        self.solar_total_kwh = 0.0
        
        # Battery tracking
        self.soc_values = []
        self.battery_energy_cycled = 0.0
        
        # Fuel tracking
        self.fuel_values = []
        
        # Derived metrics
        self.blackout_steps = 0
    
    def update(self, reward: float, info: Dict[str, Any]):
        """
        Update accumulators with step data.
        
        Args:
            reward: Step reward
            info: Environment info dict with keys:
                  e_solar, e_load, e_batt, e_diesel, unmet_kwh,
                  curtail_kwh, liters_used, soc, fuel
        """
        self.total_reward += reward
        self.steps += 1
        
        # Accumulate from info dict
        self.unmet_kwh += info.get('unmet_kwh', 0.0)
        self.diesel_liters += info.get('liters_used', 0.0)
        self.curtailment_kwh += info.get('curtail_kwh', 0.0)
        self.solar_total_kwh += info.get('e_solar', 0.0)
        
        # Track battery
        soc = info.get('soc')
        if soc is not None:
            self.soc_values.append(soc)
        
        e_batt = info.get('e_batt', 0.0)
        self.battery_energy_cycled += abs(e_batt)
        
        # Track fuel
        fuel = info.get('fuel')
        if fuel is not None:
            self.fuel_values.append(fuel)
        
        # Count blackout steps
        if info.get('unmet_kwh', 0.0) > 0.01:  # threshold for numerical noise
            self.blackout_steps += 1
    
    def get_episode_metrics(
        self,
        diesel_cost_per_liter: float = 1.5,
        co2_kg_per_liter: float = 2.68,
        step_hours: float = 1.0,
        battery_capacity_kwh: float = 80.0
    ) -> Dict[str, float]:
        """
        Get accumulated episode metrics.
        
        Args:
            diesel_cost_per_liter: Fuel cost (USD/L)
            co2_kg_per_liter: CO2 emissions factor (kg/L)
            step_hours: Duration of each timestep (hours)
            battery_capacity_kwh: Battery capacity for cycle calculation
        
        Returns:
            Dictionary of episode metrics
        """
        metrics = {
            'total_reward': self.total_reward,
            'episode_length': self.steps,
            'unmet_kwh': self.unmet_kwh,
            'diesel_liters': self.diesel_liters,
            'curtailment_kwh': self.curtailment_kwh,
            'solar_total_kwh': self.solar_total_kwh,
        }
        
        # Battery metrics
        if self.soc_values:
            metrics['soc_mean'] = np.mean(self.soc_values)
            metrics['soc_min'] = np.min(self.soc_values)
            metrics['soc_max'] = np.max(self.soc_values)
        else:
            metrics['soc_mean'] = metrics['soc_min'] = metrics['soc_max'] = ''
        
        # Battery cycles (full equivalent cycles)
        if battery_capacity_kwh > 0:
            metrics['battery_cycles'] = self.battery_energy_cycled / (2 * battery_capacity_kwh)
        else:
            metrics['battery_cycles'] = ''
        
        # Fuel metrics
        if self.fuel_values:
            metrics['fuel_mean'] = np.mean(self.fuel_values)
            metrics['fuel_end'] = self.fuel_values[-1]
        else:
            metrics['fuel_mean'] = metrics['fuel_end'] = ''
        
        # Derived metrics
        metrics['blackout_hours'] = self.blackout_steps * step_hours
        metrics['diesel_cost'] = self.diesel_liters * diesel_cost_per_liter
        metrics['co2_emissions_kg'] = self.diesel_liters * co2_kg_per_liter
        
        return metrics


def make_run_id(algorithm: str, scenario: str, seed: int) -> str:
    """
    Create standardized run ID.
    
    Args:
        algorithm: Algorithm name (e.g., 'sac', 'a2c')
        scenario: Scenario description (e.g., 'nairobi_180d')
        seed: Random seed
    
    Returns:
        Run ID string (e.g., 'sac_nairobi_180d_0')
    """
    return f"{algorithm}_{scenario}_{seed}"


def extract_sb3_training_metrics(logger) -> Dict[str, Any]:
    """
    Extract training metrics from Stable-Baselines3 logger.
    
    Args:
        logger: SB3 logger object (model.logger)
    
    Returns:
        Dictionary of training metrics
    """
    metrics = {}
    
    # Try to get common metrics from name_to_value dict
    if hasattr(logger, 'name_to_value'):
        name_to_value = logger.name_to_value
        
        # Map SB3 keys to our schema
        key_mapping = {
            'train/learning_rate': 'learning_rate',
            'train/entropy_loss': 'entropy',
            'train/actor_loss': 'actor_loss',
            'train/critic_loss': 'critic_loss',
            'train/ent_coef': 'ent_coef',
            'train/ent_coef_loss': 'ent_coef_loss',
            'train/value_loss': 'value_loss',
            'train/policy_gradient_loss': 'policy_gradient_loss',
            'train/explained_variance': 'explained_variance',
            'train/clip_fraction': 'clip_fraction',
            'train/approx_kl': 'approx_kl',
            'rollout/ep_rew_mean': 'mean_reward',
            'rollout/ep_len_mean': 'episode_length',
        }
        
        for sb3_key, our_key in key_mapping.items():
            if sb3_key in name_to_value:
                metrics[our_key] = name_to_value[sb3_key]
    
    return metrics
