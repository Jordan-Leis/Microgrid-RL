"""
Custom Stable-Baselines3 callback for structured metrics logging.
"""
from __future__ import annotations
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
from scripts.metrics_logger import MetricsLogger, EpisodeMetricsAccumulator, extract_sb3_training_metrics

# Optional W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class StructuredMetricsCallback(BaseCallback):
    """
    Callback for logging structured metrics during SB3 training.
    
    Logs to CSV with consistent schema across algorithms.
    """
    
    def __init__(
        self,
        metrics_logger: MetricsLogger,
        log_freq: int = 1000,
        eval_env=None,
        n_eval_episodes: int = 5,
        diesel_cost_per_liter: float = 1.5,
        co2_kg_per_liter: float = 2.68,
        step_hours: float = 1.0,
        battery_capacity_kwh: float = 80.0,
        use_wandb: bool = False,
        wandb_project: str = "microgrid-rl",
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[list] = None,
        verbose: int = 0
    ):
        """
        Args:
            metrics_logger: MetricsLogger instance
            log_freq: Log metrics every N timesteps
            eval_env: Optional evaluation environment
            n_eval_episodes: Number of episodes for evaluation
            diesel_cost_per_liter: Fuel cost for derived metrics
            co2_kg_per_liter: CO2 emissions factor
            step_hours: Duration of each timestep
            battery_capacity_kwh: Battery capacity for cycle calculation
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_entity: W&B entity (username/team)
            wandb_tags: Optional tags for W&B run
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.metrics_logger = metrics_logger
        self.log_freq = log_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        
        # Environment parameters for derived metrics
        self.diesel_cost_per_liter = diesel_cost_per_liter
        self.co2_kg_per_liter = co2_kg_per_liter
        self.step_hours = step_hours
        self.battery_capacity_kwh = battery_capacity_kwh
        
        # W&B configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags or []
        
        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")
            self.use_wandb = False
        
        # Episode tracking
        self.episode_accumulator = EpisodeMetricsAccumulator()
        self.episode_count = 0
        self.completed_episodes_metrics = []
        
    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.metrics_logger.open_csv()
        
        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.metrics_logger.run_id,
                tags=self.wandb_tags,
                config={
                    'algorithm': self.metrics_logger.algorithm,
                    'scenario': self.metrics_logger.scenario,
                    'seed': self.metrics_logger.seed,
                }
            )
            print(f"W&B logging enabled: {wandb.run.url}")
        
    def _on_step(self) -> bool:
        """
        Called at each step.
        
        Returns:
            bool: If False, training stops
        """
        # Get info from last step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]  # First env in vectorized env
            reward = self.locals['rewards'][0]
            done = self.locals['dones'][0]
            
            # Accumulate episode metrics
            self.episode_accumulator.update(reward, info)
            
            # Check if episode finished
            if done:
                # Get episode metrics
                ep_metrics = self.episode_accumulator.get_episode_metrics(
                    diesel_cost_per_liter=self.diesel_cost_per_liter,
                    co2_kg_per_liter=self.co2_kg_per_liter,
                    step_hours=self.step_hours,
                    battery_capacity_kwh=self.battery_capacity_kwh
                )
                
                self.completed_episodes_metrics.append(ep_metrics)
                self.episode_count += 1
                
                # Reset for next episode
                self.episode_accumulator.reset()
        
        # Log at specified frequency
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()
        
        return True
    
    def _log_metrics(self):
        """Log current metrics to CSV."""
        metrics = {
            'timestep': self.num_timesteps,
            'episode': self.episode_count,
        }
        
        # Add training metrics from SB3 logger
        sb3_metrics = extract_sb3_training_metrics(self.model.logger)
        metrics.update(sb3_metrics)
        
        # Add latest episode metrics if available
        if self.completed_episodes_metrics:
            # Use rolling average of last few episodes
            recent_episodes = self.completed_episodes_metrics[-10:]
            
            for key in ['unmet_kwh', 'diesel_liters', 'curtailment_kwh', 
                       'solar_total_kwh', 'soc_mean', 'blackout_hours',
                       'diesel_cost', 'co2_emissions_kg', 'battery_cycles']:
                values = [ep.get(key, np.nan) for ep in recent_episodes 
                         if ep.get(key, '') != '']
                if values:
                    metrics[key] = np.mean(values)
        
        # Optional: Evaluate on eval_env
        if self.eval_env is not None:
            eval_metrics = self._evaluate()
            metrics.update(eval_metrics)
        
        # Log to CSV
        self.metrics_logger.log(metrics)
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log(metrics, step=self.num_timesteps)
        
        if self.verbose > 0:
            reward = metrics.get('mean_reward', None)
            unmet = metrics.get('unmet_kwh', None)
            reward_str = f"{reward:.2f}" if reward is not None else "N/A"
            unmet_str = f"{unmet:.2f}" if unmet is not None else "N/A"
            print(f"[Step {self.num_timesteps}] reward={reward_str}, unmet={unmet_str} kWh")
    
    def _evaluate(self):
        """Run evaluation episodes and return metrics."""
        if self.eval_env is None:
            return {}
        
        episode_rewards = []
        episode_unmet = []
        episode_diesel = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()[0]
            done = False
            ep_reward = 0.0
            ep_unmet = 0.0
            ep_diesel = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                ep_unmet += info.get('unmet_kwh', 0.0)
                ep_diesel += info.get('liters_used', 0.0)
            
            episode_rewards.append(ep_reward)
            episode_unmet.append(ep_unmet)
            episode_diesel.append(ep_diesel)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'unmet_kwh': np.mean(episode_unmet),
            'diesel_liters': np.mean(episode_diesel),
        }
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Final log
        self._log_metrics()
        
        # Close logger
        self.metrics_logger.close()
        
        # Finish W&B run if enabled
        if self.use_wandb:
            wandb.finish()
