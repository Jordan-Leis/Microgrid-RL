# Logging Schema Documentation

## Overview
All training and evaluation scripts produce structured metrics logs with identical schema for reproducibility and easy aggregation across algorithms and scenarios.

## Directory Structure
```
outputs/
  metrics/
    {algo}_{scenario}_{seed}/
      metrics.csv
      config.yaml
      summary.json
```

### Naming Convention
- **algo**: Algorithm name (`sac`, `a2c`, etc.)
- **scenario**: Scenario identifier (e.g., `nairobi_180d`, `rural_60d`)
- **seed**: Random seed used (e.g., `0`, `42`)

Example: `outputs/metrics/sac_nairobi_180d_0/`

## CSV Schema: `metrics.csv`

### File Format
- **Format**: CSV with header
- **Delimiter**: `,` (comma)
- **Encoding**: UTF-8
- **Frequency**: One row per logging interval (typically every 1000-10000 timesteps)

### Columns

| Column Name | Type | Unit | Description | Source |
|-------------|------|------|-------------|--------|
| `timestep` | int | steps | Global timestep counter | Training loop |
| `episode` | int | count | Episode number | Training loop |
| `wallclock_time` | float | seconds | Time elapsed since training start | `time.time()` |
| `total_reward` | float | dimensionless | Cumulative episode reward | Episode rollout |
| `mean_reward` | float | dimensionless | Mean reward over evaluation episodes | Evaluation |
| `std_reward` | float | dimensionless | Std dev of reward over evaluation episodes | Evaluation |
| `episode_length` | float | steps | Mean episode length | SB3 logger |
| `unmet_kwh` | float | kWh | Total unmet energy demand | Environment `info` |
| `diesel_liters` | float | L | Total diesel fuel consumed | Environment `info` |
| `curtailment_kwh` | float | kWh | Total solar curtailment | Environment `info` |
| `soc_mean` | float | fraction | Mean battery state of charge (0-1) | Environment `info` |
| `soc_min` | float | fraction | Minimum battery SOC reached | Environment `info` |
| `soc_max` | float | fraction | Maximum battery SOC reached | Environment `info` |
| `fuel_mean` | float | fraction | Mean fuel tank level (0-1) | Environment `info` |
| `fuel_end` | float | fraction | Final fuel tank level | Environment `info` |
| `solar_total_kwh` | float | kWh | Total solar energy generated | Environment `info` |
| `battery_cycles` | float | count | Total battery charge/discharge cycles | Environment `info` |
| `blackout_hours` | float | hours | Total hours with unmet demand | Derived |
| `diesel_cost` | float | USD | Total diesel fuel cost | Derived |
| `co2_emissions_kg` | float | kg | Total CO2 emissions from diesel | Derived |
| `actor_loss` | float | dimensionless | Policy network loss | SB3 logger |
| `critic_loss` | float | dimensionless | Value network loss | SB3 logger |
| `entropy` | float | dimensionless | Policy entropy (exploration measure) | SB3 logger |
| `learning_rate` | float | dimensionless | Current learning rate | SB3 logger |
| `ent_coef` | float | dimensionless | Entropy coefficient (SAC only) | SB3 logger |
| `ent_coef_loss` | float | dimensionless | Entropy coefficient loss (SAC only) | SB3 logger |

### Optional Columns (Algorithm-Specific)
- **SAC**: `ent_coef`, `ent_coef_loss`, `actor_loss`, `critic_loss`
- **A2C**: `value_loss`, `policy_gradient_loss`, `explained_variance`
- **PPO**: `clip_fraction`, `approx_kl`, `policy_gradient_loss`, `value_loss`

### Missing Values
- Use empty string `""` or `NaN` for missing/unavailable metrics
- Algorithm-specific metrics should be empty for other algorithms

## Additional Files

### `config.yaml`
Copy of the configuration file used for training, including:
- Environment parameters (battery, diesel, solar, load)
- Reward function weights
- Algorithm hyperparameters
- Random seeds

### `summary.json`
Final summary statistics at end of training:
```json
{
  "run_id": "sac_nairobi_180d_0",
  "algorithm": "sac",
  "scenario": "nairobi_180d",
  "seed": 0,
  "total_timesteps": 200000,
  "training_time_seconds": 1523.45,
  "final_mean_reward": -125432.67,
  "final_unmet_kwh": 234.56,
  "final_diesel_liters": 1234.56,
  "model_path": "outputs/models/sac_nairobi_180d_0.zip"
}
```

## Units Reference

| Metric Type | Standard Unit | Notes |
|-------------|---------------|-------|
| Energy | kWh | Kilowatt-hours |
| Power | kW | Kilowatts |
| Fuel Volume | L | Liters |
| Time | hours | Hours (unless specified) |
| Money | USD | US Dollars |
| Mass | kg | Kilograms |
| SOC/Fuel Level | fraction | 0.0 to 1.0 |
| Temperature | °C | Celsius |
| Irradiance | kWh/m² | Per step duration |

## Usage Examples

### Reading Metrics in Python
```python
import pandas as pd
df = pd.read_csv('outputs/metrics/sac_nairobi_180d_0/metrics.csv')
print(df[['timestep', 'mean_reward', 'unmet_kwh']].head())
```

### Comparing Multiple Runs
```python
import pandas as pd
from pathlib import Path

runs = []
for run_dir in Path('outputs/metrics').glob('sac_*'):
    df = pd.read_csv(run_dir / 'metrics.csv')
    df['run_id'] = run_dir.name
    runs.append(df)
    
all_metrics = pd.concat(runs, ignore_index=True)
```

## Integration with External Tools

### Weights & Biases (W&B)
Optional W&B integration for cloud-based experiment tracking and visualization.

**Setup**

1. Install wandb:
```bash
pip install wandb
```

2. Login to W&B:
```bash
wandb login
```

**Usage**

**Basic W&B Logging**

Add `--wandb` flag to any training command:

```bash
python -m scripts.train_sac \
    --lat -1.2921 --lon 36.8219 \
    --days 180 \
    --scenario nairobi_180d \
    --seed 0 \
    --wandb
```

**Custom Project and Entity**

```bash
python -m scripts.train_sac \
    --lat -1.2921 --lon 36.8219 \
    --days 180 \
    --scenario nairobi_180d \
    --wandb \
    --wandb-project microgrid-project \
    --wandb-entity team-name
```

**What Gets Logged to W&B**

All metrics from the CSV schema are automatically logged to W&B:

- **Training metrics**: actor_loss, critic_loss, learning_rate, entropy
- **Episode metrics**: mean_reward, episode_length
- **Energy metrics**: unmet_kwh, diesel_liters, solar_total_kwh, curtailment_kwh
- **Economic metrics**: diesel_cost, co2_emissions_kg
- **Battery metrics**: soc_mean, soc_min, soc_max, battery_cycles

### CSV Export Only
Default mode - no external dependencies required.
