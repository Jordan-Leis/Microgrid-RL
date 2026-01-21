## Recurrent PPO (LSTM/GRU) — Results and Reproducibility

This document captures settings, commands, and high-level findings for Recurrent PPO (RPPO) on the microgrid task, with a direct comparison to feedforward PPO (FF PPO).

### Config and Hyperparameters
- Config file: `configs/rppo.yaml` (matches environment parameters used by A2C/SAC baselines).
- Default RPPO hyperparameters (can be overridden via CLI):
  - rnn_type: lstm
  - n_steps: 256
  - batch_size: 64
  - learning_rate: 3e-4
  - gamma: 0.99
  - gae_lambda: 0.95
  - clip_range: 0.2
  - ent_coef: 0.0
  - vf_coef: 0.5
  - max_grad_norm: 0.5
  - hidden_size: 128
  - n_layers: 1

### Training
Recurrent PPO (default LSTM):

```bash
python scripts/train_rppo.py \
  --lat -1.2921 --lon 36.8219 --days 180 \
  --cfg configs/rppo.yaml \
  --seed 0 \
  --total_timesteps 500000
```

Feedforward PPO (comparison; uses the same script with `--ff`):

```bash
python scripts/train_rppo.py \
  --lat -1.2921 --lon 36.8219 --days 180 \
  --cfg configs/rppo.yaml \
  --seed 0 \
  --total_timesteps 500000 \
  --ff
```

Recommended for development: run 5 seeds per algorithm (0..4). For final plots: 10 seeds.

All runs write:
- Checkpoints: `outputs/models/{run_id}.zip`
- Structured logs: `outputs/metrics/{run_id}/metrics.csv`
- Config snapshot and summary: `outputs/metrics/{run_id}/config.yaml`, `summary.json`

`run_id` uses the schema `{algo}_{scenario}_{seed}`, where `algo` is `rppo` or `ppo`.

### Evaluation and KPIs
Use the shared logging wrapper outputs to compute KPIs. The CSV contains rolling episode metrics plus derived measures. Minimum set to report (mean ± std across seeds):
- Episode return
- Diesel liters and diesel cost
- Battery throughput (kWh) and equivalent full cycles
- Curtailment and/or unmet load
- Generator starts and total operating hours

You can parse `outputs/metrics/*/metrics.csv` and `summary.json` to aggregate across seeds.

### Notes on Temporal Effects
- RPPO is expected to better capture diurnal patterns and slow dynamics.
- Inspect cycling smoothness (battery `soc` trends, `battery_cycles`) and generator toggling (`genset_on`, starts via penalties).
- Compare RPPO vs FF PPO on:
  - Unmet load and curtailment behavior across day/night cycles.
  - Diesel on/off frequency and operating hours.
  - SOC trajectories smoothness and equivalent full cycles.

### Reproducibility
- Environment dynamics and reward are unchanged from baselines.
- Scenario parameters match A2C/SAC setup.
- Seeds and total training timesteps are controlled via CLI; ensure identical budgets across algorithms for fair comparison.

