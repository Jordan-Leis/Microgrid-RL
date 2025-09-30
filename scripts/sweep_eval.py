import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, A2C
from scripts.train_common import build_env
from stable_baselines3.common.vec_env import DummyVecEnv

# sweep configurations
algorithms = ['sac', 'a2c']
seeds = [0, 1, 2]
weather_years = [2022, 2023, 2024]
load_scenarios = ['base', 'high', 'low']
n_eval_episodes = 5  

results = []

# sweep loop
for algo in algorithms:
    for seed in seeds:
        for year in weather_years:
            for load in load_scenarios:
                print(f'Running {algo} | seed {seed} | year {year} | load {load}')

                def make_env():
                    return build_env('configs/default.yaml', lat=-1.2921, lon=36.8219, days=30)
                env = DummyVecEnv([make_env])

                model_path = f'outputs/{algo}_microgrid'
                if algo == 'sac':
                    model = SAC.load(model_path, env=env, seed=seed)
                else:
                    model = A2C.load(model_path, env=env, seed=seed)

                rewards, unmet_kwh, diesel, curtailment, cycles = [], [], [], [], []

                for _ in range(n_eval_episodes):
                    obs = env.reset()
                    done = False
                    ep_reward, ep_unmet, ep_diesel, ep_curtail, ep_cycles = 0, 0, 0, 0, 0

                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, r, done, info = env.step(action)
                        r = r[0]
                        done = done[0]
                        info = info[0]
                        ep_reward += r
                        ep_unmet += info.get('unmet_kwh', 0)
                        ep_diesel += info.get('diesel_liters', 0)
                        ep_curtail += info.get('curtailment', 0)
                        ep_cycles += info.get('cycles', 0)

                    rewards.append(ep_reward)
                    unmet_kwh.append(ep_unmet)
                    diesel.append(ep_diesel)
                    curtailment.append(ep_curtail)
                    cycles.append(ep_cycles)

                results.append({
                    'algorithm': algo,
                    'seed': seed,
                    'year': year,
                    'load_scenario': load,
                    'reward_mean': np.mean(rewards),
                    'reward_std': np.std(rewards),
                    'unmet_kwh_mean': np.mean(unmet_kwh),
                    'unmet_kwh_std': np.std(unmet_kwh),
                    'diesel_liters_mean': np.mean(diesel),
                    'diesel_liters_std': np.std(diesel),
                    'curtailment_mean': np.mean(curtailment),
                    'curtailment_std': np.std(curtailment),
                    'cycles_mean': np.mean(cycles),
                    'cycles_std': np.std(cycles),
                })

os.makedirs('results', exist_ok=True)
summary_csv = 'results/metrics_summary.csv'
pd.DataFrame(results).to_csv(summary_csv, index=False)
print(f'Saved summary → {summary_csv}')



os.makedirs('results/plots', exist_ok=True)
metrics_to_plot = ['reward_mean', 'unmet_kwh_mean', 'diesel_liters_mean']

for metric in metrics_to_plot:
    plt.figure()
    for algo in algorithms:
        subset = pd.DataFrame(results)[pd.DataFrame(results)['algorithm']==algo]
        plt.plot(subset['seed'], subset[metric], marker='o', label=algo)
    plt.xlabel('Seed')
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.savefig(f'results/plots/{metric}.png')
    plt.close()
