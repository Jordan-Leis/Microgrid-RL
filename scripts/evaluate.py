import argparse
from stable_baselines3 import SAC, A2C
from scripts.train_common import build_env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--algo', type=str, choices=['sac','a2c'], required=True)
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--cfg', type=str, default='configs/default.yaml')
    args = ap.parse_args()

    env = build_env(args.cfg, args.lat, args.lon, args.days)
    model = SAC.load(args.model) if args.algo=='sac' else A2C.load(args.model)

    obs, _ = env.reset()
    done = False
    reward = 0.0
    unmet = 0.0
    liters = 0.0
    refuel_cost = 0.0
    refueled_liters = 0.0
    total_maintenance_cost = 0.0
    
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(act)
        reward += r
        unmet += info['unmet_kwh']
        liters += info['liters_used']
        refuel_cost += info.get('refuel_cost', 0.0)
        refueled_liters += info.get('refueled_liters', 0.0)
        total_maintenance_cost += info.get('maintenance_cost', 0.0)
    
    # Calculate weekly rates
    episode_days = args.days
    weeks = episode_days / 7.0
    liters_per_week = liters / weeks if weeks > 0 else 0
    
    print(f'Episode reward: {reward:.2f}')
    print(f'Unmet demand (kWh): {unmet:.2f}')
    print(f'Diesel used (L): {liters:.2f}')
    print(f'Refuel cost: ${refuel_cost:.2f}')
    print(f'Refueled liters: {refueled_liters:.2f}')
    print(f'Total maintenance cost: ${total_maintenance_cost:.2f}')
    print(f'--- Weekly Rates ---')
    print(f'Liters per week: {liters_per_week:.1f}')
    print(f'Maintenance cost per week: ${total_maintenance_cost/weeks:.2f}')

if __name__ == '__main__':
    main()
