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
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(act)
        reward += r; unmet += info['unmet_kwh']; liters += info['liters_used']
    print(f'Episode reward: {reward:.2f}\nUnmet demand (kWh): {unmet:.2f}\nDiesel used (L): {liters:.2f}')

if __name__ == '__main__':
    main()
