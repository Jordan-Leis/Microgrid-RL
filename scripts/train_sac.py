import argparse, os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.train_common import build_env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--days', type=int, default=180)
    ap.add_argument('--cfg', type=str, default='configs/default.yaml')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--total_timesteps', type=int, default=200_000)
    args = ap.parse_args()

    def make_env():
        env = build_env(args.cfg, args.lat, args.lon, args.days)
        return Monitor(env)

    env = DummyVecEnv([make_env])
    model = SAC('MlpPolicy', env, verbose=1, seed=args.seed)
    os.makedirs('outputs', exist_ok=True)
    model.learn(total_timesteps=args.total_timesteps)
    model.save('outputs/sac_microgrid.zip')
    print('Saved model → outputs/sac_microgrid.zip')

if __name__ == '__main__':
    main()
