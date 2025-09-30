import argparse, os
import yaml
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
    ap.add_argument('--total_timesteps', type=int, default=500_000)
    ap.add_argument('--save-path', type=str, default='outputs')
    ap.add_argument('--csv-log', type=str, default='outputs/training.csv')
    ap.add_argument('--vectorized', action='store_true')
    ap.add_argument('--hyperparams', type=str, default='configs/hyperparams.yaml')
    args = ap.parse_args()

    # load and use sac hyperparameters
    hp = yaml.safe_load(open(args.hyperparams))
    lr = hp['sac']['learning_rate']
    gamma = hp['sac']['gamma']
    batch_size = hp['sac']['batch_size']

    def make_env():
        env = build_env(args.cfg, args.lat, args.lon, args.days)
        env = Monitor(env, filename=args.csv_log)  # for CSV logging
        return env

    env = DummyVecEnv([make_env]) if args.vectorized else make_env()
    model = SAC('MlpPolicy', env, verbose=1, seed=args.seed,
            learning_rate=lr, gamma=gamma, batch_size=batch_size)
    
    os.makedirs(args.save_path, exist_ok=True)
    model_file = os.path.join(args.save_path, 'sac_microgrid.zip') 
    model.learn(total_timesteps=args.total_timesteps)
    model.save(model_file)
    print(f'Saved model → {model_file}')

if __name__ == '__main__':
    main()
