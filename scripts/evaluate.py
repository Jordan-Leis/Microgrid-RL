import argparse, yaml, json
from pathlib import Path
from stable_baselines3 import SAC, A2C
from scripts.train_common import build_env
from scripts.metrics_logger import EpisodeMetricsAccumulator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--algo', type=str, choices=['sac','a2c'], required=True)
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--cfg', type=str, default='configs/default.yaml')
    ap.add_argument('--n_episodes', type=int, default=1, help='Number of evaluation episodes')
    ap.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    
    # Build environment
    env = build_env(args.cfg, args.lat, args.lon, args.days)
    
    # Load model
    model = SAC.load(args.model) if args.algo=='sac' else A2C.load(args.model)
    
    print(f"Evaluating {args.algo.upper()} model: {args.model}")
    print(f"Location: ({args.lat}, {args.lon}), Duration: {args.days} days")
    print(f"Running {args.n_episodes} episode(s)...\n")
    
    # Run evaluation episodes
    all_episodes = []
    
    for ep_num in range(args.n_episodes):
        accumulator = EpisodeMetricsAccumulator()
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            accumulator.update(reward, info)
        
        # Get episode metrics
        ep_metrics = accumulator.get_episode_metrics(
            diesel_cost_per_liter=cfg['diesel']['fuel_cost_per_liter'],
            co2_kg_per_liter=cfg['diesel']['co2_kg_per_liter'],
            step_hours=cfg['time']['step_hours'],
            battery_capacity_kwh=cfg['battery']['capacity_kwh']
        )
        
        all_episodes.append(ep_metrics)
        
        print(f"Episode {ep_num + 1}:")
        print(f"  Total Reward:        {ep_metrics['total_reward']:>12,.2f}")
        print(f"  Episode Length:      {ep_metrics['episode_length']:>12,} steps")
        print(f"  Unmet Demand:        {ep_metrics['unmet_kwh']:>12,.2f} kWh")
        print(f"  Diesel Used:         {ep_metrics['diesel_liters']:>12,.2f} L")
        print(f"  Diesel Cost:         ${ep_metrics['diesel_cost']:>11,.2f}")
        print(f"  CO2 Emissions:       {ep_metrics['co2_emissions_kg']:>12,.2f} kg")
        print(f"  Solar Generated:     {ep_metrics['solar_total_kwh']:>12,.2f} kWh")
        print(f"  Solar Curtailed:     {ep_metrics['curtailment_kwh']:>12,.2f} kWh")
        print(f"  Battery Cycles:      {ep_metrics.get('battery_cycles', 0):>12,.2f}")
        print(f"  Blackout Hours:      {ep_metrics['blackout_hours']:>12,.2f} h")
        print(f"  Mean SOC:            {ep_metrics.get('soc_mean', 'N/A'):>12}")
        print()
    
    # Compute aggregate statistics
    import numpy as np
    
    summary = {
        'model_path': args.model,
        'algorithm': args.algo,
        'n_episodes': args.n_episodes,
        'config': args.cfg,
        'location': {'lat': args.lat, 'lon': args.lon},
        'days': args.days,
    }
    
    # Aggregate metrics
    for key in ['total_reward', 'unmet_kwh', 'diesel_liters', 'diesel_cost', 
                'co2_emissions_kg', 'solar_total_kwh', 'curtailment_kwh',
                'blackout_hours', 'battery_cycles']:
        values = [ep[key] for ep in all_episodes if ep.get(key, '') != '']
        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_min'] = float(np.min(values))
            summary[f'{key}_max'] = float(np.max(values))
    
    # Print summary
    print("=" * 60)
    print("SUMMARY (across all episodes):")
    print("=" * 60)
    print(f"Mean Reward:         {summary.get('total_reward_mean', 0):>12,.2f} ± {summary.get('total_reward_std', 0):.2f}")
    print(f"Mean Unmet Demand:   {summary.get('unmet_kwh_mean', 0):>12,.2f} ± {summary.get('unmet_kwh_std', 0):.2f} kWh")
    print(f"Mean Diesel Used:    {summary.get('diesel_liters_mean', 0):>12,.2f} ± {summary.get('diesel_liters_std', 0):.2f} L")
    print(f"Mean Diesel Cost:    ${summary.get('diesel_cost_mean', 0):>11,.2f} ± ${summary.get('diesel_cost_std', 0):.2f}")
    print(f"Mean CO2 Emissions:  {summary.get('co2_emissions_kg_mean', 0):>12,.2f} ± {summary.get('co2_emissions_kg_std', 0):.2f} kg")
    print("=" * 60)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'summary': summary,
                'episodes': all_episodes
            }, f, indent=2)
        
        print(f"\nEvaluation results saved to: {output_path}")

if __name__ == '__main__':
    main()
