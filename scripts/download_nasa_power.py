import argparse, os
from data_sources.nasa_power import fetch_power_hourly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--out', type=str, default='data/raw/nasa.csv')
    args = ap.parse_args()
    df = fetch_power_hourly(args.lat, args.lon, args.start, args.end)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Wrote {len(df)} rows to {args.out}')

if __name__ == '__main__':
    main()
