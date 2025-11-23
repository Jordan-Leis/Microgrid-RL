# Microgrid-RL (Gymnasium + SAC/A2C)

Reinforcement learning for a solar + battery + diesel microgrid. Uses Gymnasium + Stable-Baselines3. NASA POWER provides hourly irradiance/temperature.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip && pip install -r requirements.txt
python scripts/download_nasa_power.py --lat -1.2921 --lon 36.8219 --start 2024-01-01 --end 2024-12-31 --out data/raw/nairobi_2024.csv
python scripts/train_sac.py --lat -1.2921 --lon 36.8219 --days 180
```
## Contributors
- Cilo Zhou
- Taha Subzwari
- Siri Sujay
