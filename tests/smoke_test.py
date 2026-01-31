import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest


proj_root = Path(__file__).resolve().parents[1]


algo_models = {
    "sac": "tests/dummy_sac.zip",
    "a2c": "tests/dummy_a2c.zip",
    "ppo": "tests/dummy_ppo.zip",
    "td3": "tests/dummy_td3.zip",
    "ddpg": "tests/dummy_ddpg.zip",
    "tqc": "tests/dummy_tqc.zip",
    "rppo": "tests/dummy_rppo.zip",
}


@pytest.mark.parametrize("algo,model_path", algo_models.items())
def test_evaluate_smoke(tmp_path, algo, model_path):
    model_path = proj_root/model_path
    if not model_path.exists():
        pytest.skip(f"Dummy model for {algo} not found")
    
    out_dir = tmp_path / f"eval_{algo}"
    out_dir.mkdir()

    cmd = [
        sys.executable, "-m", "scripts.evaluate",
        "--model", str(model_path),
        "--algo", algo,
        "--lat", "0.0",
        "--lon", "0.0",
        "--days", "1",
        "--episodes", "2",
        "--seed", "0",
        "--out-dir", str(out_dir),
        "--deterministic"
    ]

    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        cwd=proj_root)
    
    
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0, (f"{algo} evaluate.py crashed:\n{result.stderr}")

    # Check that episodes.csv and summary.json exist
    episodes_csv = out_dir / "episodes.csv"
    summary_json = out_dir / "summary.json"
    assert episodes_csv.exists(), "episodes.csv not found"
    assert summary_json.exists(), "summary.json not found"

    # Checks that CSV has expected columns
    df = pd.read_csv(episodes_csv)
    expected_cols = {"episode", "return", "unmet_kwh", "litres_used", "length"}
    assert expected_cols.issubset(df.columns), f"CSV missing columns: {df.columns}"