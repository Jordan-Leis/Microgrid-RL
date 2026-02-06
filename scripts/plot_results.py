"""
Standardized plotting for survey paper evaluation outputs.

Reads:
  runs/<algo>/seed_<seed>/episodes.csv
  runs/<algo>/seed_<seed>/summary.json

Writes:
  runs/_plots/<algo>_returns.png
  runs/_plots/kpi_summary.csv
  runs/_plots/kpi_bars.png

No notebooks, runnable from repo root.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _try_load_yaml(path: Path) -> dict:
    """Load YAML if PyYAML exists; otherwise error with a helpful message."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read configs/eval.yaml. "
            "Install with: pip install pyyaml"
        ) from e
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_get(d: dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def find_algo_seed_dirs(runs_root: Path) -> List[Tuple[str, List[Path]]]:
    algos = []
    for algo_dir in sorted([p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        seed_dirs = sorted([p for p in algo_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        if seed_dirs:
            algos.append((algo_dir.name, seed_dirs))
    return algos


def load_summary(summary_json: Path) -> Optional[Dict[str, Dict[str, float]]]:
    if not summary_json.exists():
        return None
    try:
        return json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception:
        return None


def aggregate_kpis_across_seeds(
    seed_summaries: List[Dict[str, Dict[str, float]]],
    kpi_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Each summary.json expected like:
      {"return": {"mean": ..., "std": ...}, "other_kpi": {"mean": ..., "std": ...}}

    Aggregation:
      algo_mean = mean(seed_means)
      algo_std  = std(seed_means)  (std across seeds, not within-episode std)
    """
    if not seed_summaries:
        return pd.DataFrame()

    if not kpi_keys:
        kpi_keys = sorted({k for s in seed_summaries for k in s.keys()})

    rows = []
    for kpi in kpi_keys:
        seed_means = []
        for s in seed_summaries:
            if kpi in s and isinstance(s[kpi], dict) and "mean" in s[kpi]:
                try:
                    seed_means.append(float(s[kpi]["mean"]))
                except Exception:
                    pass

        if not seed_means:
            continue

        ser = pd.Series(seed_means, dtype=float)
        rows.append(
            {
                "kpi": kpi,
                "algo_mean": float(ser.mean()),
                "algo_std": float(ser.std(ddof=0)),
                "n_seeds": int(len(seed_means)),
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/eval.yaml")
    args = ap.parse_args()

    cfg = _try_load_yaml(Path(args.config))

    runs_root = Path(_deep_get(cfg, ["paths", "runs_root"], "runs"))
    out_dir = Path(_deep_get(cfg, ["paths", "out_dir"], "runs/_plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv_name = _deep_get(cfg, ["schema", "episodes_csv"], "episodes.csv")
    summary_json_name = _deep_get(cfg, ["schema", "summary_json"], "summary.json")

    episode_col = _deep_get(cfg, ["schema", "episode_col"], "episode")
    returns_col = _deep_get(cfg, ["schema", "returns_col"], "return")

    smooth_window = int(_deep_get(cfg, ["plotting", "smooth_window"], 20))
    dpi = int(_deep_get(cfg, ["plotting", "dpi"], 200))
    x_is_index = bool(_deep_get(cfg, ["plotting", "x_is_index"], True))
    max_legend_seeds = int(_deep_get(cfg, ["plotting", "max_legend_seeds"], 12))

    kpi_keys = _deep_get(cfg, ["kpis", "keys"], None)
    if isinstance(kpi_keys, list) and len(kpi_keys) == 0:
        kpi_keys = None  # auto-discover

    if not runs_root.exists():
        raise FileNotFoundError(
            f"runs_root '{runs_root}' not found.\n"
            f"Expected structure: runs/<algo>/seed_<seed>/{episodes_csv_name},{summary_json_name}\n"
            f"Tip: even if your training outputs live in outputs/, standardize eval outputs into runs/."
        )

    algo_seed_dirs = find_algo_seed_dirs(runs_root)
    if not algo_seed_dirs:
        print(f"[warn] No algo/seed dirs found under {runs_root}")
        return

    # ---------- (A) Learning curves ----------
    for algo, seed_dirs in algo_seed_dirs:
        plt.figure()
        plotted = 0

        for sd in seed_dirs:
            ep_path = sd / episodes_csv_name
            if not ep_path.exists():
                continue

            df = pd.read_csv(ep_path)

            if returns_col not in df.columns:
                raise ValueError(
                    f"{ep_path} missing '{returns_col}'. Columns found: {list(df.columns)}"
                )

            y = smooth(df[returns_col].astype(float), smooth_window)

            if x_is_index:
                plt.plot(y.values, label=sd.name)
            else:
                if episode_col not in df.columns:
                    raise ValueError(
                        f"{ep_path} missing '{episode_col}' for x-axis. Columns found: {list(df.columns)}"
                    )
                plt.plot(df[episode_col].values, y.values, label=sd.name)

            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.title(f"{algo} - episode returns (smoothed window={smooth_window})")
        plt.xlabel("Episode")
        plt.ylabel("Return")

        if plotted <= max_legend_seeds:
            plt.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(out_dir / f"{algo}_returns.png", dpi=dpi)
        plt.close()

    # ---------- (B) KPI summary + bar chart ----------
    all_algo_rows = []
    for algo, seed_dirs in algo_seed_dirs:
        seed_summaries = []
        for sd in seed_dirs:
            s = load_summary(sd / summary_json_name)
            if s:
                seed_summaries.append(s)

        df_algo = aggregate_kpis_across_seeds(seed_summaries, kpi_keys=kpi_keys)
        if df_algo.empty:
            continue

        df_algo.insert(0, "algorithm", algo)
        all_algo_rows.append(df_algo)

    if not all_algo_rows:
        print("[warn] No summary.json found anywhere; skipping KPI summary + bars.")
        return

    df_all = pd.concat(all_algo_rows, ignore_index=True)
    df_all.to_csv(out_dir / "kpi_summary.csv", index=False)

    kpis = sorted(df_all["kpi"].unique().tolist())
    algos = sorted(df_all["algorithm"].unique().tolist())

    pivot_mean = df_all.pivot(index="algorithm", columns="kpi", values="algo_mean").reindex(algos)
    pivot_std = df_all.pivot(index="algorithm", columns="kpi", values="algo_std").reindex(algos)

    x = list(range(len(algos)))
    width = 0.8 / max(1, len(kpis))

    plt.figure(figsize=(max(6, len(algos) * 1.1), 4.5))
    for i, kpi in enumerate(kpis):
        means = pivot_mean[kpi].values if kpi in pivot_mean.columns else [float("nan")] * len(algos)
        errs = pivot_std[kpi].values if kpi in pivot_std.columns else [0.0] * len(algos)

        offsets = [xi - 0.4 + (i + 0.5) * width for xi in x]
        plt.bar(offsets, means, width=width, yerr=errs, capsize=3, label=kpi)

    plt.xticks(x, algos, rotation=20, ha="right")
    plt.ylabel("KPI (mean ± std across seeds)")
    plt.title("KPI comparison across algorithms")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "kpi_bars.png", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    main()
