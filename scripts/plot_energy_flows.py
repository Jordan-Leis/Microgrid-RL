import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_energy_flows(log_path, save_path=None):
    """
    Plots time-series of load, PV, battery, and diesel energy flows (kWh).

    Expected CSV columns:
        step, e_load, e_solar, e_batt, e_diesel, soc, fuel
    """
    df = pd.read_csv(log_path)

    required = ["e_load", "e_solar", "e_batt", "e_diesel"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in {log_path}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df.index, df["e_load"], label="Load", linewidth=2)
    ax1.plot(df.index, df["e_solar"], label="PV Generation", linewidth=2)
    ax1.plot(df.index, df["e_batt"], label="Battery (+discharge, -charge)", linewidth=2)
    ax1.plot(df.index, df["e_diesel"], label="Diesel", linewidth=2)

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Microgrid Energy Flows Over Time")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.7)

    if "soc" in df.columns or "fuel" in df.columns:
        ax2 = ax1.twinx()
        if "soc" in df.columns:
            ax2.plot(df.index, df["soc"], "g--", alpha=0.6, label="SOC")
        if "fuel" in df.columns:
            ax2.plot(df.index, df["fuel"], "r--", alpha=0.6, label="Fuel Level")
        ax2.set_ylabel("State [0–1]")
        ax2.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot energy flow time-series from log CSV.")
    parser.add_argument("--log", type=str, required=True, help="Path to log CSV file (e.g., outputs/energy_log.csv)")
    parser.add_argument("--save", type=str, help="Optional path to save the plot image")
    args = parser.parse_args()

    plot_energy_flows(args.log, args.save)
