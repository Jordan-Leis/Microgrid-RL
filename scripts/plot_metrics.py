import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(log_path, save_dir=None):
    df = pd.read_csv(log_path)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Check if required metrics (reward and fuel) exist in CSV file
    required = ["reward", "fuel"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {log_path}")
        
    # Determine x-axis (timesteps)
    if "step" in df.columns: # if there is a step column
        x = df["step"]
        x_title = "step"
    else:
        x = range(len(df)) # take number of rows
        x_title = "Timesteps"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot rewards vs. timesteps and fuel vs. timesteps
    for y_col in required:
        plt.figure(figsize=(8, 6))
        plt.plot(x, df[y_col], linewidth=2)
        plt.xlabel(x_title)
        plt.ylabel(y_col)
        plt.title(f"{y_col.capitalize()} vs {x_title}")
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_dir:
            save_path = os.path.join(save_dir, f'{y_col}_vs_{x_title}.png')
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot energy flows over time")
    parser.add_argument("log_path", type=str, help="Path to CSV log file")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional directory to save plots")
    args = parser.parse_args()

    plot_metrics(args.log_path, args.save_dir)