"""Aggregate evaluation seeds and create one representative plot per scenario."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evaluation/baseline_q2")
    args = parser.parse_args()
    root = Path(args.root)
    report_dir = root / "report"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    selections = []
    for scenario_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "report"):
        scenario_metrics = []
        for seed_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            metrics_path = seed_dir / "metrics_by_run.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                metrics["source_dir"] = str(seed_dir)
                scenario_metrics.append(metrics)
                all_metrics.append(metrics)
        if not scenario_metrics:
            continue

        metrics = pd.concat(scenario_metrics, ignore_index=True)
        median_rmse = metrics["angle_rmse_rad"].median()
        representative = metrics.iloc[(metrics["angle_rmse_rad"] - median_rmse).abs().argmin()]
        source = Path(representative["source_dir"])
        env_id = int(representative["env_id"])
        trajectory = pd.read_csv(source / "trajectories.csv")
        trajectory = trajectory[trajectory.env_id == env_id].copy()
        selections.append({
            "scenario": scenario_dir.name,
            "seed": int(representative["seed"]),
            "env_id": env_id,
            "angle_rmse_rad": representative["angle_rmse_rad"],
            "source": str(source / "trajectories.csv"),
        })

        fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
        time = trajectory["time_s"]
        axes[0].plot(time, np.rad2deg(trajectory["roll_rad"]), label="roll")
        axes[0].plot(time, np.rad2deg(trajectory["pitch_rad"]), label="pitch")
        axes[0].axhline(5.0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].axhline(-5.0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Angle (deg)")
        axes[0].legend(ncol=2)
        axes[0].grid(alpha=0.3)

        axes[1].plot(time, trajectory["cmd_vx"], "k--", label="command vx")
        axes[1].plot(time, trajectory["vx"], label="actual vx")
        axes[1].plot(time, trajectory["cmd_yaw_rate"], "--", label="command yaw")
        axes[1].plot(time, trajectory["yaw_rate"], label="actual yaw")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Velocity")
        axes[1].legend(ncol=2, fontsize=8)
        axes[1].grid(alpha=0.3)

        if scenario_dir.name.startswith("push_"):
            for axis in axes:
                axis.axvline(4.0, color="red", linestyle=":", label="push")
        fig.suptitle(
            "{} representative run (seed {}, env {})".format(
                scenario_dir.name, int(representative["seed"]), env_id
            )
        )
        fig.tight_layout()
        fig.savefig(figure_dir / (scenario_dir.name + ".png"), dpi=180)
        plt.close(fig)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(report_dir / "all_metrics_by_run.csv", index=False)
    pd.DataFrame(selections).to_csv(report_dir / "representative_runs.csv", index=False)

    rows = []
    for scenario, data in combined.groupby("scenario", sort=True):
        row = {"scenario": scenario, "n": len(data)}
        for column in [
            "completed_duration", "survival_time_s", "angle_rmse_rad",
            "angular_rate_rmse", "vx_rmse", "yaw_rate_rmse",
            "max_abs_angle_rad", "min_height_m",
            "recovered", "recovery_time_s",
        ]:
            values = pd.to_numeric(data[column], errors="coerce")
            row[column + "_mean"] = values.mean()
            row[column + "_std"] = values.std(ddof=0)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(report_dir / "summary_all_seeds.csv", index=False)
    print(summary.to_string(index=False))
    print("Representative figures:", figure_dir)


if __name__ == "__main__":
    main()
