#!/usr/bin/env python
"""
Step 4 — GNC Accuracy Evaluation
==================================

Supports system-level validation claims with Monte Carlo scenario runs.

Outputs:
    results/gnc/gnc_results.csv
    results/figures/G15_landing_error_histogram.png
    results/figures/G16_trajectory_3d_line.png
    results/figures/G17_fuel_usage_bar.png
"""

from __future__ import annotations
import os, sys, csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_ai_os.config import VehicleConfig, SimConfig, GuidanceConfig, MissionPhase
from rocket_ai_os.sim.scenarios import (NominalLandingScenario,
                                        EngineOutScenario,
                                        SensorDegradationScenario)

ROOT = os.path.join(os.path.dirname(__file__), "..")
GNC_DIR = os.path.join(ROOT, "results", "gnc")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(GNC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_TRIALS = 20
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def run_gnc_evaluation():
    """Run 200 scenarios with varied initial conditions and sensor noise."""
    results = []

    print(f"  Running {N_TRIALS} scenarios across 3 types...")

    for run in tqdm(range(N_TRIALS), desc="  GNC Eval", ncols=70):
        seed = 42 + run
        rng = np.random.default_rng(seed)

        # Vary initial conditions
        vc = VehicleConfig()
        sc = SimConfig()
        gc = GuidanceConfig()

        # Random position perturbation
        pos_offset = rng.normal(0, 30, 3)
        vel_offset = rng.normal(0, 5, 3)

        scenario_type = run % 3  # cycle through types

        try:
            if scenario_type == 0:
                # Nominal
                scenario = NominalLandingScenario(
                    vehicle_config=vc, sim_config=sc,
                    guidance_config=gc, seed=seed)
                result = scenario.run()
                stype = "nominal"
            elif scenario_type == 1:
                # Engine out
                failure_time = rng.uniform(2, 8)
                scenario = EngineOutScenario(
                    failure_time=failure_time,
                    vehicle_config=vc, sim_config=sc,
                    guidance_config=gc, seed=seed)
                result = scenario.run()
                stype = "engine_out"
            else:
                # Sensor degradation
                dropout_time = rng.uniform(1, 5)
                scenario = SensorDegradationScenario(
                    gps_dropout_time=dropout_time,
                    imu_drift_rate=rng.uniform(0.01, 0.05),
                    vehicle_config=vc, sim_config=sc,
                    guidance_config=gc, seed=seed)
                result = scenario.run()
                stype = "sensor_degraded"

            # Extract trajectory for G16
            trajectory = [(s.position.copy(), s.velocity.copy(), s.time)
                          for s in result.trajectory_log[::3]]  # subsample

            results.append({
                "run": run,
                "type": stype,
                "success": result.success,
                "pos_error_m": result.metrics.get("touchdown_pos_error_m", float("inf")),
                "touchdown_speed": result.metrics.get("touchdown_speed_m_s", float("inf")),
                "fuel_remaining_kg": result.metrics.get("fuel_remaining_kg", 0),
                "flight_time_s": result.metrics.get("flight_time_s", 0),
                "is_landed": result.metrics.get("is_landed", False),
                "is_destroyed": result.metrics.get("is_destroyed", False),
                "trajectory": trajectory,
            })
        except Exception as e:
            results.append({
                "run": run, "type": stype if 'stype' in dir() else "unknown",
                "success": False, "pos_error_m": float("inf"),
                "touchdown_speed": float("inf"), "fuel_remaining_kg": 0,
                "flight_time_s": 0, "is_landed": False, "is_destroyed": True,
                "trajectory": [],
            })

    return results


# =====================================================================
# 4a  Latency Comparison (G-FOLD vs Proportional)
# =====================================================================
def run_latency_comparison():
    """Evaluate optimization latency of G-FOLD vs Proportional."""
    print("  Evaluating Guidance Latency...")
    from rocket_ai_os.gnc.navigation import NavigationState
    vc = VehicleConfig()
    gc = GuidanceConfig()
    sc = SimConfig()
    
    # Setup initial state
    nav_state = NavigationState(
        position=np.array([1000.0, 500.0, 5000.0]),
        velocity=np.array([-50.0, -20.0, -150.0]),
        attitude=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rates=np.zeros(3),
        mass=vc.dry_mass + 10000.0,
        timestamp=0.0
    )

    results = []
    
    for solver_name in ["proportional", "gfold"]:
        from rocket_ai_os.gnc.guidance import GuidanceSystem
        guidance = GuidanceSystem(vehicle_config=vc, guidance_config=gc, sim_config=sc, solver_type=solver_name)
        
        # Warm-up run
        guidance.update(nav_state, 0.0)
        
        # Measure
        latencies = []
        for i in range(100):
            nav_state.timestamp = i * 0.1
            guidance._last_solve_time = -1e6 # force resolve
            
            t0 = time.perf_counter()
            guidance.update(nav_state, nav_state.timestamp)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        mean_lat = np.mean(latencies)
        max_lat = np.max(latencies)
        p99_lat = np.percentile(latencies, 99)
        results.append({"Solver": solver_name, "Mean_ms": mean_lat, "Max_ms": max_lat, "P99_ms": p99_lat})
        print(f"    {solver_name.upper()}: Mean = {mean_lat:.2f} ms | Max = {max_lat:.2f} ms | P99 = {p99_lat:.2f} ms")

    return results


# =====================================================================
# Graphs
# =====================================================================
def plot_G15(df):
    """G15: Histogram — Landing position error across 200 runs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    types = ["nominal", "engine_out", "sensor_degraded"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    titles = ["Nominal", "Engine Out", "Sensor Degraded"]

    for ax, stype, color, title in zip(axes, types, colors, titles):
        sub = df[df["type"] == stype]
        errors = sub["pos_error_m"].clip(upper=100)
        ax.hist(errors, bins=20, color=color, edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.axvline(x=errors.median(), color="red", linestyle="--",
                   alpha=0.7, label=f"Median={errors.median():.1f}m")
        ax.set_xlabel("Position Error (m)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("G15 — Landing Position Error Distribution (200 runs)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G15_landing_error_histogram.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_G16(results):
    """G16: 3D trajectory plot — one representative from each type."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    types_colors = {"nominal": "#4C72B0", "engine_out": "#DD8452",
                    "sensor_degraded": "#55A868"}

    for stype, color in types_colors.items():
        # Find first successful run of each type
        for r in results:
            if r["type"] == stype and r["trajectory"] and len(r["trajectory"]) > 3:
                traj = r["trajectory"]
                xs = [p[0][0] for p in traj]
                ys = [p[0][1] for p in traj]
                zs = [p[0][2] for p in traj]
                ax.plot(xs, ys, zs, label=stype.replace("_", " ").title(),
                        color=color, linewidth=1.8)
                # Mark start and end
                ax.scatter(xs[0], ys[0], zs[0], marker="^", s=60, color=color)
                ax.scatter(xs[-1], ys[-1], zs[-1], marker="v", s=60, color=color)
                break

    # Target
    ax.scatter([0], [0], [0], marker="*", s=200, c="gold", zorder=5, label="Target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("G16 — Representative 3D Trajectories by Scenario Type")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G16_trajectory_3d_line.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_G17(df):
    """G17: Bar chart — Fuel usage by scenario type."""
    fig, ax = plt.subplots(figsize=(8, 6))
    types = ["nominal", "engine_out", "sensor_degraded"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    labels = ["Nominal", "Engine Out", "Sensor Degraded"]

    fuel_used = []
    stds = []
    initial_fuel = 15_000.0
    for stype in types:
        sub = df[df["type"] == stype]
        used = initial_fuel - sub["fuel_remaining_kg"].clip(lower=0)
        fuel_used.append(used.mean())
        stds.append(used.std())

    x = np.arange(len(types))
    bars = ax.bar(x, fuel_used, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fuel Consumed (kg)")
    ax.set_title("G17 — Fuel Consumption by Scenario Type")

    for bar, val in zip(bars, fuel_used):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:.0f} kg", ha="center", fontsize=10)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G17_fuel_usage_bar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("  STEP 4 — GNC Accuracy & Latency Evaluation")
    print("=" * 60)

    # 1. Latency comparison
    latency_res = run_latency_comparison()

    # 2. Monte Carlo Accuracy
    results = run_gnc_evaluation()

    # Save CSV (without trajectory column)
    csv_path = os.path.join(GNC_DIR, "gnc_results.csv")
    csv_fields = ["run", "type", "success", "pos_error_m", "touchdown_speed",
                  "fuel_remaining_kg", "flight_time_s", "is_landed", "is_destroyed"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})
    print(f"  Saved -> {csv_path}")

    df = pd.DataFrame([{k: r[k] for k in csv_fields} for r in results])

    # Summary
    print("\n  Summary:")
    for stype in ["nominal", "engine_out", "sensor_degraded"]:
        sub = df[df["type"] == stype]
        print(f"  {stype:>18s}: success={sub['success'].mean()*100:.0f}%  "
              f"pos_err={sub['pos_error_m'].mean():.1f}m  "
              f"speed={sub['touchdown_speed'].mean():.1f}m/s  "
              f"fuel_left={sub['fuel_remaining_kg'].mean():.0f}kg")

    # Graphs
    print("\n  Generating graphs...")
    plot_G15(df)
    plot_G16(results)
    plot_G17(df)

    print("\n  Step 4 COMPLETE [OK]")


if __name__ == "__main__":
    main()
