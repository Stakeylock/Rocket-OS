#!/usr/bin/env python
"""
Step 1 — Architecture Latency Benchmarking
===========================================

Proves: The layered ARINC 653 partitioned stack is not significantly slower
than a flat monolithic approach, and provides deterministic timing.

Outputs:
    results/latency/timings_partitioned.csv
    results/latency/timings_monolithic.csv
    results/figures/G01_subsystem_latency_bar.png
    results/figures/G02_latency_distribution_box.png
    results/figures/G03_cumulative_latency_line.png
    results/figures/G04_partition_utilization_heatmap.png
"""

from __future__ import annotations
import os, sys, time, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_ai_os.config import SystemConfig, MissionPhase
from rocket_ai_os.core.rtos import PartitionedRTOS
from rocket_ai_os.core.software_bus import SoftwareBus, MessagePriority
from rocket_ai_os.gnc.navigation import NavigationSystem, NavigationState
from rocket_ai_os.gnc.guidance import GuidanceSystem
from rocket_ai_os.gnc.control import FlightController
from rocket_ai_os.propulsion.engine import EngineCluster
from rocket_ai_os.propulsion.ftca import FTCAAllocator
from rocket_ai_os.propulsion.anomaly_detector import TransformerAnomalyDetector
from rocket_ai_os.fault_tolerance.fdir import FDIRSystem
from rocket_ai_os.mission.planner import HTNPlanner
from rocket_ai_os.mission.goac import GOAC, Goal, WorldState
from rocket_ai_os.mission.executive import Executive
from rocket_ai_os.mission.scheduler import UASScheduler

# ── paths ────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
LATENCY_DIR = os.path.join(ROOT, "results", "latency")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(LATENCY_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_STEPS = 1000
np.random.seed(42)

# =====================================================================
# Helper: time a callable and return (result, elapsed_ms)
# =====================================================================
def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


# =====================================================================
# 1a  Instrumented partitioned run
# =====================================================================
def run_partitioned(cfg: SystemConfig):
    """Run 1000 steps with ARINC 653 partitioned scheduling, timing each subsystem."""

    # ── initialise subsystems ────────────────────────────────────────
    nav = NavigationSystem(vehicle_config=cfg.vehicle, sim_config=cfg.sim)
    guidance = GuidanceSystem(vehicle_config=cfg.vehicle, guidance_config=cfg.guidance)
    controller = FlightController(vehicle_config=cfg.vehicle)
    cluster = EngineCluster(cfg.vehicle)
    ftca = FTCAAllocator(cluster)
    detector = TransformerAnomalyDetector(num_channels=6, window_size=64, d_model=32,
                                          n_layers=2, d_ff=64, seed=42)
    # Quick train so detector is calibrated
    nom_data = detector.generate_nominal_data(n_steps=300, noise_std=0.02)
    detector.train_nominal(nom_data, epochs=5, batch_windows=8)

    fdir = FDIRSystem()
    bus = SoftwareBus()
    planner = HTNPlanner()
    executive = Executive(planner=planner)
    scheduler = UASScheduler()
    goac = GOAC(planner=planner, executive=executive, scheduler=scheduler)
    goac.set_goal(Goal(
        name="landing_sequence",
        priority=1,
        target_state={"phase": MissionPhase.LANDING_BURN},
        utility=100.0,
    ))

    # Set controller desired state
    controller.set_desired_state(
        attitude=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.zeros(3),
        velocity=np.array([0.0, 0.0, -1.0]),
        throttle=0.6,
    )

    # ── initial state ────────────────────────────────────────────────
    true_pos = np.array([200.0, 50.0, 1500.0])
    true_vel = np.array([-50.0, 0.0, -80.0])
    mass = cfg.vehicle.dry_mass + 15_000.0
    dt = cfg.sim.dt

    # RTOS for partition utilization tracking
    rtos = PartitionedRTOS(config=cfg.rtos)
    for name in cfg.rtos.partition_schedule:
        rtos.create_partition(name=name, task=lambda: None)

    # ── logging containers ───────────────────────────────────────────
    subsystem_names = ["EKF", "G-FOLD", "PID_Controller", "RL_Controller",
                       "FTCA", "Transformer", "FDIR", "HTN_Planner",
                       "GOAC", "SW_Bus"]
    all_timings = {name: [] for name in subsystem_names}
    total_latencies = []
    frame_utilizations = []  # per-frame partition utilization

    print(f"  Running partitioned loop ({N_STEPS} steps)...", flush=True)
    for step in range(N_STEPS):
        t = step * dt
        step_t0 = time.perf_counter()
        timings = {}

        # EKF
        _, timings["EKF"] = _timed(
            nav.step, true_accel_body=np.array([0.0, 0.0, 9.81]),
            true_omega_body=np.array([0.01, -0.005, 0.0]),
            true_position=true_pos, true_velocity=true_vel,
            mass=mass, time=t)

        nav_state = nav.get_latest_state()

        # G-FOLD
        _, timings["G-FOLD"] = _timed(guidance.update, nav_state, t)

        # PID Controller
        _, timings["PID_Controller"] = _timed(controller.step, nav_state)

        # RL Controller (just the inference part)
        rl = controller.rl_controller
        obs = np.zeros(13)
        _, timings["RL_Controller"] = _timed(rl.infer, obs)

        # FTCA
        throttles = np.full(9, 0.6)
        gimbals = np.zeros((9, 2))
        cluster.command_all(throttles, gimbals)
        engine_states = cluster.step(dt, altitude=1500.0)
        _, timings["FTCA"] = _timed(
            ftca.allocate, np.array([0.0, 0.0, 400_000.0]),
            np.zeros(3), engine_states)

        # Transformer anomaly detector
        sample = nom_data[step % len(nom_data)]
        _, timings["Transformer"] = _timed(detector.detect, sample)

        # FDIR
        telemetry = {"engine_0_chamber_pressure": 9.5e6,
                     "engine_0_turbopump_rpm": 35_000.0,
                     "bus_voltage": 28.0, "avionics_temp": 45.0}
        _, timings["FDIR"] = _timed(fdir.detect, telemetry, timestamp=t)

        # HTN Planner
        _, timings["HTN_Planner"] = _timed(planner.plan, "landing_sequence", {"fuel_remaining": 0.85, "landing_legs_deployed": False, "engines_running": False})

        # GOAC
        world = WorldState(vehicle_state={"altitude": 1500.0, "speed": 90.0},
                           fuel_remaining=0.85, phase=MissionPhase.LANDING_BURN)
        _, timings["GOAC"] = _timed(goac.step, world)

        # Software Bus
        _, timings["SW_Bus"] = _timed(
            bus.publish, 0x0100, {"pos": [100, 50, 1500]},
            source="nav", priority=MessagePriority.HIGH)

        total_ms = (time.perf_counter() - step_t0) * 1000.0
        total_latencies.append(total_ms)

        for name in subsystem_names:
            all_timings[name].append(timings[name])

        # Run one RTOS frame to track utilization
        frame_timing = rtos.run_one_frame()
        frame_utilizations.append(frame_timing)

    return all_timings, total_latencies, frame_utilizations


# =====================================================================
# 1b  Monolithic baseline (flat sequential, no partitions, no bus)
# =====================================================================
def run_monolithic(cfg: SystemConfig):
    """Flat sequential loop: EKF → guidance → PID → thrust allocation."""

    nav = NavigationSystem(vehicle_config=cfg.vehicle, sim_config=cfg.sim)
    guidance = GuidanceSystem(vehicle_config=cfg.vehicle, guidance_config=cfg.guidance)
    controller = FlightController(vehicle_config=cfg.vehicle)
    cluster = EngineCluster(cfg.vehicle)
    ftca = FTCAAllocator(cluster)

    controller.set_desired_state(
        attitude=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.zeros(3),
        velocity=np.array([0.0, 0.0, -1.0]),
        throttle=0.6,
    )

    true_pos = np.array([200.0, 50.0, 1500.0])
    true_vel = np.array([-50.0, 0.0, -80.0])
    mass = cfg.vehicle.dry_mass + 15_000.0
    dt = cfg.sim.dt

    subsystem_names = ["EKF", "G-FOLD", "PID_Controller", "FTCA"]
    all_timings = {name: [] for name in subsystem_names}
    total_latencies = []

    print(f"  Running monolithic loop ({N_STEPS} steps)...", flush=True)
    for step in range(N_STEPS):
        t = step * dt
        step_t0 = time.perf_counter()
        timings = {}

        # EKF
        _, timings["EKF"] = _timed(
            nav.step, true_accel_body=np.array([0.0, 0.0, 9.81]),
            true_omega_body=np.array([0.01, -0.005, 0.0]),
            true_position=true_pos, true_velocity=true_vel,
            mass=mass, time=t)
        nav_state = nav.get_latest_state()

        # Guidance
        _, timings["G-FOLD"] = _timed(guidance.update, nav_state, t)

        # PID
        _, timings["PID_Controller"] = _timed(controller.step, nav_state)

        # Thrust allocation
        throttles = np.full(9, 0.6)
        gimbals = np.zeros((9, 2))
        cluster.command_all(throttles, gimbals)
        engine_states = cluster.step(dt, altitude=1500.0)
        _, timings["FTCA"] = _timed(
            ftca.allocate, np.array([0.0, 0.0, 400_000.0]),
            np.zeros(3), engine_states)

        total_ms = (time.perf_counter() - step_t0) * 1000.0
        total_latencies.append(total_ms)

        for name in subsystem_names:
            all_timings[name].append(timings[name])

    return all_timings, total_latencies


# =====================================================================
# 1c  Save to CSV
# =====================================================================
def save_csv(timings: dict, totals: list, filepath: str):
    """Save per-step timings to CSV."""
    names = list(timings.keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + names + ["total_ms"])
        for i in range(len(totals)):
            row = [i] + [timings[n][i] for n in names] + [totals[i]]
            writer.writerow(row)
    print(f"  Saved → {filepath}")


# =====================================================================
# Graphs
# =====================================================================
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

def plot_G01(part_timings, mono_timings):
    """G1: Bar chart — Per-subsystem latency, partitioned vs monolithic."""
    common = ["EKF", "G-FOLD", "PID_Controller", "FTCA"]
    means_p = [np.mean(part_timings[n]) for n in common]
    stds_p  = [np.std(part_timings[n])  for n in common]
    means_m = [np.mean(mono_timings[n]) for n in common]
    stds_m  = [np.std(mono_timings[n])  for n in common]

    x = np.arange(len(common))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, means_p, w, yerr=stds_p, label="Partitioned (ARINC 653)",
           capsize=4, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, means_m, w, yerr=stds_m, label="Monolithic (flat)",
           capsize=4, color="#DD8452", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=15, ha="right")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("G1 — Per-Subsystem Latency: Partitioned vs Monolithic")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G01_subsystem_latency_bar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G02(part_timings, mono_timings):
    """G2: Box plot — Latency distribution across 1000 steps."""
    common = ["EKF", "G-FOLD", "PID_Controller", "FTCA"]
    data, labels, groups = [], [], []
    for n in common:
        data.extend(part_timings[n])
        labels.extend([n] * len(part_timings[n]))
        groups.extend(["Partitioned"] * len(part_timings[n]))
        data.extend(mono_timings[n])
        labels.extend([n] * len(mono_timings[n]))
        groups.extend(["Monolithic"] * len(mono_timings[n]))

    import pandas as pd
    df = pd.DataFrame({"Latency (ms)": data, "Subsystem": labels, "Mode": groups})
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="Subsystem", y="Latency (ms)", hue="Mode", ax=ax,
                palette={"Partitioned": "#4C72B0", "Monolithic": "#DD8452"})
    ax.set_title("G2 — Latency Distribution (1000 steps) — Lower Jitter = More Deterministic")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G02_latency_distribution_box.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G03(part_totals, mono_totals):
    """G3: Line plot — Cumulative latency over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(len(part_totals))
    ax.plot(steps, np.cumsum(part_totals), label="Partitioned", color="#4C72B0", linewidth=1.2)
    ax.plot(steps, np.cumsum(mono_totals), label="Monolithic", color="#DD8452", linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative latency (ms)")
    ax.set_title("G3 — Cumulative Latency Over Time — Stability Indicator")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G03_cumulative_latency_line.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G04(frame_utils, cfg):
    """G4: Heatmap — Partition frame utilization per major frame slot."""
    partition_names = list(cfg.rtos.partition_schedule.keys())
    n_frames = min(50, len(frame_utils))  # Show first 50 frames
    matrix = np.zeros((len(partition_names), n_frames))
    for j in range(n_frames):
        fu = frame_utils[j]
        for i, pn in enumerate(partition_names):
            budgeted = cfg.rtos.partition_schedule[pn]
            actual = fu.get(pn, 0.0)
            matrix[i, j] = (actual / budgeted * 100.0) if budgeted > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(matrix, ax=ax, xticklabels=5, yticklabels=partition_names,
                cmap="YlOrRd", cbar_kws={"label": "Utilization %"},
                vmin=0, vmax=100)
    ax.set_xlabel("Major Frame #")
    ax.set_ylabel("Partition")
    ax.set_title("G4 — ARINC 653 Partition Frame Utilization (%)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G04_partition_utilization_heatmap.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("  STEP 1 — Architecture Latency Benchmarking")
    print("=" * 60)

    cfg = SystemConfig()

    # 1a + 1c: Partitioned
    part_timings, part_totals, frame_utils = run_partitioned(cfg)
    save_csv(part_timings, part_totals,
             os.path.join(LATENCY_DIR, "timings_partitioned.csv"))

    # 1b + 1c: Monolithic
    mono_timings, mono_totals = run_monolithic(cfg)
    save_csv(mono_timings, mono_totals,
             os.path.join(LATENCY_DIR, "timings_monolithic.csv"))

    # Print summary statistics
    print("\n  Summary Statistics:")
    print(f"  {'Subsystem':<20} {'Part. Mean (ms)':<18} {'Part. Std (ms)':<16} "
          f"{'Mono. Mean (ms)':<18} {'Mono. Std (ms)':<16}")
    print("  " + "-" * 86)
    common = ["EKF", "G-FOLD", "PID_Controller", "FTCA"]
    for n in common:
        pm, ps = np.mean(part_timings[n]), np.std(part_timings[n])
        mm, ms_ = np.mean(mono_timings[n]), np.std(mono_timings[n])
        print(f"  {n:<20} {pm:<18.4f} {ps:<16.4f} {mm:<18.4f} {ms_:<16.4f}")
    print(f"\n  Total end-to-end (mean): Partitioned={np.mean(part_totals):.4f} ms, "
          f"Monolithic={np.mean(mono_totals):.4f} ms")
    print(f"  Worst-case latency:      Partitioned={np.max(part_totals):.4f} ms, "
          f"Monolithic={np.max(mono_totals):.4f} ms")

    # Graphs
    print("\n  Generating graphs...")
    plot_G01(part_timings, mono_timings)
    plot_G02(part_timings, mono_timings)
    plot_G03(part_totals, mono_totals)
    plot_G04(frame_utils, cfg)

    print("\n  Step 1 COMPLETE ✓")


if __name__ == "__main__":
    main()
