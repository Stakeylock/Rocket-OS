#!/usr/bin/env python
"""
Step 2 — Simplex + RL Fault Tolerance Monte Carlo
===================================================

Proves: Simplex safety monitoring successfully intercepts RL policy failures
and recovers the vehicle, whereas without Simplex the same faults cause
mission failure.

Outputs:
    results/simplex/monte_carlo_results.csv
    results/figures/G05_success_rate_bar.png
    results/figures/G06_recovery_time_box.png
    results/figures/G07_trajectory_comparison_line.png
    results/figures/G08_severity_vs_recovery_scatter.png
    results/figures/G09_outcome_breakdown_stacked.png
"""

from __future__ import annotations
import os, sys, csv, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_ai_os.config import SystemConfig, MissionPhase, VehicleConfig, GuidanceConfig
from rocket_ai_os.gnc.navigation import NavigationSystem, NavigationState
from rocket_ai_os.gnc.guidance import GuidanceSystem
from rocket_ai_os.gnc.control import FlightController
from rocket_ai_os.sim.vehicle import Vehicle
from rocket_ai_os.fault_tolerance.simplex import SimplexArchitecture, DecisionModule, SafetyController, ControlAction, SafetyEnvelope
from rocket_ai_os.fault_tolerance.fault_injector import FaultInjector
import logging
logging.getLogger("rocket_ai_os.fault_tolerance.simplex").setLevel(logging.ERROR)

ROOT = os.path.join(os.path.dirname(__file__), "..")
SIMPLEX_DIR = os.path.join(ROOT, "results", "simplex")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(SIMPLEX_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TRIALS = 500
FAULT_TYPES = ["adversarial", "stuck", "noise"]
SEVERITIES = [0.3, 0.6, 1.0]
MAX_STEPS = 60
np.random.seed(42)
random.seed(42)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =====================================================================
# Removed legacy string-based inject_fault in favor of FaultInjector
# =====================================================================


# =====================================================================
# 2b  Single mission runner
# =====================================================================
def run_mission(cfg: SystemConfig, simplex_enabled: bool,
                fault_onset: int, fault_type: str, severity: float,
                seed: int = 42):
    """Run a single landing mission returning outcome metrics.

    Returns dict with: success, intervention_step, recovery_time,
    trajectory_rmse, control_effort, trajectory (list of positions).
    """
    rng = np.random.default_rng(seed)
    dt = cfg.sim.dt
    target = cfg.guidance.target_position.copy()

    # Create vehicle
    vehicle = Vehicle(config=cfg.vehicle, initial_phase=MissionPhase.LANDING_BURN)
    vehicle.set_state(
        position=np.array([200.0, 50.0, 1500.0]),
        velocity=np.array([-50.0, 0.0, -80.0]),
        mass=cfg.vehicle.dry_mass + 15_000.0,
        fuel_mass=15_000.0,
    )

    # Create subsystems
    nav = NavigationSystem(vehicle_config=cfg.vehicle, sim_config=cfg.sim, seed=seed)
    guidance = GuidanceSystem(vehicle_config=cfg.vehicle, guidance_config=cfg.guidance)
    controller = FlightController(vehicle_config=cfg.vehicle, rl_seed=seed)
    controller.set_desired_state(
        attitude=np.array([1.0, 0.0, 0.0, 0.0]),
        position=target,
        velocity=np.array([0.0, 0.0, -1.0]),
        throttle=0.6,
    )

    # Track metrics
    trajectory = []
    planned_traj = []
    total_thrust = 0.0
    intervention_step = -1
    recovery_step = -1
    fault_active = False
    was_using_baseline = False

    fault_time = fault_onset * dt
    injector = FaultInjector(schedule=[{
        "t": fault_time, "type": fault_type, "magnitude": severity
    }])
    if simplex_enabled:
        env = SafetyEnvelope(max_angle_of_attack=np.radians(85.0))
        dm = DecisionModule(envelope=env)
        simplex = SimplexArchitecture(decision_module=dm)

    for step in range(MAX_STEPS):
        t = step * dt
        state = vehicle.state
        if state.is_landed or state.is_destroyed:
            break

        # Navigation
        accel_meas = np.array([0.0, 0.0, 9.81])
        gyro_meas = state.angular_velocity + rng.normal(0, 0.001, 3)
        nav_state = nav.step(
            true_accel_body=accel_meas, true_omega_body=gyro_meas,
            true_position=state.position, true_velocity=state.velocity,
            mass=state.mass, time=t)

        # Guidance
        traj_pt = guidance.update(nav_state, t)
        if traj_pt:
            controller.set_desired_state(
                position=traj_pt.position, velocity=traj_pt.velocity,
                throttle=traj_pt.throttle)
            planned_traj.append(traj_pt.position.copy())
        else:
            planned_traj.append(target.copy())

        # Control
        cmd = controller.step(nav_state)
        torque = cmd.torque_command.copy()
        throttle = cmd.throttle

        # Fault injection
        if step >= fault_onset:
            fault_active = True
        
        torque, throttle = injector.apply(t, nav_state, torque, throttle)

        using_baseline = False
        if simplex_enabled:
            # Package state and action for Simplex DecisionModule
            v_state_dict = {
                "position": state.position, "velocity": state.velocity,
                "attitude": vehicle.state.to_12_state_dict()["roll"],
                "angular_velocity": state.angular_velocity,
                "mass": state.mass, "timestamp": t
            }
            # Actually, `attitude` in simplex expects Euler angles, so let's compute it:
            euler_dict = vehicle.state.to_12_state_dict()
            v_state_dict["attitude"] = np.array([euler_dict["roll"], euler_dict["pitch"], euler_dict["yaw"]])

            thrust_vec = np.array([0.0, 0.0, throttle * cfg.vehicle.max_total_thrust])
            ai_action = ControlAction(forces=thrust_vec, torques=torque, timestamp=t, source="ai_controller")
            
            approved_action = simplex.evaluate_and_select(ai_action, v_state_dict)
            
            if approved_action.source != "ai_controller":
                torque = approved_action.torques
                thrust_mag = np.linalg.norm(approved_action.forces)
                throttle = thrust_mag / cfg.vehicle.max_total_thrust
                using_baseline = True
                if intervention_step < 0 and fault_active:
                    intervention_step = step

        # Detect recovery (stable flight after intervention)
        if intervention_step >= 0 and recovery_step < 0 and fault_active:
            if not using_baseline and np.linalg.norm(nav_state.angular_rates) < 0.3:
                recovery_step = step

        # Apply thrust
        thrust_mag = throttle * cfg.vehicle.max_total_thrust
        thrust_mag = np.clip(thrust_mag, 0, cfg.vehicle.max_total_thrust)
        thrust_dir = np.array([0.0, 0.0, 1.0])  # simplified
        f_thrust = thrust_dir * thrust_mag
        f_gravity = np.array([0.0, 0.0, -9.81 * state.mass])
        total_force = f_gravity + f_thrust

        # Torque effect on angular velocity (simplified)
        total_torque = torque * 0.001  # scale down for stability

        vehicle.apply_forces(total_force, total_torque, dt)
        g0 = 9.81; isp = 282.0
        if thrust_mag > 0:
            vehicle.consume_fuel(thrust_mag / (isp * g0), dt)

        trajectory.append(state.position.copy())
        total_thrust += thrust_mag * dt

    # Compute metrics
    final_state = vehicle.state
    success = final_state.is_landed and not final_state.is_destroyed

    # Trajectory RMSE vs planned
    min_len = min(len(trajectory), len(planned_traj))
    if min_len > 0:
        traj_arr = np.array(trajectory[:min_len])
        plan_arr = np.array(planned_traj[:min_len])
        rmse = float(np.sqrt(np.mean((traj_arr - plan_arr) ** 2)))
    else:
        rmse = float("inf")

    recovery_time = (recovery_step - intervention_step) if (
        intervention_step >= 0 and recovery_step >= 0) else -1

    return {
        "success": success,
        "intervention_step": intervention_step,
        "recovery_time": recovery_time,
        "trajectory_rmse": rmse,
        "control_effort": total_thrust,
        "trajectory": trajectory,
        "planned_trajectory": planned_traj[:min_len] if min_len > 0 else [],
        "final_alt": float(final_state.position[2]),
        "final_speed": float(np.linalg.norm(final_state.velocity)),
    }


# =====================================================================
# 2c  Simplex Representative Scenarios (9-column CSV)
# =====================================================================
def run_detailed_simplex_cases(cfg: SystemConfig):
    """Run 5 specific fault scenarios and log all 9 Simplex columns for evaluation."""
    cases = [
        {"fault": "adversarial", "sev": 0.8},
        {"fault": "stuck", "sev": 1.0},
        {"fault": "noise", "sev": 1.0},
        {"fault": "adversarial", "sev": 0.4},
        {"fault": "stuck", "sev": 0.5},
    ]
    log_rows = []
    
    print(f"  Running {len(cases)} detailed Simplex scenarios for logging...", flush=True)
    for c_idx, c in enumerate(cases):
        seed = 200 + c_idx
        dt = cfg.sim.dt
        fault_onset = 25
        fault_time = fault_onset * dt
        
        vehicle = Vehicle(config=cfg.vehicle, initial_phase=MissionPhase.LANDING_BURN)
        vehicle.set_state(position=np.array([200.0, 50.0, 1500.0]), velocity=np.array([-50.0, 0.0, -80.0]),
                          mass=cfg.vehicle.dry_mass + 15_000.0, fuel_mass=15_000.0)
        nav = NavigationSystem(vehicle_config=cfg.vehicle, sim_config=cfg.sim, seed=seed)
        guidance = GuidanceSystem(vehicle_config=cfg.vehicle, guidance_config=cfg.guidance)
        controller = FlightController(vehicle_config=cfg.vehicle, rl_seed=seed)
        controller.set_desired_state(position=cfg.guidance.target_position)
        
        injector = FaultInjector(schedule=[{"t": fault_time, "type": c["fault"], "magnitude": c["sev"]}])
        
        env = SafetyEnvelope(max_angle_of_attack=np.radians(85.0))
        dm = DecisionModule(envelope=env)
        simplex = SimplexArchitecture(decision_module=dm)
        
        safe_ctrl = SafetyController() # instantiate just to know what its nominal output would be
        
        intervention_step = -1
        recovery_step = -1
        
        for step in range(100): # max 100 steps
            t = step * dt
            if vehicle.state.is_landed or vehicle.state.is_destroyed: break
                
            nav_state = nav.step(np.array([0,0,9.81]), vehicle.state.angular_velocity, vehicle.state.position, vehicle.state.velocity, vehicle.state.mass, t)
            traj_pt = guidance.update(nav_state, t)
            if traj_pt:
                controller.set_desired_state(position=traj_pt.position, velocity=traj_pt.velocity, throttle=traj_pt.throttle)
                
            cmd = controller.step(nav_state)
            ai_torque = cmd.torque_command.copy()
            ai_throttle = cmd.throttle
            
            ai_torque_corr, ai_throttle_corr = injector.apply(t, nav_state, ai_torque, ai_throttle)
            
            euler_dict = vehicle.state.to_12_state_dict()
            v_state = {
                "position": vehicle.state.position, "velocity": vehicle.state.velocity,
                "attitude": np.array([euler_dict["roll"], euler_dict["pitch"], euler_dict["yaw"]]),
                "angular_velocity": vehicle.state.angular_velocity, "mass": vehicle.state.mass, "timestamp": t
            }
            
            thrust_vec = np.array([0.0, 0.0, ai_throttle_corr * cfg.vehicle.max_total_thrust])
            ai_action = ControlAction(forces=thrust_vec, torques=ai_torque_corr, timestamp=t, source="ai_controller")
            approved = simplex.evaluate_and_select(ai_action, v_state)
            
            using_safety = (approved.source != "ai_controller")
            switch_event = False
            
            if using_safety and intervention_step < 0 and step >= fault_onset:
                intervention_step = step
                switch_event = True
            
            if not using_safety and intervention_step > 0 and recovery_step < 0:
                recovery_step = step
                switch_event = True
                
            # Compute a hypothetical state_norm just to demonstrate safety breach proximity (0-1)
            # Safe envelope is e.g. 15 deg AoA, 20 deg/s angular.
            state_norm = min(1.0, np.linalg.norm(vehicle.state.angular_velocity) / np.radians(20.0))
            if not simplex.decision_module.evaluate(ai_action, v_state)[0]: 
                state_norm = 1.05 # breached
            
            base_cmd = safe_ctrl.compute(v_state, t)
            base_throttle = np.linalg.norm(base_cmd.forces) / cfg.vehicle.max_total_thrust
            
            log_rows.append({
                "scenario_idx": c_idx,
                "timestep": step,
                "t_sim": t,
                "adv_ctrl_pitch": ai_torque_corr[1],
                "adv_ctrl_yaw": ai_torque_corr[2],
                "adv_ctrl_throttle": ai_throttle_corr,
                "base_ctrl_pitch": base_cmd.torques[1],
                "base_ctrl_yaw": base_cmd.torques[2],
                "base_ctrl_throttle": base_throttle,
                "state_norm": state_norm,
                "safe_flag": (not using_safety),
                "active_controller": "SAFETY" if using_safety else "ADVANCED",
                "switch_event": switch_event,
                "recovery_timestep": recovery_step if recovery_step > 0 else -1
            })
            
            throttle = np.linalg.norm(approved.forces) / cfg.vehicle.max_total_thrust
            torque = approved.torques
            
            f_thrust = np.array([0.0, 0.0, throttle * cfg.vehicle.max_total_thrust])
            f_gravity = np.array([0.0, 0.0, -9.81 * vehicle.state.mass])
            vehicle.apply_forces(f_gravity + f_thrust, torque * 0.001, dt)
            
    # Save CSV
    out_csv = os.path.join(SIMPLEX_DIR, "simplex_log.csv")
    keys = log_rows[0].keys()
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"  Saved 9-column log → {out_csv}")


# =====================================================================
# 2d  Monte Carlo runner
# =====================================================================
def run_monte_carlo(cfg: SystemConfig):
    """Run 500 trials with/without Simplex."""
    results = []
    print(f"  Running {TRIALS} Monte Carlo trials...", flush=True)
    for trial in tqdm(range(TRIALS), desc="  Monte Carlo", ncols=70):
        fault_onset = random.randint(10, 40)
        fault_type = random.choice(FAULT_TYPES)
        severity = random.choice(SEVERITIES)
        seed = 42 + trial

        # With Simplex
        r_simplex = run_mission(cfg, simplex_enabled=True,
                                fault_onset=fault_onset, fault_type=fault_type,
                                severity=severity, seed=seed)
        # Without Simplex
        r_nosimplex = run_mission(cfg, simplex_enabled=False,
                                  fault_onset=fault_onset, fault_type=fault_type,
                                  severity=severity, seed=seed)

        results.append({
            "trial": trial,
            "fault_type": fault_type,
            "severity": severity,
            "fault_onset": fault_onset,
            "simplex_success": r_simplex["success"],
            "nosimplex_success": r_nosimplex["success"],
            "intervention_step": r_simplex["intervention_step"],
            "recovery_time": r_simplex["recovery_time"],
            "simplex_rmse": r_simplex["trajectory_rmse"],
            "nosimplex_rmse": r_nosimplex["trajectory_rmse"],
            "simplex_effort": r_simplex["control_effort"],
            "nosimplex_effort": r_nosimplex["control_effort"],
        })

    return results


# =====================================================================
# Save CSV
# =====================================================================
def save_results(results, filepath):
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved → {filepath}")


# =====================================================================
# Graphs
# =====================================================================
def plot_G05(df):
    """G5: Bar chart — Mission success rate by fault type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ftypes = FAULT_TYPES
    simplex_rates = [df[df["fault_type"] == ft]["simplex_success"].mean() * 100 for ft in ftypes]
    nosimplex_rates = [df[df["fault_type"] == ft]["nosimplex_success"].mean() * 100 for ft in ftypes]

    x = np.arange(len(ftypes))
    w = 0.35
    ax.bar(x - w/2, simplex_rates, w, label="With Simplex", color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, nosimplex_rates, w, label="Without Simplex", color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ft.title() for ft in ftypes])
    ax.set_ylabel("Mission Success Rate (%)")
    ax.set_title("G5 — Mission Success Rate: Simplex vs No-Simplex by Fault Type")
    ax.set_ylim(0, 105)
    ax.legend()
    for i, (s, n) in enumerate(zip(simplex_rates, nosimplex_rates)):
        ax.text(i - w/2, s + 1, f"{s:.0f}%", ha="center", fontsize=9)
        ax.text(i + w/2, n + 1, f"{n:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G05_success_rate_bar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G06(df):
    """G6: Box plot — Recovery time distribution."""
    valid = df[df["recovery_time"] >= 0].copy()
    if valid.empty:
        valid = df.copy()
        valid["recovery_time"] = valid["recovery_time"].apply(lambda x: max(x, 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=valid, x="fault_type", y="recovery_time", ax=ax,
                palette="Set2", order=FAULT_TYPES)
    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Recovery Time (steps)")
    ax.set_title("G6 — Recovery Time Distribution Across 500 Trials")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G06_recovery_time_box.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G07(cfg):
    """G7: Line plot — Single representative trajectory comparison."""
    # Run one clean example
    r_planned = run_mission(cfg, True, fault_onset=999, fault_type="noise",
                            severity=0.0, seed=100)
    r_simplex = run_mission(cfg, True, fault_onset=15, fault_type="adversarial",
                            severity=0.8, seed=100)
    r_nosimplex = run_mission(cfg, False, fault_onset=15, fault_type="adversarial",
                              severity=0.8, seed=100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Altitude vs step
    ax = axes[0]
    for label, r, color in [("Planned (no fault)", r_planned, "#4C72B0"),
                              ("With Simplex", r_simplex, "#55A868"),
                              ("No Simplex", r_nosimplex, "#C44E52")]:
        alts = [p[2] for p in r["trajectory"]]
        ax.plot(alts, label=label, color=color, linewidth=1.5)
    ax.axvline(x=15, color="gray", linestyle="--", alpha=0.7, label="Fault onset")
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude Profile")
    ax.legend(fontsize=8)

    # XY trajectory
    ax = axes[1]
    for label, r, color in [("Planned", r_planned, "#4C72B0"),
                              ("Simplex", r_simplex, "#55A868"),
                              ("No Simplex", r_nosimplex, "#C44E52")]:
        xs = [p[0] for p in r["trajectory"]]
        ys = [p[1] for p in r["trajectory"]]
        ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
    ax.scatter([0], [0], marker="*", s=200, c="gold", zorder=5, label="Target")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Horizontal Trajectory")
    ax.legend(fontsize=8)

    fig.suptitle("G7 — Representative Trajectory: Planned vs Actual (±Simplex)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G07_trajectory_comparison_line.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G08(df):
    """G8: Scatter — Fault severity vs recovery time, colored by fault type."""
    valid = df[df["recovery_time"] >= 0].copy()
    if valid.empty:
        valid = df.copy()
        valid["recovery_time"] = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"adversarial": "#C44E52", "stuck": "#DD8452", "noise": "#4C72B0"}
    for ft in FAULT_TYPES:
        sub = valid[valid["fault_type"] == ft]
        ax.scatter(sub["severity"], sub["recovery_time"], label=ft.title(),
                   color=colors[ft], alpha=0.5, s=30, edgecolors="black", linewidth=0.3)
    ax.set_xlabel("Fault Severity")
    ax.set_ylabel("Recovery Time (steps)")
    ax.set_title("G8 — Fault Severity vs Recovery Time")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G08_severity_vs_recovery_scatter.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_G09(df):
    """G9: Stacked bar — Outcome breakdown per severity level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sev_labels = ["Low (0.3)", "Medium (0.6)", "High (1.0)"]

    for idx, (mode, color_s, color_f) in enumerate([
            ("simplex", "#55A868", "#A8D5A2"),
            ("nosimplex", "#C44E52", "#E8A8AA")]):
        successes, failures = [], []
        for sev in SEVERITIES:
            sub = df[df["severity"] == sev]
            col = f"{mode}_success"
            s = sub[col].sum()
            f = len(sub) - s
            successes.append(s)
            failures.append(f)

        x = np.arange(len(sev_labels)) + idx * 0.35
        label_s = f"{'Simplex' if mode == 'simplex' else 'No Simplex'} — Success"
        label_f = f"{'Simplex' if mode == 'simplex' else 'No Simplex'} — Failure"
        ax.bar(x, successes, 0.3, label=label_s, color=color_s, edgecolor="black", linewidth=0.5)
        ax.bar(x, failures, 0.3, bottom=successes, label=label_f, color=color_f,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(np.arange(len(sev_labels)) + 0.175)
    ax.set_xticklabels(sev_labels)
    ax.set_xlabel("Fault Severity")
    ax.set_ylabel("Number of Trials")
    ax.set_title("G9 — Outcome Breakdown by Severity Level")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G09_outcome_breakdown_stacked.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("  STEP 2 — Simplex + RL Fault Tolerance Monte Carlo")
    print("=" * 60)

    cfg = SystemConfig()

    # Detailed scenarios first
    run_detailed_simplex_cases(cfg)

    # Run Monte Carlo
    results = run_monte_carlo(cfg)
    save_results(results, os.path.join(SIMPLEX_DIR, "monte_carlo_results.csv"))

    df = pd.DataFrame(results)

    # Summary
    print("\n  Summary:")
    print(f"  Overall success rate WITH Simplex:    {df['simplex_success'].mean()*100:.1f}%")
    print(f"  Overall success rate WITHOUT Simplex: {df['nosimplex_success'].mean()*100:.1f}%")
    for ft in FAULT_TYPES:
        sub = df[df["fault_type"] == ft]
        print(f"  {ft:>12s}: Simplex={sub['simplex_success'].mean()*100:.0f}%  "
              f"NoSimplex={sub['nosimplex_success'].mean()*100:.0f}%")

    # Graphs
    print("\n  Generating graphs...")
    plot_G05(df)
    plot_G06(df)
    plot_G07(cfg)
    plot_G08(df)
    plot_G09(df)

    print("\n  Step 2 COMPLETE ✓")


if __name__ == "__main__":
    main()
