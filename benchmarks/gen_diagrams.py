#!/usr/bin/env python
"""
Step 5 — Architecture Diagrams (Publication-Quality)
=====================================================

Generates 5 architecture diagrams for the Rocket-OS research paper.

Outputs:
    results/figures/FigA_system_architecture.pdf / .png
    results/figures/FigB_simplex_architecture.pdf / .png
    results/figures/FigC_transformer_architecture.pdf / .png
    results/figures/FigD_arinc653_schedule.pdf / .png
    results/figures/FigE_monte_carlo_setup.pdf / .png
"""

from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── Colour palette ───────────────────────────────────────────────────
COL = {
    "gnc":       "#4C72B0",
    "prop":      "#DD8452",
    "ft":        "#55A868",
    "mission":   "#C44E52",
    "rtos":      "#8172B3",
    "env":       "#937860",
    "comm":      "#DA8BC3",
    "sim":       "#8C8C8C",
    "bg":        "#F8F8F8",
    "arrow":     "#333333",
    "highlight": "#FFD700",
}


def _make_box(ax, xy, w, h, label, color, fontsize=9, alpha=0.85):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="black",
                         linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=3)
    return box


def _arrow(ax, start, end, color="#333", style="->", lw=1.5, connectionstyle="arc3,rad=0"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle),
                zorder=4)


# =====================================================================
# Fig A — Full System Architecture
# =====================================================================
def fig_a_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(7, 9.6, "Fig A — Rocket-OS Full System Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # Layer backgrounds
    layers = [
        (0.3, 7.3, 13.4, 2.0, "Mission Layer", COL["mission"], 0.12),
        (0.3, 4.8, 13.4, 2.3, "GNC Layer", COL["gnc"], 0.12),
        (0.3, 2.5, 13.4, 2.1, "Propulsion & Fault Tolerance Layer", COL["prop"], 0.12),
        (0.3, 0.3, 13.4, 2.0, "RTOS Layer (ARINC 653)", COL["rtos"], 0.12),
    ]
    for x, y, w, h, label, col, alpha in layers:
        bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                            facecolor=col, alpha=alpha, edgecolor="gray",
                            linewidth=0.8, linestyle="--", zorder=0)
        ax.add_patch(bg)
        ax.text(x + 0.2, y + h - 0.2, label, fontsize=8, style="italic",
                color=col, alpha=0.9, va="top")

    # Mission layer boxes
    _make_box(ax, (1, 7.7), 2.5, 1.2, "GOAC\nGoal Manager", COL["mission"])
    _make_box(ax, (4.5, 7.7), 2.5, 1.2, "HTN\nPlanner", COL["mission"])
    _make_box(ax, (8, 7.7), 2.5, 1.2, "Executive\nDispatcher", COL["mission"])
    _make_box(ax, (11.5, 7.7), 1.8, 1.2, "UAS\nScheduler", COL["mission"])

    # GNC layer boxes
    _make_box(ax, (0.8, 5.2), 2.5, 1.5, "EKF\nNavigation\n(16-state)", COL["gnc"])
    _make_box(ax, (4.2, 5.2), 2.5, 1.5, "G-FOLD\nGuidance\n(Convex Opt)", COL["gnc"])
    _make_box(ax, (7.6, 5.2), 2.5, 1.5, "PID + RL\nFlight Control", COL["gnc"])
    _make_box(ax, (10.8, 5.2), 2.5, 1.5, "Simplex\nSafety Monitor", COL["ft"])

    # Propulsion & FT layer
    _make_box(ax, (0.8, 2.8), 2.5, 1.5, "Engine Cluster\n(9 Merlins)", COL["prop"])
    _make_box(ax, (4.2, 2.8), 2.5, 1.5, "FTCA\n(WLS Alloc)", COL["prop"])
    _make_box(ax, (7.6, 2.8), 2.5, 1.5, "Transformer\nAnomaly Det.", COL["prop"])
    _make_box(ax, (10.8, 2.8), 2.5, 1.5, "FDIR\nFault Manager", COL["ft"])

    # RTOS layer
    _make_box(ax, (0.8, 0.6), 3, 1.4, "Cyclic Executive\nScheduler", COL["rtos"])
    _make_box(ax, (4.5, 0.6), 2.5, 1.4, "Memory\nPartitions", COL["rtos"])
    _make_box(ax, (7.8, 0.6), 2.5, 1.4, "IPC Ports\n(Sampling/Queue)", COL["rtos"])
    _make_box(ax, (11, 0.6), 2.5, 1.4, "Health\nMonitor", COL["rtos"])

    # Key arrows (data flow)
    _arrow(ax, (2.25, 7.7), (2.25, 6.7), COL["arrow"])  # GOAC → EKF area
    _arrow(ax, (5.75, 7.7), (5.75, 6.7))  # HTN → G-FOLD area
    _arrow(ax, (3.3, 5.95), (4.2, 5.95))  # EKF → G-FOLD
    _arrow(ax, (6.7, 5.95), (7.6, 5.95))  # G-FOLD → Controller
    _arrow(ax, (10.1, 5.95), (10.8, 5.95))  # Controller → Simplex
    _arrow(ax, (8.85, 5.2), (8.85, 4.3))  # Controller → FTCA
    _arrow(ax, (5.45, 4.3), (5.45, 2.8), style="->")  # FTCA
    _arrow(ax, (3.3, 3.55), (4.2, 3.55))  # Engine → FTCA
    _arrow(ax, (6.7, 3.55), (7.6, 3.55))  # FTCA → Anomaly
    _arrow(ax, (10.1, 3.55), (10.8, 3.55))  # Anomaly → FDIR
    _arrow(ax, (2.3, 2.8), (2.3, 2.0))  # Engine → Scheduler
    _arrow(ax, (12.05, 2.8), (12.05, 2.0))  # FDIR → Health Mon

    for fmt in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"FigA_system_architecture.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → FigA_system_architecture.pdf/.png")


# =====================================================================
# Fig B — Simplex Architecture Detail
# =====================================================================
def fig_b_simplex_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6, 6.6, "Fig B — Simplex Architecture for AI Safety Assurance",
            ha="center", fontsize=13, fontweight="bold")

    # Input
    _make_box(ax, (0.5, 3.5), 2.0, 1.5, "Navigation\nState\n(EKF Output)", COL["gnc"])

    # RL Controller
    _make_box(ax, (3.5, 4.5), 2.5, 1.5, "RL Adaptive\nController\n(AI, unverified)", "#E8A8AA", fontsize=9)

    # Safety Controller
    _make_box(ax, (3.5, 1.5), 2.5, 1.5, "Safety PD\nController\n(verified)", "#A8D5A2", fontsize=9)

    # Decision Module
    _make_box(ax, (7.0, 2.5), 2.5, 2.5, "Decision\nModule\n\n• Forward Sim\n• Reachable Set\n• Envelope Check",
              COL["highlight"], fontsize=8, alpha=0.9)

    # Output
    _make_box(ax, (10.2, 3.2), 1.5, 1.2, "Actuator\nCommands", COL["prop"])

    # Arrows
    _arrow(ax, (2.5, 4.25), (3.5, 5.0))    # Nav → RL
    _arrow(ax, (2.5, 4.25), (3.5, 2.5))    # Nav → Safety
    _arrow(ax, (6.0, 5.25), (7.0, 4.5))    # RL → Decision
    _arrow(ax, (6.0, 2.25), (7.0, 3.0))    # Safety → Decision
    _arrow(ax, (9.5, 3.75), (10.2, 3.8))  # Decision → Output

    # Labels on arrows
    ax.text(4.7, 5.2, "proposed\naction", fontsize=7, ha="center",
            color="#C44E52", style="italic")
    ax.text(4.7, 1.5, "safe\nfallback", fontsize=7, ha="center",
            color="#55A868", style="italic")
    ax.text(9.8, 4.5, "approved\nor safety", fontsize=7, ha="center",
            color=COL["arrow"], style="italic")

    # Safety envelope box
    env_box = FancyBboxPatch((6.8, 0.3), 3.0, 1.5, boxstyle="round,pad=0.1",
                             facecolor="#FFF3CD", edgecolor="#856404",
                             linewidth=1, alpha=0.8, zorder=2)
    ax.add_patch(env_box)
    ax.text(8.3, 1.05, "Safety Envelope\nMax-Q | AoA | ω | Alt | Accel",
            ha="center", va="center", fontsize=8, fontweight="bold")
    _arrow(ax, (8.25, 1.8), (8.25, 2.5), color="#856404")

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"FigB_simplex_architecture.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → FigB_simplex_architecture.pdf/.png")


# =====================================================================
# Fig C — Transformer Model Architecture
# =====================================================================
def fig_c_transformer_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6, 7.5, "Fig C — Transformer-Based Anomaly Detector (T-BAD)",
            ha="center", fontsize=13, fontweight="bold")

    # Input
    _make_box(ax, (0.5, 3.0), 2.0, 2.0, "Sensor\nWindow\n(T × 6)\n\nRPM, P,\nVib, Temp",
              COL["prop"], fontsize=8)

    # Input projection
    _make_box(ax, (3.3, 3.5), 1.8, 1.2, "Linear\nProjection\n→ d_model", "#B8D4E3")

    # Positional encoding
    _make_box(ax, (3.3, 5.5), 1.8, 1.0, "Positional\nEncoding\n(sinusoidal)", "#D4E3B8")
    _arrow(ax, (4.2, 5.5), (4.2, 4.7))  # PE → +

    # Transformer blocks
    _make_box(ax, (5.8, 2.5), 2.2, 3.5, "Self-Attention\nBlock ×2\n\n• Q, K, V\n• Softmax\n• Residual\n• LayerNorm\n• FFN + GELU",
              "#E3D4B8", fontsize=8, alpha=0.9)

    # Output
    _make_box(ax, (8.8, 3.5), 1.8, 1.2, "Output\nProjection\n→ 6 ch", "#B8D4E3")

    # Anomaly computation
    _make_box(ax, (8.8, 5.5), 1.8, 1.2, "MSE Error\n→ Anomaly\nScore [0,1]", "#FFCCCC")

    # Result
    _make_box(ax, (8.8, 1.0), 1.8, 1.2, "Failure\nClassifier\n(Cosine Sim)", "#CCE5CC")

    # Arrows
    _arrow(ax, (2.5, 4.0), (3.3, 4.1))   # Input → Proj
    _arrow(ax, (5.1, 4.1), (5.8, 4.1))   # Proj → Blocks
    _arrow(ax, (8.0, 4.0), (8.8, 4.1))   # Blocks → Output
    _arrow(ax, (9.7, 4.7), (9.7, 5.5))   # Output → MSE
    _arrow(ax, (9.7, 3.5), (9.7, 2.2))   # Output → Classifier

    # Labels
    ax.text(2.0, 2.5, "T=64 steps\n6 channels", fontsize=7, ha="center",
            style="italic", color="gray")

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"FigC_transformer_architecture.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → FigC_transformer_architecture.pdf/.png")


# =====================================================================
# Fig D — ARINC 653 Partition Schedule
# =====================================================================
def fig_d_arinc653_schedule():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Partition schedule (from config.py defaults)
    partitions = [
        ("navigation",    5.0, COL["gnc"]),
        ("guidance",      8.0, COL["gnc"]),
        ("control",       4.0, COL["gnc"]),
        ("propulsion",    3.0, COL["prop"]),
        ("fault_mgmt",    3.0, COL["ft"]),
        ("comms",         2.0, COL["comm"]),
        ("telemetry",     2.0, COL["sim"]),
        ("mission_mgmt",  3.0, COL["mission"]),
    ]

    major_frame = 50.0  # ms
    n_frames = 3  # show 3 major frames

    ax.set_xlim(-2, major_frame * n_frames + 5)
    ax.set_ylim(-0.5, len(partitions) + 1)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Fig D — ARINC 653 Cyclic Executive Schedule (3 Major Frames)", fontsize=13)

    yticks = []
    ylabels = []

    for frame in range(n_frames):
        offset = frame * major_frame
        t_cursor = offset

        # Major frame boundary
        ax.axvline(x=offset, color="red", linewidth=2, linestyle="-", alpha=0.6)
        ax.text(offset + major_frame / 2, len(partitions) + 0.5,
                f"Major Frame #{frame + 1}", ha="center", fontsize=9,
                fontweight="bold", color="red", alpha=0.8)

        for i, (name, duration, color) in enumerate(partitions):
            y = len(partitions) - 1 - i
            bar = ax.barh(y, duration, left=t_cursor, height=0.7,
                          color=color, edgecolor="black", linewidth=0.8, alpha=0.85)

            if frame == 0:
                yticks.append(y)
                ylabels.append(name)

            # Label inside bar
            if duration >= 2.5:
                ax.text(t_cursor + duration / 2, y, f"{duration:.0f}ms",
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white")

            t_cursor += duration

        # Slack
        slack = major_frame - sum(d for _, d, _ in partitions)
        if slack > 0:
            ax.barh(-0.3, slack, left=t_cursor, height=len(partitions) * 0.8 + 0.3,
                    color="lightgray", alpha=0.3, edgecolor="none")
            ax.text(t_cursor + slack / 2, -0.1, f"Slack\n{slack:.0f}ms",
                    ha="center", va="center", fontsize=7, color="gray")

    # End boundary
    ax.axvline(x=n_frames * major_frame, color="red", linewidth=2, alpha=0.6)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"FigD_arinc653_schedule.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → FigD_arinc653_schedule.pdf/.png")


# =====================================================================
# Fig E — Monte Carlo Experiment Setup
# =====================================================================
def fig_e_monte_carlo_setup():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6.5, 6.6, "Fig E — Monte Carlo Evaluation Framework",
            ha="center", fontsize=13, fontweight="bold")

    # Step 1: Configuration
    _make_box(ax, (0.3, 4.5), 2.5, 1.8, "Configuration\n\n• N=500 trials\n• 3 fault types\n• 3 severities",
              "#B8D4E3", fontsize=8)

    # Step 2: Scenario Generator
    _make_box(ax, (3.5, 4.5), 2.3, 1.8, "Scenario\nGenerator\n\n• Random onset\n• Random IC's\n• Seed control",
              "#D4E3B8", fontsize=8)

    # Step 3: Simulation (two paths)
    _make_box(ax, (6.5, 5.2), 2.3, 1.3, "Sim + Simplex\n(safety ON)", "#A8D5A2", fontsize=9)
    _make_box(ax, (6.5, 3.5), 2.3, 1.3, "Sim − Simplex\n(safety OFF)", "#E8A8AA", fontsize=9)

    # Step 4: Metrics
    _make_box(ax, (9.5, 4.3), 2.8, 2.0, "Metrics\n\n• Success / Fail\n• Recovery time\n• Trajectory RMSE\n• Control effort",
              COL["highlight"], fontsize=8, alpha=0.85)

    # Step 5: Aggregation
    _make_box(ax, (4.0, 1.0), 3.0, 1.8, "Statistical\nAggregation\n\n• Mean ± Std\n• Effect of severity\n• Per fault-type",
              "#E3D4B8", fontsize=8)

    # Step 6: Figures
    _make_box(ax, (8.0, 1.0), 3.0, 1.8, "Graphs\nG05—G09\n\n• Bar, Box, Line\n• Scatter, Stacked",
              "#D4B8E3", fontsize=8)

    # Arrows
    _arrow(ax, (2.8, 5.4), (3.5, 5.4))
    _arrow(ax, (5.8, 5.7), (6.5, 5.7))
    _arrow(ax, (5.8, 5.0), (6.5, 4.3))
    _arrow(ax, (8.8, 5.85), (9.5, 5.5))
    _arrow(ax, (8.8, 4.15), (9.5, 4.8))
    _arrow(ax, (10.9, 4.3), (9.5, 2.2), style="->",
           connectionstyle="arc3,rad=-0.2")
    _arrow(ax, (7.0, 2.8), (8.0, 2.0))

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"FigE_monte_carlo_setup.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → FigE_monte_carlo_setup.pdf/.png")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("  STEP 5 — Architecture Diagrams")
    print("=" * 60)

    fig_a_system_architecture()
    fig_b_simplex_architecture()
    fig_c_transformer_architecture()
    fig_d_arinc653_schedule()
    fig_e_monte_carlo_setup()

    print("\n  Step 5 COMPLETE ✓")


if __name__ == "__main__":
    main()
