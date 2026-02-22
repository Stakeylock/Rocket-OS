# Autonomous Rocket AI OS

> **Resilient Autonomy: A Next-Generation Avionics Architecture for Self-Governing Launch Vehicles**

A comprehensive simulation framework implementing a full autonomous avionics stack for launch vehicles, covering 20 integrated subsystems from GNC and propulsion to fault tolerance and mission management.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/dependency-numpy-orange)
![Version](https://img.shields.io/badge/version-1.0.0-green)

---

## Overview

This project is the reference implementation for the research paper *"Resilient Autonomy: A Next-Generation Avionics Architecture for Self-Governing Launch Vehicles"*. It demonstrates a complete, integrated avionics software stack for an autonomous rocket — from low-level RTOS partitioning through high-level goal-oriented mission planning — running inside a 6-DOF physics simulation.

## Key Features

| # | Subsystem | Description |
|---|-----------|-------------|
| 1 | **ARINC 653 RTOS** | Partitioned RTOS with time/space isolation |
| 2 | **cFS Software Bus + DDS** | Middleware bridge for publish-subscribe messaging |
| 3 | **EKF Navigation** | Extended Kalman Filter fusing IMU + GPS |
| 4 | **G-FOLD Guidance** | Convex trajectory optimisation for powered descent |
| 5 | **Flight Controller** | PID + RL Adaptive control with Simplex safety |
| 6 | **Engine Cluster + FTCA** | 9-engine cluster with fault-tolerant control allocation |
| 7 | **Anomaly Detection** | Transformer-based engine anomaly detection |
| 8 | **TTEthernet** | Deterministic networking with triplex redundancy |
| 9 | **Simplex Architecture** | Safety assurance for AI-in-the-loop control |
| 10 | **TMR + SEU Scrubbing** | Triple Modular Redundancy with radiation mitigation |
| 11 | **FDIR** | Hierarchical Fault Detection, Isolation & Recovery |
| 12 | **Cognitive Radio** | Adaptive modulation with link recovery |
| 13 | **DTN** | Bundle protocol with custody transfer |
| 14 | **Mesh Network** | Self-healing mesh topology |
| 15 | **GOAC** | Goal-Oriented Autonomous Controller |
| 16 | **HTN Planner** | Hierarchical Task Network planning |
| 17 | **UA Scheduler** | Utility Accrual scheduling |
| 18 | **ALHAT** | Autonomous Landing Hazard Avoidance Technology |
| 19 | **Space Weather** | Solar radiation monitoring and shielding |
| 20 | **Debris Avoidance** | Orbital debris tracking and collision assessment |

## Project Structure

```
rocket_ai_os/
├── __init__.py          # Package metadata
├── __main__.py          # Entry point (python -m rocket_ai_os)
├── config.py            # System-wide configuration dataclasses
├── main.py              # Integrated demonstration runner
├── core/                # RTOS, Software Bus, DDS middleware
├── gnc/                 # Navigation (EKF), Guidance (G-FOLD), Control
├── propulsion/          # Engine cluster, FTCA, fuel management, anomaly detection
├── fault_tolerance/     # TTEthernet, Simplex, TMR, FDIR
├── comms/               # Cognitive radio, DTN, mesh networking
├── mission/             # GOAC, HTN planner, executive, UA scheduler
├── environment/         # ALHAT, space weather, debris tracking
└── sim/                 # 6-DOF vehicle dynamics, atmosphere, scenarios
test_mission_smoke.py    # Smoke tests for mission management subsystem
```

## Requirements

- **Python** >= 3.10
- **NumPy**

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd paper_fsd

# Install dependencies
pip install numpy
```

## Usage

### Run the full demonstration

```bash
python -m rocket_ai_os
```

This executes all 13 demonstration sections:

1. ARINC 653 Partitioned RTOS initialisation
2. cFS Software Bus + DDS middleware bridge
3. Extended Kalman Filter navigation
4. G-FOLD convex trajectory optimisation
5. Flight control (PID + RL adaptive + Simplex)
6. Propulsion (9-engine cluster + FTCA reconfiguration)
7. Transformer-based anomaly detection
8. Fault tolerance (TTEthernet + TMR + FDIR)
9. Communications (cognitive radio + DTN + mesh)
10. Mission management (GOAC + HTN + UA scheduling)
11. Environmental analysis (ALHAT + weather + debris)
12. Simulation scenarios (nominal, engine-out, sensor degradation)
13. Integrated 6-DOF system loop

### Run the mission smoke tests

```bash
python test_mission_smoke.py
```

Tests the HTN planner, executive, UA scheduler, GOAC, and package imports.

## Configuration

All system parameters are defined as dataclasses in `config.py`:

| Config Class | Purpose |
|---|---|
| `VehicleConfig` | Physical vehicle parameters (mass, engines, geometry) |
| `RTOSConfig` | ARINC 653 partition schedule and criticality levels |
| `NetworkConfig` | TTEthernet timing and redundancy settings |
| `GuidanceConfig` | G-FOLD trajectory constraints |
| `SimConfig` | Physics timestep, gravity, atmosphere model |
| `SystemConfig` | Top-level aggregate of all configs |

## Example Output (abridged)

```
========================================================================
  AUTONOMOUS ROCKET AI OS
  Resilient Autonomy: Next-Gen Avionics for Self-Governing
  Launch Vehicles -- Integrated Demonstration
========================================================================

  1. ARINC 653 Partitioned RTOS
  [OK] Major frame: 100.0 ms, 8 partitions executed

  5. Flight Control (PID + RL Adaptive + Simplex)
  [OK] PID controller: 3-axis with anti-windup
  [OK] RL adaptive: 2-layer neural network (15->64->64->4)

  6. Propulsion (9-Engine Cluster + FTCA)
  [OK] Engine 0 FAILED_OFF -- FTCA reconfigured
       FTCA achieved force error............... 0.000 N

  13. Integrated System Loop (50 steps)
  [OK] Integrated loop completed: 50 steps
       Final altitude..........................   1469.844 m
       Final speed.............................     50.715 m/s

  DEMONSTRATION COMPLETE
       Total elapsed time......................      3.568 s
       Subsystems demonstrated.................         13
```

## License

This project is provided for research and educational purposes.
