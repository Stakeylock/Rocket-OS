"""GNC (Guidance, Navigation, Control) subsystem package.

Provides the core flight software components for autonomous rocket
powered-descent and landing:

- **navigation** -- State estimation via Extended Kalman Filter fusing
  IMU and GPS sensor data.
- **guidance** -- Fuel-optimal trajectory planning using G-FOLD
  (lossless convexification).
- **control** -- Flight control with PID baseline, RL-adaptive
  corrections, and Simplex-architecture safety switching.
"""

# Navigation
from rocket_ai_os.gnc.navigation import (
    NavigationState,
    IMUSensor,
    GPSSensor,
    ExtendedKalmanFilter,
    NavigationSystem,
)

# Guidance
from rocket_ai_os.gnc.guidance import (
    TrajectoryPoint,
    GFOLDSolver,
    GuidanceSystem,
)

# Control
from rocket_ai_os.gnc.control import (
    PIDController,
    RLAdaptiveController,
    SimplexControlSwitch,
    ControlCommand,
    FlightController,
)

__all__ = [
    # Navigation
    "NavigationState",
    "IMUSensor",
    "GPSSensor",
    "ExtendedKalmanFilter",
    "NavigationSystem",
    # Guidance
    "TrajectoryPoint",
    "GFOLDSolver",
    "GuidanceSystem",
    # Control
    "PIDController",
    "RLAdaptiveController",
    "SimplexControlSwitch",
    "ControlCommand",
    "FlightController",
]
