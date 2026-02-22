"""System-wide configuration for the Autonomous Rocket AI OS."""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class CriticalityLevel(Enum):
    """DO-178C Design Assurance Levels (DAL)."""
    DAL_A = auto()  # Catastrophic failure condition
    DAL_B = auto()  # Hazardous/Severe-Major
    DAL_C = auto()  # Major
    DAL_D = auto()  # Minor
    DAL_E = auto()  # No effect


class AutonomyLevel(Enum):
    """ESA Autonomy Levels."""
    E1 = auto()  # Execution under ground control
    E2 = auto()  # Execution of pre-planned operations
    E3 = auto()  # Execution of adaptive operations
    E4 = auto()  # Goal-oriented mission execution


class MissionPhase(Enum):
    """Flight phases of a launch vehicle."""
    PRE_LAUNCH = auto()
    LIFTOFF = auto()
    MAX_Q = auto()
    MECO = auto()         # Main Engine Cutoff
    STAGE_SEP = auto()
    SECOND_STAGE = auto()
    SECO = auto()         # Second Engine Cutoff
    COAST = auto()
    ENTRY_BURN = auto()
    AERODYNAMIC_DESCENT = auto()
    LANDING_BURN = auto()
    TERMINAL_LANDING = auto()
    LANDED = auto()
    ABORT = auto()


@dataclass
class VehicleConfig:
    """Physical parameters of the launch vehicle."""
    dry_mass: float = 22_200.0        # kg (first stage dry mass)
    fuel_mass: float = 395_700.0      # kg (propellant mass)
    num_engines: int = 9
    max_thrust_per_engine: float = 845_000.0   # N (sea level)
    min_throttle: float = 0.4
    max_gimbal_angle: float = np.radians(5.0)  # rad
    vehicle_length: float = 47.7      # m
    vehicle_diameter: float = 3.66    # m
    drag_coefficient: float = 0.3
    reference_area: float = 10.52     # m^2
    moment_of_inertia: np.ndarray = field(
        default_factory=lambda: np.diag([1e6, 1e6, 5e4])
    )
    engine_positions: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 0.0],       # Center engine
        [1.5, 0.0, 0.0],       # Ring engines
        [0.75, 1.3, 0.0],
        [-0.75, 1.3, 0.0],
        [-1.5, 0.0, 0.0],
        [-0.75, -1.3, 0.0],
        [0.75, -1.3, 0.0],
        [1.0, 0.65, 0.0],
        [-1.0, 0.65, 0.0],
    ]))

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def max_total_thrust(self) -> float:
        return self.num_engines * self.max_thrust_per_engine


@dataclass
class RTOSConfig:
    """RTOS and partition configuration (ARINC 653 style)."""
    major_frame_ms: float = 100.0         # Major frame period
    partition_schedule: Dict[str, float] = field(default_factory=lambda: {
        "flight_control": 30.0,   # ms - DAL-A, highest priority
        "navigation": 20.0,       # ms - DAL-A
        "guidance": 15.0,         # ms - DAL-B
        "fdir": 10.0,             # ms - DAL-A
        "ai_planner": 10.0,       # ms - DAL-C
        "telemetry": 8.0,         # ms - DAL-D
        "comms": 5.0,             # ms - DAL-C
        "housekeeping": 2.0,      # ms - DAL-E
    })
    partition_criticality: Dict[str, CriticalityLevel] = field(
        default_factory=lambda: {
            "flight_control": CriticalityLevel.DAL_A,
            "navigation": CriticalityLevel.DAL_A,
            "guidance": CriticalityLevel.DAL_B,
            "fdir": CriticalityLevel.DAL_A,
            "ai_planner": CriticalityLevel.DAL_C,
            "telemetry": CriticalityLevel.DAL_D,
            "comms": CriticalityLevel.DAL_C,
            "housekeeping": CriticalityLevel.DAL_E,
        }
    )
    memory_budget_kb: Dict[str, int] = field(default_factory=lambda: {
        "flight_control": 2048,
        "navigation": 4096,
        "guidance": 4096,
        "fdir": 1024,
        "ai_planner": 8192,
        "telemetry": 2048,
        "comms": 2048,
        "housekeeping": 512,
    })


@dataclass
class NetworkConfig:
    """TTEthernet network configuration."""
    tt_cycle_us: float = 500.0           # Time-triggered cycle in microseconds
    rc_bandwidth_mbps: float = 100.0     # Rate-constrained bandwidth
    be_bandwidth_mbps: float = 1000.0    # Best-effort bandwidth
    redundancy: int = 3                  # Triplex redundancy
    guardian_enabled: bool = True


@dataclass
class GuidanceConfig:
    """G-FOLD and trajectory optimization parameters."""
    update_rate_hz: float = 10.0
    glide_slope_angle: float = np.radians(86.0)  # Near-vertical approach
    min_thrust_ratio: float = 0.4
    max_thrust_ratio: float = 0.8
    target_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    target_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0])  # Soft touchdown
    )
    max_tilt_angle: float = np.radians(60.0)


@dataclass
class SimConfig:
    """Simulation configuration."""
    dt: float = 0.01                 # 100 Hz physics
    max_time: float = 120.0          # Max simulation time (s)
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )
    atmosphere_scale_height: float = 8500.0   # m
    sea_level_density: float = 1.225          # kg/m^3
    seed: int = 42


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    rtos: RTOSConfig = field(default_factory=RTOSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    autonomy_level: AutonomyLevel = AutonomyLevel.E4
