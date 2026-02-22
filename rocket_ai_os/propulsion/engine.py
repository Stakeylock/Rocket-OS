"""
Rocket engine model and engine cluster management.

Provides first-order thrust response dynamics, gimbal actuation with rate limits,
throttle constraints, fault injection, and performance degradation modelling.
The EngineCluster aggregates all engines and computes net forces and torques.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from ..config import VehicleConfig


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class EngineHealth(Enum):
    """Health status of a single engine."""
    NOMINAL = auto()
    DEGRADED = auto()       # Reduced performance (e.g., 70-90 % thrust ceiling)
    FAILED_OFF = auto()     # Engine stuck off -- produces no thrust
    FAILED_ON = auto()      # Engine stuck at current throttle -- cannot command
    GIMBAL_STUCK = auto()   # Thrust available but gimbal locked in place


@dataclass
class EngineState:
    """Observable state of a single engine at one time-step."""
    engine_id: int
    health: EngineHealth = EngineHealth.NOMINAL
    throttle: float = 0.0                       # Commanded throttle [0, 1]
    throttle_actual: float = 0.0                # Realised throttle after lag
    gimbal_angles: np.ndarray = field(          # [pitch_gimbal, yaw_gimbal] (rad)
        default_factory=lambda: np.zeros(2)
    )
    gimbal_angles_actual: np.ndarray = field(   # After actuation dynamics
        default_factory=lambda: np.zeros(2)
    )
    thrust_actual: float = 0.0                  # Newtons
    temperature: float = 300.0                  # Kelvin (chamber wall estimate)
    chamber_pressure: float = 0.0               # Pa
    turbopump_rpm: float = 0.0                  # RPM
    specific_impulse: float = 282.0             # s (sea-level default)

    def copy(self) -> "EngineState":
        """Return a deep copy of this state."""
        s = EngineState(
            engine_id=self.engine_id,
            health=self.health,
            throttle=self.throttle,
            throttle_actual=self.throttle_actual,
            gimbal_angles=self.gimbal_angles.copy(),
            gimbal_angles_actual=self.gimbal_angles_actual.copy(),
            thrust_actual=self.thrust_actual,
            temperature=self.temperature,
            chamber_pressure=self.chamber_pressure,
            turbopump_rpm=self.turbopump_rpm,
            specific_impulse=self.specific_impulse,
        )
        return s


# ---------------------------------------------------------------------------
# Single engine physics
# ---------------------------------------------------------------------------

class RocketEngine:
    """High-fidelity model of a single liquid-propellant rocket engine.

    Features
    --------
    * First-order lag on thrust response (tau ~ 50 ms).
    * Gimbal actuation model with configurable rate limit.
    * Throttle constraints (40 % -- 100 % when running).
    * Fault injection API for testing FTCA.
    * Performance degradation (efficiency multiplier).
    """

    # Nominal engine constants
    NOMINAL_ISP_SL: float = 282.0           # s  -- sea-level
    NOMINAL_ISP_VAC: float = 311.0          # s  -- vacuum
    NOMINAL_CHAMBER_PRESSURE: float = 9.7e6 # Pa
    NOMINAL_TURBOPUMP_RPM: float = 36_000.0
    NOMINAL_TEMPERATURE: float = 3_600.0    # K  -- combustion chamber

    def __init__(
        self,
        engine_id: int,
        position: np.ndarray,
        config: VehicleConfig,
        thrust_tau: float = 0.05,
        gimbal_rate_limit: float = np.radians(20.0),   # rad/s
    ):
        self.engine_id = engine_id
        self.position = np.array(position, dtype=np.float64)  # [x, y, z] in body frame
        self.config = config

        # Dynamics parameters
        self.thrust_tau = thrust_tau                # First-order time constant (s)
        self.gimbal_rate_limit = gimbal_rate_limit  # rad/s per axis
        self.max_gimbal = config.max_gimbal_angle   # rad
        self.min_throttle = config.min_throttle
        self.max_thrust = config.max_thrust_per_engine

        # Internal continuous state
        self._throttle_cmd: float = 0.0
        self._throttle_actual: float = 0.0
        self._gimbal_cmd: np.ndarray = np.zeros(2)
        self._gimbal_actual: np.ndarray = np.zeros(2)

        # Degradation & fault state
        self._health: EngineHealth = EngineHealth.NOMINAL
        self._efficiency: float = 1.0          # 1.0 = nominal, <1 = degraded
        self._stuck_throttle: Optional[float] = None
        self._stuck_gimbal: Optional[np.ndarray] = None

        # Derived telemetry
        self._temperature: float = 300.0
        self._chamber_pressure: float = 0.0
        self._turbopump_rpm: float = 0.0
        self._isp: float = self.NOMINAL_ISP_SL

    # --- public command interface ----------------------------------------

    def command_throttle(self, throttle: float) -> None:
        """Set commanded throttle in [0, 1].  0 means engine off."""
        self._throttle_cmd = float(np.clip(throttle, 0.0, 1.0))

    def command_gimbal(self, angles: np.ndarray) -> None:
        """Set commanded gimbal angles [pitch, yaw] in radians."""
        angles = np.asarray(angles, dtype=np.float64)
        self._gimbal_cmd = np.clip(angles, -self.max_gimbal, self.max_gimbal)

    # --- fault injection -------------------------------------------------

    def inject_fault(self, fault: EngineHealth, **kwargs) -> None:
        """Inject a fault for FTCA testing.

        Parameters
        ----------
        fault : EngineHealth
            The fault type to inject.
        kwargs :
            efficiency : float  -- for DEGRADED, sets thrust ceiling (0-1).
        """
        self._health = fault
        if fault == EngineHealth.DEGRADED:
            self._efficiency = kwargs.get("efficiency", 0.75)
        elif fault == EngineHealth.FAILED_OFF:
            self._stuck_throttle = 0.0
            self._efficiency = 0.0
        elif fault == EngineHealth.FAILED_ON:
            # Freeze throttle at current actual value
            self._stuck_throttle = self._throttle_actual
        elif fault == EngineHealth.GIMBAL_STUCK:
            self._stuck_gimbal = self._gimbal_actual.copy()
        elif fault == EngineHealth.NOMINAL:
            self._efficiency = 1.0
            self._stuck_throttle = None
            self._stuck_gimbal = None

    def clear_fault(self) -> None:
        """Restore engine to nominal."""
        self.inject_fault(EngineHealth.NOMINAL)

    # --- dynamics step ---------------------------------------------------

    def step(self, dt: float, altitude: float = 0.0) -> EngineState:
        """Advance the engine model by *dt* seconds.

        Parameters
        ----------
        dt : float
            Integration time-step (s).
        altitude : float
            Vehicle altitude (m) -- used for ISP interpolation.

        Returns
        -------
        EngineState
            Updated observable state after this step.
        """
        # 1. Throttle dynamics
        effective_cmd = self._throttle_cmd
        if self._stuck_throttle is not None:
            effective_cmd = self._stuck_throttle

        # Enforce dead-band: below min_throttle the engine is off
        if effective_cmd > 0.0 and effective_cmd < self.min_throttle:
            effective_cmd = self.min_throttle

        # First-order lag: x_dot = (cmd - x) / tau
        alpha = 1.0 - np.exp(-dt / self.thrust_tau)
        self._throttle_actual += alpha * (effective_cmd - self._throttle_actual)

        # 2. Gimbal dynamics (rate-limited first-order)
        gimbal_cmd = self._gimbal_cmd.copy()
        if self._stuck_gimbal is not None:
            gimbal_cmd = self._stuck_gimbal

        gimbal_error = gimbal_cmd - self._gimbal_actual
        max_delta = self.gimbal_rate_limit * dt
        gimbal_delta = np.clip(gimbal_error, -max_delta, max_delta)
        self._gimbal_actual += gimbal_delta
        self._gimbal_actual = np.clip(
            self._gimbal_actual, -self.max_gimbal, self.max_gimbal
        )

        # 3. Thrust computation
        thrust_magnitude = (
            self._throttle_actual * self.max_thrust * self._efficiency
        )

        # 4. ISP interpolation (simple exponential atmosphere model)
        # Fraction of vacuum: 0 at sea level, ~1 above ~80 km
        vacuum_frac = 1.0 - np.exp(-altitude / 8500.0)
        self._isp = (
            self.NOMINAL_ISP_SL
            + (self.NOMINAL_ISP_VAC - self.NOMINAL_ISP_SL) * vacuum_frac
        )

        # 5. Derived telemetry (linear approximations)
        throttle_frac = self._throttle_actual
        self._chamber_pressure = (
            throttle_frac * self._efficiency * self.NOMINAL_CHAMBER_PRESSURE
        )
        self._turbopump_rpm = (
            throttle_frac * self._efficiency * self.NOMINAL_TURBOPUMP_RPM
        )
        self._temperature = 300.0 + (
            throttle_frac * self._efficiency
            * (self.NOMINAL_TEMPERATURE - 300.0)
        )

        return EngineState(
            engine_id=self.engine_id,
            health=self._health,
            throttle=self._throttle_cmd,
            throttle_actual=self._throttle_actual,
            gimbal_angles=self._gimbal_cmd.copy(),
            gimbal_angles_actual=self._gimbal_actual.copy(),
            thrust_actual=thrust_magnitude,
            temperature=self._temperature,
            chamber_pressure=self._chamber_pressure,
            turbopump_rpm=self._turbopump_rpm,
            specific_impulse=self._isp,
        )

    # --- force / torque helpers ------------------------------------------

    def get_thrust_vector(self) -> np.ndarray:
        """Return 3-D thrust vector in the body frame.

        Convention: nominal thrust is along +Z (up through the vehicle).
        Gimbal pitch rotates about Y, gimbal yaw rotates about X.
        """
        thrust_mag = self._throttle_actual * self.max_thrust * self._efficiency
        if thrust_mag < 1e-3:
            return np.zeros(3)

        pitch = self._gimbal_actual[0]
        yaw = self._gimbal_actual[1]

        # Rotation: small-angle approximation is fine for |angle| < 5 deg,
        # but we use exact trig for correctness.
        fx = thrust_mag * np.sin(yaw)
        fy = -thrust_mag * np.sin(pitch) * np.cos(yaw)
        fz = thrust_mag * np.cos(pitch) * np.cos(yaw)
        return np.array([fx, fy, fz])

    @property
    def health(self) -> EngineHealth:
        return self._health

    @property
    def is_available(self) -> bool:
        """Engine can produce commanded thrust (possibly degraded)."""
        return self._health not in (EngineHealth.FAILED_OFF,)


# ---------------------------------------------------------------------------
# Engine cluster
# ---------------------------------------------------------------------------

class EngineCluster:
    """Manages the full complement of engines on the vehicle stage.

    Responsibilities
    ----------------
    * Expose a uniform command interface (throttle + gimbal per engine).
    * Aggregate net body-frame force and torque about the CoM.
    * Provide engine health monitoring for FDIR.
    """

    def __init__(self, config: VehicleConfig):
        self.config = config
        self.engines: List[RocketEngine] = []
        for i in range(config.num_engines):
            pos = config.engine_positions[i]
            self.engines.append(RocketEngine(engine_id=i, position=pos, config=config))

        # Reference CoM offset along Z from engine plane (updated by FuelManager)
        self._com_offset: np.ndarray = np.array([0.0, 0.0, 0.0])

    @property
    def num_engines(self) -> int:
        return len(self.engines)

    # --- command interface -----------------------------------------------

    def command_all(
        self,
        throttles: np.ndarray,
        gimbals: np.ndarray,
    ) -> None:
        """Command all engines simultaneously.

        Parameters
        ----------
        throttles : (N,) array
            Throttle command per engine in [0, 1].
        gimbals : (N, 2) array
            Gimbal commands [pitch, yaw] per engine in radians.
        """
        throttles = np.asarray(throttles)
        gimbals = np.asarray(gimbals)
        for i, eng in enumerate(self.engines):
            eng.command_throttle(float(throttles[i]))
            eng.command_gimbal(gimbals[i])

    def step(self, dt: float, altitude: float = 0.0) -> List[EngineState]:
        """Step all engines forward by *dt* and return their states."""
        states = []
        for eng in self.engines:
            states.append(eng.step(dt, altitude))
        return states

    # --- aggregation -----------------------------------------------------

    def set_com_offset(self, offset: np.ndarray) -> None:
        """Update the centre-of-mass position in the body frame.

        The offset is used to compute torque arms: r = engine_pos - com.
        """
        self._com_offset = np.asarray(offset, dtype=np.float64)

    def get_total_force_and_torque(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute net body-frame force and torque about the current CoM.

        Returns
        -------
        force : (3,) ndarray -- total force in body frame (N).
        torque : (3,) ndarray -- total torque about CoM (N*m).
        """
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        for eng in self.engines:
            f = eng.get_thrust_vector()
            total_force += f
            r = eng.position - self._com_offset
            total_torque += np.cross(r, f)

        return total_force, total_torque

    # --- health monitoring -----------------------------------------------

    def get_health_summary(self) -> Dict[int, EngineHealth]:
        """Return a dict mapping engine_id -> EngineHealth."""
        return {eng.engine_id: eng.health for eng in self.engines}

    def get_available_engine_ids(self) -> List[int]:
        """Return list of engine IDs that can still produce commanded thrust."""
        return [eng.engine_id for eng in self.engines if eng.is_available]

    def get_engine(self, engine_id: int) -> RocketEngine:
        return self.engines[engine_id]
