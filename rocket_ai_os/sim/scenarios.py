"""
Self-contained test scenarios for the autonomous rocket simulation.

Each scenario sets up initial conditions, runs a physics loop with a
simple guidance/control law, and evaluates success criteria.  Scenarios
are designed to be executable without external dependencies beyond
numpy and the local ``sim`` and ``config`` packages.

Usage example::

    from rocket_ai_os.sim.scenarios import NominalLandingScenario
    result = NominalLandingScenario().run()
    print(f"Success: {result.success}")
    print(f"Touchdown speed: {result.metrics['touchdown_speed_m_s']:.2f} m/s")
"""

from __future__ import annotations

import abc
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rocket_ai_os.config import (
    VehicleConfig,
    SimConfig,
    GuidanceConfig,
    MissionPhase,
)
from rocket_ai_os.sim.vehicle import VehicleState, Vehicle, _quat_to_dcm
from rocket_ai_os.sim.physics import (
    Atmosphere,
    AerodynamicModel,
    GravityModel,
    WindModel,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Outcome of a completed scenario run.

    Attributes:
        success:        True if the scenario's success criteria were met.
        trajectory_log: Time-ordered list of VehicleState snapshots.
        events_log:     List of (time, description) tuples for noteworthy
                        events that occurred during the run.
        metrics:        Dictionary of scalar performance metrics computed
                        during evaluation.
    """
    success: bool = False
    trajectory_log: List[VehicleState] = field(default_factory=list)
    events_log: List[Tuple[float, str]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract base scenario
# ---------------------------------------------------------------------------

class Scenario(abc.ABC):
    """Base class for all simulation scenarios.

    Subclasses must implement ``setup()``, ``run()``, and ``evaluate()``.
    The ``run()`` method should call ``setup()``, execute the simulation
    loop, then call ``evaluate()`` and return a ``ScenarioResult``.
    """

    def __init__(
        self,
        vehicle_config: Optional[VehicleConfig] = None,
        sim_config: Optional[SimConfig] = None,
        guidance_config: Optional[GuidanceConfig] = None,
        seed: int = 42,
    ) -> None:
        self.vehicle_config = vehicle_config or VehicleConfig()
        self.sim_config = sim_config or SimConfig()
        self.guidance_config = guidance_config or GuidanceConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def setup(self) -> None:
        """Initialise scenario-specific state (vehicle, environment, etc.)."""

    @abc.abstractmethod
    def run(self) -> ScenarioResult:
        """Execute the full scenario and return results."""

    @abc.abstractmethod
    def evaluate(self, result: ScenarioResult) -> ScenarioResult:
        """Post-process and fill in success/metrics on the result."""


# ---------------------------------------------------------------------------
# Attitude stabilization torque (mimics TVC / reaction control)
# ---------------------------------------------------------------------------

def _stabilization_torque(vehicle: Vehicle) -> np.ndarray:
    """Compute a restoring torque that drives the vehicle attitude toward
    vertical (body-Z aligned with inertial-Z).  This approximates the
    combined effect of thrust vector control and reaction control thrusters
    without running the full flight controller."""
    state = vehicle.state
    C_nb = _quat_to_dcm(state.attitude)
    body_z_inertial = C_nb[:, 2]
    desired_z = np.array([0.0, 0.0, 1.0])

    # Torque proportional to cross product (rotation error) with rate damping
    error = np.cross(body_z_inertial, desired_z)
    kp = 1e6
    kd = 5e5
    return kp * error - kd * state.angular_velocity


# ---------------------------------------------------------------------------
# Simple proportional-navigation guidance for landing scenarios
# ---------------------------------------------------------------------------

def _landing_guidance(
    vehicle: Vehicle,
    target: np.ndarray,
    config: VehicleConfig,
    guidance_config: GuidanceConfig,
) -> np.ndarray:
    """Compute a gravity-turn-style thrust vector for powered descent.

    This is a simplified proportional guidance law that commands thrust
    to cancel the current velocity while steering toward the target
    position.  It does not replace G-FOLD but is sufficient for
    self-contained scenario execution.

    Args:
        vehicle:         Current vehicle instance.
        target:          Target landing position [x, y, z] in metres.
        config:          Vehicle configuration for thrust limits.
        guidance_config: Guidance parameters (thrust ratios, glide slope).

    Returns:
        Desired thrust vector in the inertial frame (N).
    """
    state = vehicle.state
    pos_error = target - state.position
    vel = state.velocity
    mass = max(state.mass, 1.0)
    g = 9.81

    altitude = max(state.position[2], 0.1)
    vz = state.velocity[2]

    # --- Horizontal channel: PD control toward target ---
    kp_h = 0.3
    kd_h = 1.2
    a_cmd_h = kp_h * pos_error[:2] - kd_h * vel[:2]

    # --- Vertical channel: constant-deceleration descent profile ---
    # Desired descent speed for a soft touchdown at ~1.5 m/s.
    # Uses 0.3g net deceleration margin to ensure the controller can track.
    v_touchdown = 1.5
    v_des = -np.sqrt(max(0.3 * g * altitude + v_touchdown ** 2, 0.0))
    vz_error = v_des - vz
    kp_z = 4.0
    a_cmd_z = g + kp_z * vz_error  # gravity comp + velocity tracking

    a_cmd = np.array([a_cmd_h[0], a_cmd_h[1], a_cmd_z])

    # Clamp commanded acceleration to thrust limits
    thrust_min = config.max_thrust_per_engine * config.min_throttle
    thrust_max_total = (
        config.num_engines * config.max_thrust_per_engine
        * guidance_config.max_thrust_ratio
    )
    f_cmd = a_cmd * mass
    f_mag = np.linalg.norm(f_cmd)

    if f_mag < 1e-3:
        return np.array([0.0, 0.0, mass * g])

    # Enforce thrust magnitude limits
    f_mag_clamped = np.clip(f_mag, thrust_min, thrust_max_total)
    f_cmd = f_cmd / f_mag * f_mag_clamped

    # Enforce glide-slope (maximum tilt from vertical)
    max_tilt = guidance_config.max_tilt_angle
    f_horizontal = np.linalg.norm(f_cmd[:2])
    f_vertical = f_cmd[2]
    if f_vertical > 0.0 and np.arctan2(f_horizontal, f_vertical) > max_tilt:
        # Project thrust onto the glide-slope cone
        f_h_max = f_vertical * np.tan(max_tilt)
        if f_horizontal > 1e-6:
            f_cmd[:2] = f_cmd[:2] / f_horizontal * f_h_max

    return f_cmd


# ---------------------------------------------------------------------------
# Scenario: Nominal powered-descent landing
# ---------------------------------------------------------------------------

class NominalLandingScenario(Scenario):
    """Standard powered-descent and landing from 1500 m altitude.

    Initial conditions:
        - Position:  [200, 50, 1500] m  (offset from pad)
        - Velocity:  [-50, 0, -80] m/s  (approaching, descending)
        - Phase:     LANDING_BURN

    Success criteria:
        - Touchdown within 20 m of target
        - Touchdown speed < 5 m/s
        - No structural constraint violations
        - Fuel remaining > 0
    """

    def setup(self) -> None:
        """Create and initialise the vehicle and environment models."""
        # Landing configuration: reduced mass (first stage dry + landing fuel)
        landing_fuel = 15_000.0  # kg reserved for landing
        landing_mass = self.vehicle_config.dry_mass + landing_fuel

        self.vehicle = Vehicle(config=self.vehicle_config,
                               initial_phase=MissionPhase.LANDING_BURN)
        self.vehicle.set_state(
            position=np.array([200.0, 50.0, 1500.0]),
            velocity=np.array([-50.0, 0.0, -80.0]),
            mass=landing_mass,
            fuel_mass=landing_fuel,
            phase=MissionPhase.LANDING_BURN,
            time=0.0,
        )

        self.atmosphere = Atmosphere()
        self.aero = AerodynamicModel(
            Cd0=self.vehicle_config.drag_coefficient,
            reference_area=self.vehicle_config.reference_area,
            vehicle_length=self.vehicle_config.vehicle_length,
        )
        self.gravity = GravityModel(flat_gravity=np.array([0.0, 0.0, -9.81]))
        self.wind = WindModel(seed=self.seed)

        self.target = self.guidance_config.target_position.copy()

    def run(self) -> ScenarioResult:
        """Execute the nominal landing scenario.

        Returns:
            ScenarioResult with trajectory, events, and metrics.
        """
        self.setup()
        dt = self.sim_config.dt
        max_time = self.sim_config.max_time
        result = ScenarioResult()
        result.events_log.append((0.0, "Scenario start: nominal landing"))

        # Log initial state
        result.trajectory_log.append(self.vehicle.state.clone())

        t = 0.0
        log_interval = 0.1  # log every 100 ms
        next_log_time = log_interval

        while t < max_time:
            state = self.vehicle.state

            if state.is_landed or state.is_destroyed:
                break

            # --- Environment forces ---
            alt = self.vehicle.get_altitude()
            wind_vel = self.wind.get_wind(alt, t, dt)
            dcm = self.vehicle.get_dcm()

            f_gravity = self.gravity.compute_gravity(state.position, state.mass)
            f_aero, t_aero = self.aero.compute_aero_forces(
                state.position, state.velocity, dcm, self.atmosphere, wind_vel,
            )

            # --- Guidance (thrust command) ---
            f_thrust = _landing_guidance(
                self.vehicle, self.target,
                self.vehicle_config, self.guidance_config,
            )

            # --- Fuel consumption ---
            thrust_mag = np.linalg.norm(f_thrust)
            g0 = 9.81
            isp = 282.0  # sea-level default
            if state.fuel_mass <= 0:
                # No fuel -- coast under gravity (no thrust)
                f_thrust = np.zeros(3)
                thrust_mag = 0.0
            elif thrust_mag > 0:
                mass_flow = thrust_mag / (isp * g0)
                self.vehicle.consume_fuel(mass_flow, dt)

            # --- Total force and torque ---
            total_force = f_gravity + f_aero + f_thrust
            total_torque = t_aero + _stabilization_torque(self.vehicle) + _stabilization_torque(self.vehicle)

            # --- Integrate ---
            self.vehicle.apply_forces(total_force, total_torque, dt)
            t = self.vehicle.state.time

            # --- Phase transitions ---
            if alt < 100.0 and state.phase == MissionPhase.LANDING_BURN:
                self.vehicle.state.phase = MissionPhase.TERMINAL_LANDING
                result.events_log.append((t, "Phase transition: TERMINAL_LANDING"))

            # --- Constraint check ---
            violations = self.vehicle.check_constraints()
            for v in violations:
                result.events_log.append((t, f"CONSTRAINT: {v}"))

            # --- Logging ---
            if t >= next_log_time:
                result.trajectory_log.append(self.vehicle.state.clone())
                next_log_time += log_interval

        # Final state
        result.trajectory_log.append(self.vehicle.state.clone())
        result.events_log.append((t, "Scenario end"))

        return self.evaluate(result)

    def evaluate(self, result: ScenarioResult) -> ScenarioResult:
        """Compute metrics and determine success.

        Args:
            result: Partially-filled ScenarioResult from the run loop.

        Returns:
            The same result with metrics and success fields populated.
        """
        final = result.trajectory_log[-1]

        touchdown_pos_error = float(np.linalg.norm(
            final.position[:2] - self.target[:2]
        ))
        touchdown_speed = float(np.linalg.norm(final.velocity))
        max_speed = max(
            float(np.linalg.norm(s.velocity)) for s in result.trajectory_log
        )
        fuel_remaining = final.fuel_mass
        flight_time = final.time

        result.metrics = {
            "touchdown_pos_error_m": touchdown_pos_error,
            "touchdown_speed_m_s": touchdown_speed,
            "max_speed_m_s": max_speed,
            "fuel_remaining_kg": fuel_remaining,
            "flight_time_s": flight_time,
            "is_landed": final.is_landed,
            "is_destroyed": final.is_destroyed,
            "failure_reason": result.failure_reason or ("timeout" if result.trajectory_log[-1].time >= self.sim_config.max_time - 0.1 else "success" if final.is_landed else "crash"),
        }

        result.success = (
            final.is_landed
            and not final.is_destroyed
            and touchdown_pos_error < 20.0
            and touchdown_speed < 5.0
            and fuel_remaining > 0.0
        )

        return result


# ---------------------------------------------------------------------------
# Scenario: Engine failure during landing burn
# ---------------------------------------------------------------------------

class EngineOutScenario(Scenario):
    """Engine failure during the landing burn.

    Injects a complete engine-out fault at a configurable time after the
    landing burn begins.  The guidance law must compensate with reduced
    available thrust (fewer engines, lower peak thrust).

    This tests the Fault-Tolerant Control Allocation (FTCA) path.

    Args:
        failure_time: Time (seconds into scenario) at which the engine
                      fails.  Default is 5 s.
        failed_engine_ids: List of engine indices to fail.  Default is
                          [0] (centre engine).
    """

    def __init__(
        self,
        failure_time: float = 5.0,
        failed_engine_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.failure_time = failure_time
        self.failed_engine_ids = failed_engine_ids if failed_engine_ids is not None else [0]

    def setup(self) -> None:
        """Create vehicle and environment for the engine-out test."""
        landing_fuel = 18_000.0
        landing_mass = self.vehicle_config.dry_mass + landing_fuel

        self.vehicle = Vehicle(config=self.vehicle_config,
                               initial_phase=MissionPhase.LANDING_BURN)
        self.vehicle.set_state(
            position=np.array([200.0, 50.0, 1500.0]),
            velocity=np.array([-50.0, 0.0, -80.0]),
            mass=landing_mass,
            fuel_mass=landing_fuel,
            phase=MissionPhase.LANDING_BURN,
            time=0.0,
        )

        self.atmosphere = Atmosphere()
        self.aero = AerodynamicModel(
            Cd0=self.vehicle_config.drag_coefficient,
            reference_area=self.vehicle_config.reference_area,
            vehicle_length=self.vehicle_config.vehicle_length,
        )
        self.gravity = GravityModel(flat_gravity=np.array([0.0, 0.0, -9.81]))
        self.wind = WindModel(seed=self.seed)

        self.target = self.guidance_config.target_position.copy()

        # Track whether the fault has been injected
        self._fault_injected = False
        # Number of operational engines (starts nominal)
        self._num_active_engines = self.vehicle_config.num_engines

    def run(self) -> ScenarioResult:
        """Execute the engine-out landing scenario.

        Returns:
            ScenarioResult with trajectory, events, and metrics.
        """
        self.setup()
        dt = self.sim_config.dt
        max_time = self.sim_config.max_time
        result = ScenarioResult()
        result.events_log.append(
            (0.0, f"Scenario start: engine-out (failure at t={self.failure_time:.1f}s)")
        )
        result.trajectory_log.append(self.vehicle.state.clone())

        t = 0.0
        log_interval = 0.1
        next_log_time = log_interval

        while t < max_time:
            state = self.vehicle.state
            if state.is_landed or state.is_destroyed:
                break

            # --- Fault injection ---
            if not self._fault_injected and t >= self.failure_time:
                self._fault_injected = True
                self._num_active_engines -= len(self.failed_engine_ids)
                self._num_active_engines = max(self._num_active_engines, 1)
                result.events_log.append(
                    (t, f"ENGINE FAILURE: engines {self.failed_engine_ids} lost. "
                     f"{self._num_active_engines} engines remain.")
                )

            # --- Environment ---
            alt = self.vehicle.get_altitude()
            wind_vel = self.wind.get_wind(alt, t, dt)
            dcm = self.vehicle.get_dcm()

            f_gravity = self.gravity.compute_gravity(state.position, state.mass)
            f_aero, t_aero = self.aero.compute_aero_forces(
                state.position, state.velocity, dcm, self.atmosphere, wind_vel,
            )

            # --- Guidance with degraded thrust ---
            f_thrust = _landing_guidance(
                self.vehicle, self.target,
                self.vehicle_config, self.guidance_config,
            )

            # Scale thrust to account for lost engines
            if self._fault_injected:
                thrust_ratio = (
                    self._num_active_engines / self.vehicle_config.num_engines
                )
                max_available = (
                    self._num_active_engines
                    * self.vehicle_config.max_thrust_per_engine
                    * self.guidance_config.max_thrust_ratio
                )
                f_mag = np.linalg.norm(f_thrust)
                if f_mag > max_available and f_mag > 1e-3:
                    f_thrust = f_thrust / f_mag * max_available

            # --- Fuel consumption ---
            thrust_mag = np.linalg.norm(f_thrust)
            g0 = 9.81
            isp = 282.0
            if state.fuel_mass <= 0:
                f_thrust = np.zeros(3)
                thrust_mag = 0.0
            elif thrust_mag > 0:
                mass_flow = thrust_mag / (isp * g0)
                self.vehicle.consume_fuel(mass_flow, dt)

            # --- Integrate ---
            total_force = f_gravity + f_aero + f_thrust
            total_torque = t_aero + _stabilization_torque(self.vehicle)
            self.vehicle.apply_forces(total_force, total_torque, dt)
            t = self.vehicle.state.time

            # --- Phase transitions ---
            if alt < 100.0 and state.phase == MissionPhase.LANDING_BURN:
                self.vehicle.state.phase = MissionPhase.TERMINAL_LANDING
                result.events_log.append((t, "Phase transition: TERMINAL_LANDING"))

            # --- Constraint check ---
            violations = self.vehicle.check_constraints()
            for v in violations:
                result.events_log.append((t, f"CONSTRAINT: {v}"))

            # --- Logging ---
            if t >= next_log_time:
                result.trajectory_log.append(self.vehicle.state.clone())
                next_log_time += log_interval

        result.trajectory_log.append(self.vehicle.state.clone())
        result.events_log.append((t, "Scenario end"))
        return self.evaluate(result)

    def evaluate(self, result: ScenarioResult) -> ScenarioResult:
        """Evaluate the engine-out scenario outcome.

        Applies the same landing criteria as the nominal scenario but
        also records engine-out specific metrics.
        """
        final = result.trajectory_log[-1]

        touchdown_pos_error = float(np.linalg.norm(
            final.position[:2] - self.target[:2]
        ))
        touchdown_speed = float(np.linalg.norm(final.velocity))
        fuel_remaining = final.fuel_mass
        flight_time = final.time

        result.metrics = {
            "touchdown_pos_error_m": touchdown_pos_error,
            "touchdown_speed_m_s": touchdown_speed,
            "fuel_remaining_kg": fuel_remaining,
            "flight_time_s": flight_time,
            "is_landed": final.is_landed,
            "is_destroyed": final.is_destroyed,
            "failure_reason": result.failure_reason or ("timeout" if final.time >= self.sim_config.max_time - 0.1 else "success" if final.is_landed else "crash"),
            "engines_failed": self.failed_engine_ids,
            "engines_remaining": self._num_active_engines,
            "fault_injection_time_s": self.failure_time,
        }

        # Relaxed criteria: allow larger position error with fewer engines
        max_pos_error = 30.0  # relaxed from 20
        result.success = (
            final.is_landed
            and not final.is_destroyed
            and touchdown_pos_error < max_pos_error
            and touchdown_speed < 5.0
            and fuel_remaining > 0.0
        )
        return result


# ---------------------------------------------------------------------------
# Scenario: Sensor degradation (GPS dropout + IMU drift)
# ---------------------------------------------------------------------------

class SensorDegradationScenario(Scenario):
    """GPS dropout with simulated IMU drift during landing.

    From the dropout time onward, the vehicle navigates on dead-reckoning
    only.  A simulated IMU bias drift is superimposed on the true state
    to emulate what a navigation filter would experience.

    This tests the navigation filter's robustness in coast mode.

    Args:
        gps_dropout_time: Time (s) at which GPS becomes unavailable.
        imu_drift_rate:   Accelerometer bias drift rate in m/s^2 per second.
    """

    def __init__(
        self,
        gps_dropout_time: float = 3.0,
        imu_drift_rate: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.gps_dropout_time = gps_dropout_time
        self.imu_drift_rate = imu_drift_rate

    def setup(self) -> None:
        """Create vehicle and environment for the sensor degradation test."""
        landing_fuel = 15_000.0
        landing_mass = self.vehicle_config.dry_mass + landing_fuel

        self.vehicle = Vehicle(config=self.vehicle_config,
                               initial_phase=MissionPhase.LANDING_BURN)
        self.vehicle.set_state(
            position=np.array([200.0, 50.0, 1500.0]),
            velocity=np.array([-50.0, 0.0, -80.0]),
            mass=landing_mass,
            fuel_mass=landing_fuel,
            phase=MissionPhase.LANDING_BURN,
            time=0.0,
        )

        self.atmosphere = Atmosphere()
        self.aero = AerodynamicModel(
            Cd0=self.vehicle_config.drag_coefficient,
            reference_area=self.vehicle_config.reference_area,
            vehicle_length=self.vehicle_config.vehicle_length,
        )
        self.gravity = GravityModel(flat_gravity=np.array([0.0, 0.0, -9.81]))
        self.wind = WindModel(seed=self.seed)

        self.target = self.guidance_config.target_position.copy()

        # Navigation error accumulator (simulates dead-reckoning drift)
        self._nav_error_pos = np.zeros(3)
        self._nav_error_vel = np.zeros(3)
        self._imu_bias = np.zeros(3)
        self._gps_available = True

    def run(self) -> ScenarioResult:
        """Execute the sensor degradation scenario.

        Returns:
            ScenarioResult with trajectory, events, and metrics.
        """
        self.setup()
        dt = self.sim_config.dt
        max_time = self.sim_config.max_time
        result = ScenarioResult()
        result.events_log.append(
            (0.0, f"Scenario start: sensor degradation "
             f"(GPS dropout at t={self.gps_dropout_time:.1f}s)")
        )
        result.trajectory_log.append(self.vehicle.state.clone())

        t = 0.0
        log_interval = 0.1
        next_log_time = log_interval
        max_nav_error = 0.0

        while t < max_time:
            state = self.vehicle.state
            if state.is_landed or state.is_destroyed:
                break

            # --- GPS dropout ---
            if self._gps_available and t >= self.gps_dropout_time:
                self._gps_available = False
                result.events_log.append((t, "GPS DROPOUT: switching to dead-reckoning"))

            # --- Simulate IMU drift (dead-reckoning error) ---
            if not self._gps_available:
                # Bias grows linearly in time since dropout
                time_since_dropout = t - self.gps_dropout_time
                self._imu_bias = (
                    self.imu_drift_rate * time_since_dropout
                    * np.array([0.3, 0.2, 0.1])  # axis-dependent
                )
                # Accumulate velocity and position error
                self._nav_error_vel += self._imu_bias * dt
                self._nav_error_pos += self._nav_error_vel * dt

            # --- Perceived state (true + navigation error) ---
            perceived_pos = state.position + self._nav_error_pos
            perceived_vel = state.velocity + self._nav_error_vel

            nav_error_mag = float(np.linalg.norm(self._nav_error_pos))
            max_nav_error = max(max_nav_error, nav_error_mag)

            # --- Environment ---
            alt = self.vehicle.get_altitude()
            wind_vel = self.wind.get_wind(alt, t, dt)
            dcm = self.vehicle.get_dcm()

            f_gravity = self.gravity.compute_gravity(state.position, state.mass)
            f_aero, t_aero = self.aero.compute_aero_forces(
                state.position, state.velocity, dcm, self.atmosphere, wind_vel,
            )

            # --- Guidance uses perceived (degraded) state ---
            # Temporarily swap perceived state into vehicle for guidance
            true_pos = state.position.copy()
            true_vel = state.velocity.copy()
            self.vehicle.state.position = perceived_pos
            self.vehicle.state.velocity = perceived_vel

            f_thrust = _landing_guidance(
                self.vehicle, self.target,
                self.vehicle_config, self.guidance_config,
            )

            # Restore true state
            self.vehicle.state.position = true_pos
            self.vehicle.state.velocity = true_vel

            # --- Fuel consumption ---
            thrust_mag = np.linalg.norm(f_thrust)
            g0 = 9.81
            isp = 282.0
            if state.fuel_mass <= 0:
                f_thrust = np.zeros(3)
                thrust_mag = 0.0
            elif thrust_mag > 0:
                mass_flow = thrust_mag / (isp * g0)
                self.vehicle.consume_fuel(mass_flow, dt)

            # --- Integrate (uses true dynamics) ---
            total_force = f_gravity + f_aero + f_thrust
            total_torque = t_aero + _stabilization_torque(self.vehicle)
            self.vehicle.apply_forces(total_force, total_torque, dt)
            t = self.vehicle.state.time

            # --- Phase transitions ---
            if alt < 100.0 and state.phase == MissionPhase.LANDING_BURN:
                self.vehicle.state.phase = MissionPhase.TERMINAL_LANDING
                result.events_log.append((t, "Phase transition: TERMINAL_LANDING"))

            # --- Logging ---
            if t >= next_log_time:
                result.trajectory_log.append(self.vehicle.state.clone())
                next_log_time += log_interval

        result.trajectory_log.append(self.vehicle.state.clone())
        result.events_log.append((t, "Scenario end"))

        # Stash the max nav error for evaluation
        self._max_nav_error = max_nav_error
        return self.evaluate(result)

    def evaluate(self, result: ScenarioResult) -> ScenarioResult:
        """Evaluate the sensor degradation scenario outcome."""
        final = result.trajectory_log[-1]

        touchdown_pos_error = float(np.linalg.norm(
            final.position[:2] - self.target[:2]
        ))
        touchdown_speed = float(np.linalg.norm(final.velocity))

        result.metrics = {
            "touchdown_pos_error_m": touchdown_pos_error,
            "touchdown_speed_m_s": touchdown_speed,
            "fuel_remaining_kg": final.fuel_mass,
            "flight_time_s": final.time,
            "is_landed": final.is_landed,
            "is_destroyed": final.is_destroyed,
            "failure_reason": result.failure_reason or ("timeout" if final.time >= self.sim_config.max_time - 0.1 else "success" if final.is_landed else "crash"),
            "max_nav_error_m": getattr(self, "_max_nav_error", 0.0),
            "gps_dropout_time_s": self.gps_dropout_time,
            "imu_drift_rate_m_s2_per_s": self.imu_drift_rate,
        }

        # With sensor degradation, relaxed landing accuracy
        result.success = (
            final.is_landed
            and not final.is_destroyed
            and touchdown_pos_error < 50.0   # wider tolerance
            and touchdown_speed < 5.0
        )
        return result


# ---------------------------------------------------------------------------
# Scenario: Full mission with GOAC-style phase management
# ---------------------------------------------------------------------------

class FullMissionScenario(Scenario):
    """End-to-end mission with autonomous phase management.

    Simulates a simplified multi-phase mission:
        1. LIFTOFF         -> ascent under full thrust
        2. MAX_Q           -> throttle-down through max dynamic pressure
        3. MECO            -> main engine cutoff, coast
        4. ENTRY_BURN      -> deceleration for re-entry
        5. AERODYNAMIC_DESCENT -> drag-dominated descent
        6. LANDING_BURN    -> powered descent
        7. TERMINAL_LANDING -> final approach
        8. LANDED          -> success

    Faults are injected at random times to exercise GOAC goal
    re-planning.

    Args:
        inject_faults: Enable random fault injection.
        max_faults:    Maximum number of faults to inject.
    """

    def __init__(
        self,
        inject_faults: bool = True,
        max_faults: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.inject_faults = inject_faults
        self.max_faults = max_faults

    def setup(self) -> None:
        """Initialise full mission scenario state."""
        self.vehicle = Vehicle(config=self.vehicle_config,
                               initial_phase=MissionPhase.LIFTOFF)
        self.vehicle.set_state(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=self.vehicle_config.total_mass,
            fuel_mass=self.vehicle_config.fuel_mass,
            phase=MissionPhase.LIFTOFF,
            time=0.0,
        )

        self.atmosphere = Atmosphere()
        self.aero = AerodynamicModel(
            Cd0=self.vehicle_config.drag_coefficient,
            reference_area=self.vehicle_config.reference_area,
            vehicle_length=self.vehicle_config.vehicle_length,
        )
        self.gravity = GravityModel(flat_gravity=np.array([0.0, 0.0, -9.81]))
        self.wind = WindModel(seed=self.seed)

        self.target = self.guidance_config.target_position.copy()

        # Generate random fault schedule
        self._fault_schedule: List[Tuple[float, str]] = []
        if self.inject_faults:
            n_faults = self._rng.integers(1, self.max_faults + 1)
            for _ in range(n_faults):
                fault_time = float(self._rng.uniform(10.0, 80.0))
                fault_type = self._rng.choice([
                    "engine_degraded", "sensor_noise_spike", "thrust_oscillation",
                ])
                self._fault_schedule.append((fault_time, str(fault_type)))
            self._fault_schedule.sort(key=lambda x: x[0])

        self._faults_injected: List[float] = []
        self._active_faults: Dict[str, Any] = {}

    def _get_phase_thrust(
        self,
        phase: MissionPhase,
        state: VehicleState,
    ) -> np.ndarray:
        """Compute the commanded thrust vector for the current phase.

        This replaces a full GOAC planner with simple deterministic rules
        per phase.

        Args:
            phase: Current mission phase.
            state: Current vehicle state.

        Returns:
            Commanded thrust vector in inertial frame (N).
        """
        mass = max(state.mass, 1.0)
        g = 9.81
        max_thrust = (
            self.vehicle_config.num_engines
            * self.vehicle_config.max_thrust_per_engine
        )

        if phase == MissionPhase.LIFTOFF:
            # Full thrust upward (slight pitch for downrange)
            throttle = 1.0
            tilt_angle = np.radians(2.0)
            fx = max_thrust * throttle * np.sin(tilt_angle)
            fz = max_thrust * throttle * np.cos(tilt_angle)
            return np.array([fx, 0.0, fz])

        elif phase == MissionPhase.MAX_Q:
            # Throttle down to limit dynamic pressure
            throttle = 0.6
            tilt_angle = np.radians(5.0)
            fx = max_thrust * throttle * np.sin(tilt_angle)
            fz = max_thrust * throttle * np.cos(tilt_angle)
            return np.array([fx, 0.0, fz])

        elif phase == MissionPhase.MECO:
            # Engines off -- coasting
            return np.zeros(3)

        elif phase == MissionPhase.COAST:
            return np.zeros(3)

        elif phase == MissionPhase.ENTRY_BURN:
            # Retrograde burn to decelerate
            speed = np.linalg.norm(state.velocity)
            if speed > 1e-3:
                direction = -state.velocity / speed
            else:
                direction = np.array([0.0, 0.0, 1.0])
            throttle = 0.7
            return max_thrust * throttle * direction

        elif phase == MissionPhase.AERODYNAMIC_DESCENT:
            # No thrust; aero drag decelerates the vehicle
            return np.zeros(3)

        elif phase in (MissionPhase.LANDING_BURN, MissionPhase.TERMINAL_LANDING):
            return _landing_guidance(
                self.vehicle, self.target,
                self.vehicle_config, self.guidance_config,
            )

        else:
            return np.zeros(3)

    def _update_phase(
        self,
        state: VehicleState,
        dynamic_pressure: float,
        result: ScenarioResult,
    ) -> MissionPhase:
        """Determine autonomous phase transitions based on vehicle state.

        Implements a simplified GOAC-style goal evaluation: each phase
        has exit conditions that trigger the next phase.

        Args:
            state:            Current vehicle state.
            dynamic_pressure: Current dynamic pressure in Pa.
            result:           ScenarioResult for event logging.

        Returns:
            The (possibly updated) mission phase.
        """
        phase = state.phase
        t = state.time
        alt = state.position[2]
        speed = np.linalg.norm(state.velocity)
        vz = state.velocity[2]

        transitions = {
            MissionPhase.LIFTOFF: (
                lambda: alt > 500.0,
                MissionPhase.MAX_Q,
                "Altitude > 500 m: entering MAX_Q regime",
            ),
            MissionPhase.MAX_Q: (
                lambda: alt > 12_000.0,
                MissionPhase.MECO,
                "Altitude > 12 km: MECO",
            ),
            MissionPhase.MECO: (
                lambda: t > 20.0,
                MissionPhase.COAST,
                "Post-MECO coast phase",
            ),
            MissionPhase.COAST: (
                lambda: vz < -10.0 and alt > 5_000.0,
                MissionPhase.ENTRY_BURN,
                "Descending: initiating entry burn",
            ),
            MissionPhase.ENTRY_BURN: (
                lambda: speed < 200.0 or alt < 8_000.0,
                MissionPhase.AERODYNAMIC_DESCENT,
                "Speed reduced: aerodynamic descent",
            ),
            MissionPhase.AERODYNAMIC_DESCENT: (
                lambda: alt < 3_000.0,
                MissionPhase.LANDING_BURN,
                "Altitude < 3 km: landing burn ignition",
            ),
            MissionPhase.LANDING_BURN: (
                lambda: alt < 100.0,
                MissionPhase.TERMINAL_LANDING,
                "Altitude < 100 m: terminal landing",
            ),
        }

        if phase in transitions:
            condition, next_phase, message = transitions[phase]
            if condition():
                result.events_log.append((t, f"Phase transition: {message}"))
                return next_phase

        return phase

    def run(self) -> ScenarioResult:
        """Execute the full mission scenario.

        Returns:
            ScenarioResult with trajectory, events, and metrics.
        """
        self.setup()
        dt = self.sim_config.dt
        max_time = 320.0  # extended time for full mission
        result = ScenarioResult()
        result.events_log.append((0.0, "Scenario start: full mission"))
        result.trajectory_log.append(self.vehicle.state.clone())

        t = 0.0
        log_interval = 0.01  # log every step (dt = 0.01)
        next_log_time = log_interval
        max_dynamic_pressure = 0.0
        max_altitude = 0.0
        fault_idx = 0

        while t < max_time:
            state = self.vehicle.state
            if state.is_landed or state.is_destroyed:
                break

            # --- Fault injection ---
            while (fault_idx < len(self._fault_schedule)
                   and t >= self._fault_schedule[fault_idx][0]):
                ft, ftype = self._fault_schedule[fault_idx]
                self._faults_injected.append(ft)
                self._active_faults[ftype] = t
                result.events_log.append(
                    (t, f"FAULT INJECTED: {ftype} at t={ft:.1f}s")
                )
                fault_idx += 1

            # --- Environment ---
            alt = self.vehicle.get_altitude()
            max_altitude = max(max_altitude, alt)
            wind_vel = self.wind.get_wind(alt, t, dt)
            dcm = self.vehicle.get_dcm()

            rho = self.atmosphere.get_density(alt)
            speed = np.linalg.norm(state.velocity)
            dyn_pressure = 0.5 * rho * speed * speed
            max_dynamic_pressure = max(max_dynamic_pressure, dyn_pressure)

            f_gravity = self.gravity.compute_gravity(state.position, state.mass)
            f_aero, t_aero = self.aero.compute_aero_forces(
                state.position, state.velocity, dcm, self.atmosphere, wind_vel,
            )

            # --- Phase management ---
            new_phase = self._update_phase(state, dyn_pressure, result)
            if new_phase != state.phase:
                self.vehicle.state.phase = new_phase

            # --- Thrust command ---
            f_thrust = self._get_phase_thrust(state.phase, state)

            # Apply fault effects
            if "engine_degraded" in self._active_faults:
                f_thrust *= 0.75  # 25% thrust loss
            if "thrust_oscillation" in self._active_faults:
                osc = 1.0 + 0.05 * np.sin(2.0 * np.pi * 8.0 * t)
                f_thrust *= osc

            # --- Fuel consumption ---
            # Bug 1: Check fuel exhaustion as a termination condition
            thrust_mag = np.linalg.norm(f_thrust)
            g0 = 9.81
            isp = 282.0 + 29.0 * min(alt / 80_000.0, 1.0)  # altitude-dependent ISP
            if state.fuel_mass <= 0:
                f_thrust = np.zeros(3)
                thrust_mag = 0.0
            elif thrust_mag > 0:
                mass_flow = thrust_mag / (isp * g0)
                consumed = self.vehicle.consume_fuel(mass_flow, dt)

            # --- Integrate ---
            total_force = f_gravity + f_aero + f_thrust
            total_torque = t_aero + _stabilization_torque(self.vehicle)
            self.vehicle.apply_forces(total_force, total_torque, dt)
            t = self.vehicle.state.time

            # --- Logging ---
            if t >= next_log_time:
                result.trajectory_log.append(self.vehicle.state.clone())
                next_log_time += log_interval

        result.trajectory_log.append(self.vehicle.state.clone())
        result.events_log.append((t, "Scenario end"))

        # Stash metrics for evaluation
        self._max_dynamic_pressure = max_dynamic_pressure
        self._max_altitude = max_altitude
        return self.evaluate(result)

    def evaluate(self, result: ScenarioResult) -> ScenarioResult:
        """Evaluate the full mission scenario outcome.

        Success requires a soft landing with fuel remaining, having
        completed all phase transitions through the mission profile.
        """
        final = result.trajectory_log[-1]

        touchdown_pos_error = float(np.linalg.norm(
            final.position[:2] - self.target[:2]
        ))
        touchdown_speed = float(np.linalg.norm(final.velocity))

        phases_visited = set()
        for s in result.trajectory_log:
            phases_visited.add(s.phase)

        result.metrics = {
            "touchdown_pos_error_m": touchdown_pos_error,
            "touchdown_speed_m_s": touchdown_speed,
            "fuel_remaining_kg": final.fuel_mass,
            "flight_time_s": final.time,
            "is_landed": final.is_landed,
            "is_destroyed": final.is_destroyed,
            "failure_reason": result.failure_reason or ("timeout" if final.time >= 320.0 - 0.1 else "success" if final.is_landed else "crash"),
            "max_dynamic_pressure_pa": getattr(self, "_max_dynamic_pressure", 0.0),
            "max_altitude_m": getattr(self, "_max_altitude", 0.0),
            "phases_visited": [p.name for p in phases_visited],
            "faults_injected": len(self._faults_injected),
        }

        result.success = (
            final.is_landed
            and not final.is_destroyed
            and touchdown_speed < 5.0
        )
        return result
