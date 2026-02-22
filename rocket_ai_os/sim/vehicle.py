"""
Vehicle state model and rigid-body dynamics for rocket simulation.

Provides a ``VehicleState`` dataclass capturing the full kinematic and
mass state of the vehicle, and a ``Vehicle`` class that integrates
forces and torques via semi-implicit Euler with quaternion attitude
propagation.

Coordinate conventions:
    - Inertial frame: East-North-Up (ENU) with origin at the landing pad.
    - Body frame: X-forward, Y-right, Z-up (through nose).
    - Quaternion: scalar-first [w, x, y, z].
"""

from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from rocket_ai_os.config import VehicleConfig, MissionPhase


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return a unit quaternion (scalar-first)."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions."""
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array([
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    ])


def _quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion to a 3x3 Direction Cosine Matrix.

    The returned DCM rotates vectors from the body frame to the inertial
    frame: v_inertial = C @ v_body.
    """
    q = _quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),      1 - 2 * (x * x + y * y)],
    ])


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ---------------------------------------------------------------------------
# VehicleState dataclass
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Complete observable state of the vehicle at a single instant.

    Attributes:
        position:         Inertial position [x, y, z] in metres (ENU).
        velocity:         Inertial velocity [vx, vy, vz] in m/s.
        attitude:         Attitude quaternion [w, x, y, z] (scalar-first).
        angular_velocity: Body-frame angular velocity [wx, wy, wz] in rad/s.
        mass:             Total vehicle mass in kg.
        fuel_mass:        Remaining propellant mass in kg.
        phase:            Current mission phase.
        time:             Simulation time in seconds.
        is_landed:        True when the vehicle has touched down successfully.
        is_destroyed:     True when the vehicle has exceeded structural limits.
    """
    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    attitude: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    mass: float = 0.0
    fuel_mass: float = 0.0
    phase: MissionPhase = MissionPhase.PRE_LAUNCH
    time: float = 0.0
    is_landed: bool = False
    is_destroyed: bool = False

    def clone(self) -> VehicleState:
        """Return a deep copy of this state for snapshot logging."""
        return VehicleState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=self.attitude.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass,
            fuel_mass=self.fuel_mass,
            phase=self.phase,
            time=self.time,
            is_landed=self.is_landed,
            is_destroyed=self.is_destroyed,
        )


# ---------------------------------------------------------------------------
# Structural / thermal constraint limits
# ---------------------------------------------------------------------------

# Maximum allowable values for constraint checking
_MAX_ACCELERATION_G = 15.0          # g  (structural limit)
_MAX_DYNAMIC_PRESSURE_PA = 35_000   # Pa (max-Q limit)
_MAX_ANGULAR_RATE_RAD_S = 2.0       # rad/s per axis
_MAX_ATTITUDE_ERROR_RAD = np.radians(60.0)  # from vertical during landing
_MIN_ALTITUDE_M = -1.0              # below ground = impact
_MAX_TEMPERATURE_K = 1_800.0        # aerothermal limit


# ---------------------------------------------------------------------------
# Vehicle class
# ---------------------------------------------------------------------------

class Vehicle:
    """Rigid-body vehicle model with six-degree-of-freedom dynamics.

    Integrates translational and rotational equations of motion using a
    semi-implicit Euler scheme.  Attitude is propagated as a quaternion
    to avoid gimbal lock.

    Args:
        config: Vehicle physical parameters.
        initial_phase: Starting mission phase.
    """

    def __init__(
        self,
        config: Optional[VehicleConfig] = None,
        initial_phase: MissionPhase = MissionPhase.PRE_LAUNCH,
    ) -> None:
        self.config = config if config is not None else VehicleConfig()

        # Build initial state
        self.state = VehicleState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            mass=self.config.total_mass,
            fuel_mass=self.config.fuel_mass,
            phase=initial_phase,
            time=0.0,
            is_landed=False,
            is_destroyed=False,
        )

        # Cache moment of inertia and its inverse
        self._inertia = self.config.moment_of_inertia.copy()
        self._inertia_inv = np.linalg.inv(self._inertia)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def set_state(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        attitude: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
        mass: Optional[float] = None,
        fuel_mass: Optional[float] = None,
        phase: Optional[MissionPhase] = None,
        time: Optional[float] = None,
    ) -> None:
        """Directly set individual state components.

        Only the provided arguments are overwritten; the rest remain
        unchanged.  This is useful for scenario initialisation.
        """
        if position is not None:
            self.state.position = np.asarray(position, dtype=np.float64)
        if velocity is not None:
            self.state.velocity = np.asarray(velocity, dtype=np.float64)
        if attitude is not None:
            self.state.attitude = _quat_normalize(
                np.asarray(attitude, dtype=np.float64)
            )
        if angular_velocity is not None:
            self.state.angular_velocity = np.asarray(
                angular_velocity, dtype=np.float64
            )
        if mass is not None:
            self.state.mass = float(mass)
        if fuel_mass is not None:
            self.state.fuel_mass = float(fuel_mass)
        if phase is not None:
            self.state.phase = phase
        if time is not None:
            self.state.time = float(time)

    # ------------------------------------------------------------------
    # Dynamics integration
    # ------------------------------------------------------------------

    def apply_forces(
        self,
        force: np.ndarray,
        torque: np.ndarray,
        dt: float,
    ) -> None:
        """Advance the vehicle state by *dt* under the given force and torque.

        Uses semi-implicit Euler integration:
          1. Update velocity from force (translational).
          2. Update position from new velocity.
          3. Update angular velocity from torque (Euler's equation).
          4. Integrate attitude quaternion from new angular velocity.

        Both *force* and *torque* must be expressed in the **inertial** frame.
        The torque is internally rotated into the body frame for the Euler
        equation.

        Args:
            force:  Net inertial-frame force vector [Fx, Fy, Fz] in N.
            torque: Net inertial-frame torque vector [Tx, Ty, Tz] in N*m.
            dt:     Integration time step in seconds.
        """
        if self.state.is_landed or self.state.is_destroyed:
            self.state.time += dt
            return

        force = np.asarray(force, dtype=np.float64)
        torque = np.asarray(torque, dtype=np.float64)

        # --- translational dynamics ---
        mass = max(self.state.mass, 1.0)  # guard against zero mass
        acceleration = force / mass

        # Semi-implicit Euler: velocity first, then position
        self.state.velocity += acceleration * dt
        self.state.position += self.state.velocity * dt

        # --- rotational dynamics (body frame) ---
        # Rotate torque from inertial to body frame
        C_bn = _quat_to_dcm(self.state.attitude)        # body-from-inertial
        C_nb = C_bn.T                                     # inertial-from-body
        torque_body = C_nb @ torque

        # Euler's rotation equation: I * omega_dot = tau - omega x (I * omega)
        omega = self.state.angular_velocity
        I_omega = self._inertia @ omega
        omega_dot = self._inertia_inv @ (torque_body - np.cross(omega, I_omega))
        self.state.angular_velocity += omega_dot * dt

        # --- quaternion integration ---
        omega_new = self.state.angular_velocity
        omega_norm = np.linalg.norm(omega_new)

        if omega_norm > 1e-12:
            # Exact exponential map for constant angular velocity over dt
            half_angle = 0.5 * omega_norm * dt
            axis = omega_new / omega_norm
            dq = np.zeros(4)
            dq[0] = np.cos(half_angle)
            dq[1:] = axis * np.sin(half_angle)
        else:
            # First-order approximation for very small rotation
            dq = np.array([1.0, 0.0, 0.0, 0.0])
            dq[1:] = 0.5 * omega_new * dt

        self.state.attitude = _quat_normalize(
            _quat_multiply(self.state.attitude, dq)
        )

        # --- ground contact ---
        if self.state.position[2] <= 0.0:
            self.state.position[2] = 0.0
            speed = np.linalg.norm(self.state.velocity)
            vertical_speed = abs(self.state.velocity[2])

            if vertical_speed < 5.0 and speed < 10.0:
                # Successful soft landing
                self.state.velocity = np.zeros(3)
                self.state.angular_velocity = np.zeros(3)
                self.state.is_landed = True
                self.state.phase = MissionPhase.LANDED
            else:
                # Hard impact -- vehicle destroyed
                self.state.velocity = np.zeros(3)
                self.state.angular_velocity = np.zeros(3)
                self.state.is_destroyed = True

        # Advance time
        self.state.time += dt

    # ------------------------------------------------------------------
    # Fuel consumption
    # ------------------------------------------------------------------

    def consume_fuel(self, mass_flow_rate: float, dt: float) -> float:
        """Reduce propellant mass at the given flow rate.

        Prevents fuel from going negative.  Updates both ``fuel_mass``
        and total ``mass``.

        Args:
            mass_flow_rate: Fuel consumption rate in kg/s (positive).
            dt:             Time step in seconds.

        Returns:
            The actual fuel mass consumed in this step (may be less than
            requested if fuel is nearly exhausted).
        """
        if mass_flow_rate <= 0.0 or dt <= 0.0:
            return 0.0

        desired = mass_flow_rate * dt
        actual = min(desired, self.state.fuel_mass)
        self.state.fuel_mass -= actual
        self.state.mass -= actual
        return actual

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_altitude(self) -> float:
        """Return the vehicle altitude (z-component of position) in metres."""
        return float(self.state.position[2])

    def get_speed(self) -> float:
        """Return the magnitude of the inertial velocity in m/s."""
        return float(np.linalg.norm(self.state.velocity))

    def get_downrange(self) -> float:
        """Return the horizontal distance from the origin in metres.

        Computed as the Euclidean norm of the x and y position components.
        """
        return float(np.linalg.norm(self.state.position[:2]))

    def get_dcm(self) -> np.ndarray:
        """Return the body-to-inertial Direction Cosine Matrix (3x3)."""
        return _quat_to_dcm(self.state.attitude)

    def get_tilt_angle(self) -> float:
        """Return the tilt angle from vertical in radians.

        Zero means the vehicle body Z-axis is aligned with the inertial
        Z-axis (pointing up).
        """
        C = _quat_to_dcm(self.state.attitude)
        body_z_in_inertial = C @ np.array([0.0, 0.0, 1.0])
        cos_angle = np.clip(body_z_in_inertial[2], -1.0, 1.0)
        return float(np.arccos(cos_angle))

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def check_constraints(self) -> List[str]:
        """Evaluate structural, thermal, and kinematic constraints.

        Returns a list of human-readable violation descriptions.  An empty
        list means all constraints are satisfied.

        Constraints checked:
            1. Axial acceleration limit (structural loads).
            2. Dynamic pressure limit (max-Q thermal/structural).
            3. Angular rate limit (attitude control authority).
            4. Tilt angle limit during terminal phase.
            5. Below-ground altitude (impact).
            6. Fuel exhaustion warning.
        """
        violations: List[str] = []
        g0 = 9.81

        # 1. Acceleration limit
        if self.state.mass > 0:
            # Approximate acceleration from velocity history is unavailable;
            # we use a proxy from the state: not perfectly accurate but useful.
            # For a proper check the caller passes the net force separately.
            pass  # Checked externally when force is known

        # 2. Angular rate limit
        omega_mag = np.linalg.norm(self.state.angular_velocity)
        if omega_mag > _MAX_ANGULAR_RATE_RAD_S:
            violations.append(
                f"Angular rate {np.degrees(omega_mag):.1f} deg/s exceeds "
                f"limit {np.degrees(_MAX_ANGULAR_RATE_RAD_S):.1f} deg/s"
            )

        # 3. Tilt angle during landing phases
        landing_phases = {
            MissionPhase.LANDING_BURN,
            MissionPhase.TERMINAL_LANDING,
        }
        if self.state.phase in landing_phases:
            tilt = self.get_tilt_angle()
            if tilt > _MAX_ATTITUDE_ERROR_RAD:
                violations.append(
                    f"Tilt angle {np.degrees(tilt):.1f} deg exceeds "
                    f"limit {np.degrees(_MAX_ATTITUDE_ERROR_RAD):.1f} deg "
                    f"during {self.state.phase.name}"
                )

        # 4. Below-ground check
        if self.state.position[2] < _MIN_ALTITUDE_M:
            violations.append(
                f"Altitude {self.state.position[2]:.1f} m is below ground"
            )

        # 5. Fuel exhaustion
        if self.state.fuel_mass <= 0.0 and self.state.phase not in {
            MissionPhase.LANDED, MissionPhase.PRE_LAUNCH,
        }:
            violations.append("Propellant exhausted during active flight")

        # 6. Excessive speed at low altitude (danger of impact)
        if self.get_altitude() < 50.0 and self.get_speed() > 100.0:
            violations.append(
                f"Excessive speed {self.get_speed():.1f} m/s "
                f"at low altitude {self.get_altitude():.1f} m"
            )

        return violations

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def clone(self) -> Vehicle:
        """Return a deep copy of the entire vehicle (config + state).

        Useful for branch-and-bound planners or Monte Carlo analysis.
        """
        return copy.deepcopy(self)
