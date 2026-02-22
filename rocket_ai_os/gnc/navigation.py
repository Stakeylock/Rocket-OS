"""Navigation subsystem for autonomous rocket flight.

Implements state estimation via an Extended Kalman Filter fusing
IMU (accelerometer + gyroscope) and GPS measurements. Sensor models
include configurable noise, bias drift, and GPS outage simulation.

State vector (16 elements):
    [pos_x, pos_y, pos_z,           # 0-2   inertial position (m)
     vel_x, vel_y, vel_z,           # 3-5   inertial velocity (m/s)
     q_w, q_x, q_y, q_z,           # 6-9   attitude quaternion (scalar-first)
     bias_ax, bias_ay, bias_az,     # 10-12 accelerometer bias (m/s^2)
     bias_gx, bias_gy, bias_gz]     # 13-15 gyroscope bias (rad/s)

References:
    - Titterton & Weston, "Strapdown Inertial Navigation Technology"
    - Crassidis & Junkins, "Optimal Estimation of Dynamic Systems"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from rocket_ai_os.config import VehicleConfig, SimConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NavigationState:
    """Estimated vehicle state produced by the navigation filter.

    All quantities are expressed in the launch-site inertial frame
    (East-North-Up).

    Attributes:
        position:       3D position [x, y, z] in metres.
        velocity:       3D velocity [vx, vy, vz] in m/s.
        attitude:       Unit quaternion [qw, qx, qy, qz] (scalar-first).
        angular_rates:  Body-frame angular rates [wx, wy, wz] in rad/s.
        mass:           Current vehicle mass in kg.
        timestamp:      Simulation time in seconds.
        covariance:     State covariance diagonal (16,) for health monitoring.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    angular_rates: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 0.0
    timestamp: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros(16))


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return unit quaternion (scalar-first)."""
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
    """Convert scalar-first quaternion to 3x3 Direction Cosine Matrix."""
    q = _quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ])


def _omega_matrix(w: np.ndarray) -> np.ndarray:
    """Build the 4x4 quaternion rate matrix from angular velocity vector.

    Used in the kinematic relation q_dot = 0.5 * Omega(w) * q.
    """
    wx, wy, wz = w
    return 0.5 * np.array([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0],
    ])


# ---------------------------------------------------------------------------
# Sensor models
# ---------------------------------------------------------------------------

class IMUSensor:
    """Strapdown IMU sensor model (accelerometer + gyroscope).

    Simulates measurement noise (white), slowly-drifting bias (random-walk),
    and quantisation effects for a tactical-grade MEMS IMU.

    Args:
        accel_noise_std: Accelerometer white noise sigma (m/s^2).
        gyro_noise_std:  Gyroscope white noise sigma (rad/s).
        accel_bias_std:  Accelerometer bias instability sigma (m/s^2).
        gyro_bias_std:   Gyroscope bias instability sigma (rad/s).
        dt:              Sample period in seconds.
        rng:             Numpy random generator for reproducibility.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.05,
        gyro_noise_std: float = 0.001,
        accel_bias_std: float = 0.005,
        gyro_bias_std: float = 0.0001,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_std = accel_bias_std
        self.gyro_bias_std = gyro_bias_std
        self.dt = dt
        self._rng = rng if rng is not None else np.random.default_rng()

        # Internal bias states (random walk)
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)

    def measure(
        self,
        true_accel_body: np.ndarray,
        true_omega_body: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate noisy IMU reading.

        Args:
            true_accel_body: True specific force in body frame (m/s^2).
            true_omega_body: True angular velocity in body frame (rad/s).

        Returns:
            Tuple of (measured_accel, measured_gyro) in body frame.
        """
        # Random-walk bias drift
        self._accel_bias += (
            self._rng.normal(0, self.accel_bias_std, 3) * np.sqrt(self.dt)
        )
        self._gyro_bias += (
            self._rng.normal(0, self.gyro_bias_std, 3) * np.sqrt(self.dt)
        )

        measured_accel = (
            true_accel_body
            + self._accel_bias
            + self._rng.normal(0, self.accel_noise_std, 3)
        )
        measured_gyro = (
            true_omega_body
            + self._gyro_bias
            + self._rng.normal(0, self.gyro_noise_std, 3)
        )
        return measured_accel, measured_gyro

    def reset(self) -> None:
        """Reset internal bias states to zero."""
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)


class GPSSensor:
    """GPS receiver model with noise and outage simulation.

    Provides position and velocity measurements in the inertial frame with
    configurable white noise.  Outages can be triggered programmatically to
    test filter coast performance.

    Args:
        pos_noise_std: Position noise sigma per axis (m).
        vel_noise_std: Velocity noise sigma per axis (m/s).
        update_rate_hz: Nominal GPS update rate.
        rng: Numpy random generator.
    """

    def __init__(
        self,
        pos_noise_std: float = 2.5,
        vel_noise_std: float = 0.1,
        update_rate_hz: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.update_period = 1.0 / update_rate_hz
        self._rng = rng if rng is not None else np.random.default_rng()
        self._last_update_time: float = -1.0

        # Outage control
        self._outage_active: bool = False

    # -- public API ----------------------------------------------------------

    def set_outage(self, active: bool) -> None:
        """Enable or disable a simulated GPS outage."""
        self._outage_active = active

    @property
    def is_available(self) -> bool:
        """True when GPS is providing valid fixes."""
        return not self._outage_active

    def measure(
        self,
        true_position: np.ndarray,
        true_velocity: np.ndarray,
        time: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return a GPS fix if available and the update interval has elapsed.

        Args:
            true_position: True inertial position (m).
            true_velocity: True inertial velocity (m/s).
            time:          Current simulation time (s).

        Returns:
            Tuple (measured_position, measured_velocity) or None.
        """
        if self._outage_active:
            return None

        if time - self._last_update_time < self.update_period - 1e-9:
            return None

        self._last_update_time = time
        meas_pos = true_position + self._rng.normal(0, self.pos_noise_std, 3)
        meas_vel = true_velocity + self._rng.normal(0, self.vel_noise_std, 3)
        return meas_pos, meas_vel


# ---------------------------------------------------------------------------
# Extended Kalman Filter
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """16-state Extended Kalman Filter for rocket navigation.

    State vector layout (16 elements):
        x[0:3]   - Inertial position (m)
        x[3:6]   - Inertial velocity (m/s)
        x[6:10]  - Attitude quaternion [w, x, y, z]
        x[10:13] - Accelerometer bias (m/s^2)
        x[13:16] - Gyroscope bias (rad/s)

    The filter propagates using body-frame IMU data and updates with
    GPS position/velocity fixes.

    Args:
        dt:       Propagation time step (s).
        gravity:  Gravity vector in inertial frame (m/s^2).
    """

    N_STATES: int = 16

    def __init__(
        self,
        dt: float = 0.01,
        gravity: Optional[np.ndarray] = None,
    ) -> None:
        self.dt = dt
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

        # State vector
        self.x = np.zeros(self.N_STATES)
        self.x[6] = 1.0  # identity quaternion w-component

        # Covariance
        self.P = np.eye(self.N_STATES)
        self.P[0:3, 0:3] *= 10.0       # position uncertainty
        self.P[3:6, 3:6] *= 5.0        # velocity uncertainty
        self.P[6:10, 6:10] *= 0.01     # quaternion uncertainty
        self.P[10:13, 10:13] *= 0.01   # accel bias
        self.P[13:16, 13:16] *= 0.001  # gyro bias

        # Process noise
        self.Q = np.eye(self.N_STATES) * 1e-4
        self.Q[0:3, 0:3] *= 0.01
        self.Q[3:6, 3:6] *= 0.1
        self.Q[6:10, 6:10] *= 0.001
        self.Q[10:13, 10:13] *= 0.0001
        self.Q[13:16, 13:16] *= 0.00001

        # GPS measurement noise (pos + vel = 6 measurements)
        self._R_gps = np.diag([
            2.5**2, 2.5**2, 2.5**2,   # position variance (m^2)
            0.1**2, 0.1**2, 0.1**2,   # velocity variance (m/s)^2
        ])

    # -- helpers -------------------------------------------------------------

    def _get_quat(self) -> np.ndarray:
        return self.x[6:10].copy()

    def _get_dcm(self) -> np.ndarray:
        return _quat_to_dcm(self._get_quat())

    # -- predict -------------------------------------------------------------

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
    ) -> None:
        """Propagate the state one time step using IMU measurements.

        Applies strap-down inertial navigation equations:
        - Remove estimated bias from IMU readings
        - Rotate body-frame acceleration to inertial frame
        - Integrate position and velocity with gravity
        - Propagate quaternion with angular velocity
        - Drive bias states as random walk (no deterministic model)

        Args:
            accel_body: Measured specific force in body frame (m/s^2).
            gyro_body:  Measured angular rate in body frame (rad/s).
        """
        dt = self.dt

        # Bias-corrected measurements
        accel_corrected = accel_body - self.x[10:13]
        gyro_corrected = gyro_body - self.x[13:16]

        # Current DCM (body -> inertial)
        C_bn = self._get_dcm()

        # Inertial acceleration = C_bn * specific_force + gravity
        accel_inertial = C_bn @ accel_corrected + self.gravity

        # --- State propagation ---
        # Position
        self.x[0:3] += self.x[3:6] * dt + 0.5 * accel_inertial * dt**2
        # Velocity
        self.x[3:6] += accel_inertial * dt

        # Quaternion propagation via matrix exponential (first-order)
        omega_mat = _omega_matrix(gyro_corrected)
        q_old = self._get_quat()
        q_new = q_old + omega_mat @ q_old * dt
        self.x[6:10] = _quat_normalize(q_new)

        # Biases: random walk -- no deterministic propagation
        # x[10:16] unchanged

        # --- Covariance propagation P = F P F^T + Q ---
        F = self._compute_jacobian(accel_corrected, gyro_corrected, C_bn)
        self.P = F @ self.P @ F.T + self.Q * dt

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def _compute_jacobian(
        self,
        accel_corrected: np.ndarray,
        gyro_corrected: np.ndarray,
        C_bn: np.ndarray,
    ) -> np.ndarray:
        """Compute the discrete-time state transition Jacobian F.

        This is the linearised system matrix used for covariance propagation.
        """
        dt = self.dt
        F = np.eye(self.N_STATES)

        # d(pos)/d(vel)
        F[0:3, 3:6] = np.eye(3) * dt

        # d(vel)/d(quat) -- skew-symmetric approximation
        a = C_bn @ accel_corrected
        skew_a = np.array([
            [0,    -a[2],  a[1]],
            [a[2],  0,    -a[0]],
            [-a[1], a[0],  0],
        ])
        # Map quaternion perturbation to velocity (simplified 3x4 -> use 3x3 slice)
        F[3:6, 6:9] = -skew_a * dt

        # d(vel)/d(accel_bias)
        F[3:6, 10:13] = -C_bn * dt

        # d(quat)/d(quat) -- from omega_matrix
        omega_mat = _omega_matrix(gyro_corrected)
        F[6:10, 6:10] = np.eye(4) + omega_mat * dt

        # d(quat)/d(gyro_bias) -- first-order sensitivity
        q = self._get_quat()
        # dq/d(gyro) = -0.5 * Xi(q) * dt  where Xi is the 4x3 matrix
        Xi = 0.5 * np.array([
            [-q[1], -q[2], -q[3]],
            [ q[0], -q[3],  q[2]],
            [ q[3],  q[0], -q[1]],
            [-q[2],  q[1],  q[0]],
        ])
        F[6:10, 13:16] = -Xi * dt

        return F

    # -- update --------------------------------------------------------------

    def update_gps(
        self,
        meas_pos: np.ndarray,
        meas_vel: np.ndarray,
    ) -> None:
        """Incorporate a GPS position + velocity fix via standard Kalman update.

        Args:
            meas_pos: Measured position in inertial frame (m).
            meas_vel: Measured velocity in inertial frame (m/s).
        """
        # Measurement vector z (6,)
        z = np.concatenate([meas_pos, meas_vel])

        # Predicted measurement h(x) -- direct observation of pos/vel states
        z_pred = np.concatenate([self.x[0:3], self.x[3:6]])

        # Innovation
        y = z - z_pred

        # Observation matrix H (6 x 16)
        H = np.zeros((6, self.N_STATES))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 3:6] = np.eye(3)  # velocity

        # Innovation covariance
        S = H @ self.P @ H.T + self._R_gps

        # Kalman gain (use solve for numerical stability)
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x += K @ y

        # Re-normalise quaternion
        self.x[6:10] = _quat_normalize(self.x[6:10])

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self._R_gps @ K.T

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update(
        self,
        imu_accel: np.ndarray,
        imu_gyro: np.ndarray,
        gps_measurement: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Full filter cycle: predict with IMU, optionally update with GPS.

        This is the primary entry point called by :class:`NavigationSystem`
        each time step.

        Args:
            imu_accel: Body-frame accelerometer reading (m/s^2).
            imu_gyro:  Body-frame gyroscope reading (rad/s).
            gps_measurement: Optional tuple (position, velocity) from GPS.
        """
        self.predict(imu_accel, imu_gyro)
        if gps_measurement is not None:
            self.update_gps(gps_measurement[0], gps_measurement[1])

    # -- output --------------------------------------------------------------

    def get_state(self, mass: float = 0.0, timestamp: float = 0.0) -> NavigationState:
        """Package current estimate as a NavigationState.

        Args:
            mass:      Current vehicle mass (passed through -- not estimated).
            timestamp: Current simulation time (s).

        Returns:
            NavigationState with all fields populated.
        """
        return NavigationState(
            position=self.x[0:3].copy(),
            velocity=self.x[3:6].copy(),
            attitude=self.x[6:10].copy(),
            angular_rates=np.zeros(3),  # not in state vector -- filled by NavSystem
            mass=mass,
            timestamp=timestamp,
            covariance=np.diag(self.P).copy(),
        )

    def set_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude: Optional[np.ndarray] = None,
    ) -> None:
        """Hard-set filter states (used for initialisation).

        Args:
            position: Initial position (m).
            velocity: Initial velocity (m/s).
            attitude: Initial quaternion [w, x, y, z].  Defaults to identity.
        """
        self.x[0:3] = position
        self.x[3:6] = velocity
        if attitude is not None:
            self.x[6:10] = _quat_normalize(attitude)


# ---------------------------------------------------------------------------
# Navigation System (orchestrator)
# ---------------------------------------------------------------------------

class NavigationSystem:
    """Top-level navigation system orchestrating sensors and the EKF.

    Maintains an IMU, GPS, and Extended Kalman Filter.  Each ``step()``
    call produces an updated :class:`NavigationState`.

    Args:
        vehicle_config: Vehicle parameters (used for mass tracking).
        sim_config:     Simulation parameters (dt, gravity).
        seed:           RNG seed for reproducible sensor noise.
    """

    def __init__(
        self,
        vehicle_config: Optional[VehicleConfig] = None,
        sim_config: Optional[SimConfig] = None,
        seed: int = 42,
    ) -> None:
        self._vehicle = vehicle_config if vehicle_config else VehicleConfig()
        self._sim = sim_config if sim_config else SimConfig()
        dt = self._sim.dt

        self._rng = np.random.default_rng(seed)

        self.imu = IMUSensor(
            accel_noise_std=0.05,
            gyro_noise_std=0.001,
            accel_bias_std=0.005,
            gyro_bias_std=0.0001,
            dt=dt,
            rng=self._rng,
        )
        self.gps = GPSSensor(
            pos_noise_std=2.5,
            vel_noise_std=0.1,
            update_rate_hz=10.0,
            rng=self._rng,
        )
        self.ekf = ExtendedKalmanFilter(
            dt=dt,
            gravity=self._sim.gravity.copy(),
        )

        self._current_mass: float = self._vehicle.total_mass
        self._time: float = 0.0
        self._last_gyro: np.ndarray = np.zeros(3)

    # -- public API ----------------------------------------------------------

    def initialise(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude: Optional[np.ndarray] = None,
        mass: Optional[float] = None,
    ) -> None:
        """Set initial conditions for the navigation filter.

        Args:
            position: Initial inertial position (m).
            velocity: Initial inertial velocity (m/s).
            attitude: Initial quaternion.  Defaults to identity.
            mass:     Initial vehicle mass.  Defaults to config total_mass.
        """
        self.ekf.set_state(position, velocity, attitude)
        if mass is not None:
            self._current_mass = mass
        self._time = 0.0
        self.imu.reset()

    def step(
        self,
        true_accel_body: np.ndarray,
        true_omega_body: np.ndarray,
        true_position: np.ndarray,
        true_velocity: np.ndarray,
        mass: float,
        time: float,
    ) -> NavigationState:
        """Execute one navigation cycle.

        1. Generate noisy IMU measurements from truth data.
        2. Attempt a GPS fix (may return None due to rate or outage).
        3. Run the EKF predict + update.
        4. Return the navigation solution.

        Args:
            true_accel_body: True specific force in body frame (m/s^2).
            true_omega_body: True angular rate in body frame (rad/s).
            true_position:   True inertial position (m).
            true_velocity:   True inertial velocity (m/s).
            mass:            Current vehicle mass (kg).
            time:            Current simulation time (s).

        Returns:
            NavigationState with latest estimates.
        """
        self._time = time
        self._current_mass = mass

        # 1. Sensor measurements
        imu_accel, imu_gyro = self.imu.measure(true_accel_body, true_omega_body)
        gps_fix = self.gps.measure(true_position, true_velocity, time)

        # 2. EKF cycle
        self.ekf.update(imu_accel, imu_gyro, gps_fix)

        # 3. Package state
        nav_state = self.ekf.get_state(mass=mass, timestamp=time)

        # Angular rates are taken from bias-corrected gyro (not in EKF state
        # vector directly, but readily available)
        self._last_gyro = imu_gyro - self.ekf.x[13:16]
        nav_state.angular_rates = self._last_gyro.copy()

        return nav_state

    def get_latest_state(self) -> NavigationState:
        """Return the most recent navigation solution without stepping."""
        nav = self.ekf.get_state(
            mass=self._current_mass, timestamp=self._time,
        )
        nav.angular_rates = self._last_gyro.copy()
        return nav
