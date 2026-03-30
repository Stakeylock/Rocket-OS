"""Flight control subsystem for autonomous rocket landing.

Implements a layered control architecture:

1. **PIDController** -- Classical 3-axis PID with anti-windup, used as
   the deterministic baseline for attitude and rate control.

2. **RLAdaptiveController** -- A lightweight neural-network policy that
   mimics a PPO-trained agent. Runs inference using pure NumPy and outputs
   additive control corrections. Supports domain randomisation of mass and
   thrust parameters.

3. **SimplexControlSwitch** -- Simplex-architecture safety monitor that
   wraps the RL controller with the PID baseline. If the RL output violates
   safety constraints the switch reverts to the PID controller transparently.

4. **FlightController** -- Top-level controller that blends PID and RL
   outputs to produce gimbal angle commands and throttle adjustments via
   quaternion-error attitude control, thrust-vector control, and rate
   damping.

All math uses NumPy. No external ML frameworks are required.

References:
    - Sha et al., "Using Simplicity to Control Complexity", IEEE Software 2001.
    - Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from dataclasses import dataclass, field
from typing import Optional, Tuple

from rocket_ai_os.config import VehicleConfig, GuidanceConfig, SimConfig
from rocket_ai_os.gnc.navigation import NavigationState, _quat_multiply, _quat_normalize


# ---------------------------------------------------------------------------
# Quaternion helpers local to control
# ---------------------------------------------------------------------------

def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternion), scalar-first."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_error(q_current: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
    """Compute the rotation error quaternion: q_err = q_desired^* * q_current.

    The vector part of q_err is proportional to the small-angle attitude
    error and is suitable as input to a PID controller.

    Returns:
        Error quaternion (scalar-first).  The scalar part is forced positive
        to avoid the double-cover ambiguity.
    """
    q_err = _quat_multiply(_quat_conjugate(q_desired), q_current)
    if q_err[0] < 0:
        q_err = -q_err
    return _quat_normalize(q_err)


# ---------------------------------------------------------------------------
# Discrete LQR Controller
# ---------------------------------------------------------------------------   

class DiscreteLQR:
    """Discrete-time Linear Quadratic Regulator for optimal baseline tracking. 
    
    Replaces rudimentary PID logic to allow mathematically optimal trajectory
    tracking over the baseline dynamics.
    
    Args:
        A: Linearized state transition matrix (n, n)
        B: Linearized control input matrix (n, m)
        Q: State cost matrix (n, n)
        R: Control cost matrix (m, m)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        
        # solve discrete algebraic Riccati equation
        self.P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        # Compute optimal gain matrix K
        # K = (R + B^T P B)^-1 (B^T P A)
        bpb = self.B.T @ self.P @ self.B
        bpa = self.B.T @ self.P @ self.A
        self.K = np.linalg.inv(self.R + bpb) @ bpa

    def compute(self, x_err: np.ndarray) -> np.ndarray:
        """Compute LQR control given the state error (x_desired - x_current).
        Since we want u to drive x towards the origin of the error space, 
        u = K * x_err.
        """
        # Original LQR formulation solves u = -Kx for moving a state to zero.
        # So we pass in error = state_desired - state_current 
        # meaning state_err needs to be driven to zero by applying K * err
        return self.K @ x_err

# ---------------------------------------------------------------------------   
# ---------------------------------------------------------------------------

class PIDController:
    """3-axis PID controller with integrator anti-windup.

    Each axis is independently controlled.  The integrator is clamped to
    ``[-windup_limit, +windup_limit]`` to prevent saturation-induced
    divergence.

    Args:
        kp: Proportional gains (3,).
        ki: Integral gains (3,).
        kd: Derivative gains (3,).
        windup_limit: Maximum absolute integrator value per axis.
        output_limit: Maximum absolute output per axis.
        dt: Control time step (s).
    """

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        windup_limit: float = 50.0,
        output_limit: float = 100.0,
        dt: float = 0.01,
    ) -> None:
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.windup_limit = windup_limit
        self.output_limit = output_limit
        self.dt = dt

        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._initialized = False

    def reset(self) -> None:
        """Zero the integrator and derivative memory."""
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._initialized = False

    def compute(self, error: np.ndarray) -> np.ndarray:
        """Compute PID output for a 3-axis error signal.

        Args:
            error: Error vector (3,).  Positive error produces positive output.

        Returns:
            Control output (3,), clamped to ``[-output_limit, +output_limit]``.
        """
        # Integral with anti-windup (clamping)
        self._integral += error * self.dt
        self._integral = np.clip(
            self._integral, -self.windup_limit, self.windup_limit,
        )

        # Derivative (backward difference; skip first step)
        if self._initialized:
            derivative = (error - self._prev_error) / self.dt
        else:
            derivative = np.zeros(3)
            self._initialized = True

        self._prev_error = error.copy()

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(output, -self.output_limit, self.output_limit)


# ---------------------------------------------------------------------------
# RL Adaptive Controller (PPO-style policy, NumPy inference)
# ---------------------------------------------------------------------------

class RLAdaptiveController:
    """Lightweight neural-network controller simulating a PPO-trained policy.

    Architecture:
        - Input:  13-dimensional observation
                  [quat_err(3), angular_rate(3), pos_err(3), vel_err(3), throttle(1)]
        - Hidden: Two fully-connected layers (64 neurons each, tanh activation)
        - Output: 4-dimensional action [torque_correction(3), throttle_correction(1)]

    Weights are initialised with Xavier/Glorot uniform scaling.  For
    simulation purposes this produces *plausible* adaptive corrections;
    in a real system the weights would be loaded from a trained checkpoint.

    Domain randomisation is supported by injecting mass and thrust scale
    factors into the observation vector (appended).

    Args:
        hidden_size: Neurons per hidden layer.
        seed:        RNG seed for weight initialisation.
    """

    OBS_DIM: int = 13
    ACT_DIM: int = 4

    def __init__(
        self,
        hidden_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self._rng = np.random.default_rng(seed)

        # Domain randomisation parameters (nominal = 1.0)
        self._mass_scale: float = 1.0
        self._thrust_scale: float = 1.0

        # Xavier-initialised weights
        self._W1 = self._xavier((self.OBS_DIM + 2, hidden_size))  # +2 for domain rand
        self._b1 = np.zeros(hidden_size)
        self._W2 = self._xavier((hidden_size, hidden_size))
        self._b2 = np.zeros(hidden_size)
        self._W3 = self._xavier((hidden_size, self.ACT_DIM))
        self._b3 = np.zeros(self.ACT_DIM)

        # Output scaling (keep corrections small relative to PID)
        self._output_scale = np.array([5.0, 5.0, 5.0, 0.1])

    def _xavier(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Xavier/Glorot uniform initialisation."""
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self._rng.uniform(-limit, limit, shape)

    def set_domain_params(
        self, mass_scale: float = 1.0, thrust_scale: float = 1.0,
    ) -> None:
        """Set domain-randomisation scale factors.

        Args:
            mass_scale:   mass / nominal_mass  (>1 = heavier).
            thrust_scale: thrust / nominal_thrust (>1 = stronger).
        """
        self._mass_scale = mass_scale
        self._thrust_scale = thrust_scale

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run a forward pass through the policy network.

        Args:
            observation: 13-element observation vector.

        Returns:
            4-element action: [torque_x, torque_y, torque_z, throttle_delta].
        """
        # Append domain randomisation features
        obs = np.concatenate([
            observation,
            np.array([self._mass_scale, self._thrust_scale]),
        ])

        # Layer 1
        h1 = np.tanh(obs @ self._W1 + self._b1)
        # Layer 2
        h2 = np.tanh(h1 @ self._W2 + self._b2)
        # Output layer (no activation -- linear, then scale)
        raw = h2 @ self._W3 + self._b3
        action = np.tanh(raw) * self._output_scale
        return action

    def compute(
        self,
        quat_error_vec: np.ndarray,
        angular_rate: np.ndarray,
        position_error: np.ndarray,
        velocity_error: np.ndarray,
        throttle: float,
    ) -> Tuple[np.ndarray, float]:
        """Build observation, run inference, return corrections.

        Args:
            quat_error_vec: Attitude error vector part (3,).
            angular_rate:   Body-frame angular rate (3,) rad/s.
            position_error: Inertial position error (3,) m.
            velocity_error: Inertial velocity error (3,) m/s.
            throttle:       Current throttle [0, 1].

        Returns:
            Tuple of (torque_correction (3,), throttle_correction (float)).
        """
        obs = np.concatenate([
            quat_error_vec,
            angular_rate,
            position_error,
            velocity_error,
            np.array([throttle]),
        ])
        action = self.infer(obs)
        return action[0:3], float(action[3])


# ---------------------------------------------------------------------------
# Simplex Architecture Switch
# ---------------------------------------------------------------------------

class SimplexControlSwitch:
    """Simplex-architecture safety wrapper around an AI controller.

    The Simplex approach runs two controllers in parallel:
        - **Advanced controller** (RL-based) -- primary, performance-oriented.
        - **Baseline controller** (PID) -- proven safe, conservative.

    A decision module monitors the advanced controller's output and switches
    to the baseline if any safety constraint is violated.

    Safety constraints checked:
        1. Torque magnitude within limits.
        2. Throttle correction within bounds.
        3. Combined output keeps vehicle within a safe envelope (attitude
           error bound, rate bound).

    Args:
        max_torque:     Maximum allowable torque magnitude (N-m).
        max_throttle_delta: Maximum throttle correction magnitude.
        max_rate:       Maximum allowable angular rate magnitude (rad/s).
        max_attitude_err: Maximum attitude error vector norm (rad).
    """

    def __init__(
        self,
        max_torque: float = 50.0,
        max_throttle_delta: float = 0.2,
        max_rate: float = 1.0,
        max_attitude_err: float = 0.5,
    ) -> None:
        self.max_torque = max_torque
        self.max_throttle_delta = max_throttle_delta
        self.max_rate = max_rate
        self.max_attitude_err = max_attitude_err

        self._using_baseline: bool = False
        self._switch_count: int = 0

    @property
    def is_using_baseline(self) -> bool:
        """True if the baseline (PID) controller is currently active."""
        return self._using_baseline

    @property
    def switch_count(self) -> int:
        """Number of times the switch has reverted to baseline."""
        return self._switch_count

    def evaluate(
        self,
        rl_torque: np.ndarray,
        rl_throttle_delta: float,
        pid_torque: np.ndarray,
        pid_throttle_delta: float,
        angular_rate: np.ndarray,
        attitude_error_norm: float,
    ) -> Tuple[np.ndarray, float, bool]:
        """Decide which controller output to use.

        Args:
            rl_torque:            RL controller torque command (3,) N-m.
            rl_throttle_delta:    RL throttle correction.
            pid_torque:           PID controller torque command (3,) N-m.
            pid_throttle_delta:   PID throttle correction.
            angular_rate:         Current angular rates (3,) rad/s.
            attitude_error_norm:  Norm of attitude error vector (rad).

        Returns:
            Tuple of (torque, throttle_delta, used_baseline).
        """
        safe = True

        # Check RL output magnitude
        if np.linalg.norm(rl_torque) > self.max_torque:
            safe = False
        if abs(rl_throttle_delta) > self.max_throttle_delta:
            safe = False

        # Check vehicle state safety
        if np.linalg.norm(angular_rate) > self.max_rate:
            safe = False
        if attitude_error_norm > self.max_attitude_err:
            safe = False

        if safe:
            self._using_baseline = False
            return rl_torque, rl_throttle_delta, False
        else:
            if not self._using_baseline:
                self._switch_count += 1
            self._using_baseline = True
            return pid_torque, pid_throttle_delta, True

    def reset(self) -> None:
        """Reset switch state (e.g. on phase change)."""
        self._using_baseline = False


# ---------------------------------------------------------------------------
# Flight Controller (top-level)
# ---------------------------------------------------------------------------

@dataclass
class ControlCommand:
    """Output of the flight controller.

    Attributes:
        gimbal_angles:    [pitch_gimbal, yaw_gimbal] in radians.
        throttle:         Throttle command in [0, 1].
        torque_command:   Body-frame torque command (3,) N-m.
        using_baseline:   True if Simplex reverted to PID.
    """
    gimbal_angles: np.ndarray = field(default_factory=lambda: np.zeros(2))
    throttle: float = 0.0
    torque_command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    using_baseline: bool = False


class FlightController:
    """Top-level flight controller blending PID and RL-adaptive outputs.

    Processing pipeline (each ``step``):

    1. Compute quaternion attitude error from guidance commands.
    2. Run 3-axis PID on attitude error for baseline torque.
    3. Run PID on position/velocity error for baseline throttle correction.
    4. Run RL adaptive controller for corrections.
    5. Pass both through Simplex switch.
    6. Convert torque command to gimbal angles.
    7. Apply rate-damping augmentation.

    Args:
        vehicle_config: Vehicle parameters (inertia, gimbal limits, etc.).
        sim_config:     Simulation configuration.
        rl_seed:        Seed for RL policy weight initialisation.
    """

    def __init__(
        self,
        vehicle_config: Optional[VehicleConfig] = None,
        sim_config: Optional[SimConfig] = None,
        rl_seed: int = 42,
    ) -> None:
        self._vc = vehicle_config if vehicle_config else VehicleConfig()
        self._sc = sim_config if sim_config else SimConfig()
        dt = self._sc.dt

        # --- Attitude LQR Baseline (Replaces PID) ---
        # A simple double-integrator plant for attitude dynamics
        # state: [theta, theta_dot], control: [torque]
        # Very simplified decoupled baseline model for LQR derivation
        dt_lqr = dt
        A_1d = np.array([[1.0, dt_lqr], [0.0, 1.0]])
        # I ~ mass * 0.5 roughly, assume 1.0 for normalized design
        B_1d = np.array([[0.5 * dt_lqr**2], [dt_lqr]]) 
        Q_1d = np.diag([800.0, 50.0])
        R_1d = np.array([[1.0]])
        
        self.attitude_lqr = DiscreteLQR(A_1d, B_1d, Q_1d, R_1d)

        # Build 3-axis decoupled LQR (we can compute per-axis or apply the 1D LQR sequentially)
        # We will apply the 1D LQR to each axis (Roll, Pitch, Yaw)

        # --- Position / velocity LQR (replaces throttle PID) ---
        A_pos = np.array([[1.0, dt_lqr], [0.0, 1.0]])
        B_pos = np.array([[0.5 * dt_lqr**2], [dt_lqr]])
        Q_pos = np.diag([2.0, 4.0])
        R_pos = np.array([[10.0]])
        self.pos_lqr = DiscreteLQR(A_pos, B_pos, Q_pos, R_pos)

        # --- RL adaptive controller ---
        self.rl_controller = RLAdaptiveController(
            hidden_size=64,
            seed=rl_seed,
        )

        # --- Simplex switch ---
        self.simplex = SimplexControlSwitch(
            max_torque=80.0,
            max_throttle_delta=0.2,
            max_rate=1.5,
            max_attitude_err=0.5,
        )

        # --- Blending weight for RL corrections [0=PID only, 1=full RL] ---
        self._rl_blend: float = 0.3

        # --- Desired state (set by guidance) ---
        self._desired_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self._desired_position = np.zeros(3)
        self._desired_velocity = np.zeros(3)
        self._desired_throttle: float = 0.0

    # -- configuration -------------------------------------------------------

    def set_desired_state(
        self,
        attitude: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        throttle: Optional[float] = None,
    ) -> None:
        """Update the desired state from guidance commands.

        Args:
            attitude: Desired quaternion [w, x, y, z].
            position: Desired inertial position (m).
            velocity: Desired inertial velocity (m/s).
            throttle: Desired throttle [0, 1].
        """
        if attitude is not None:
            self._desired_attitude = _quat_normalize(attitude)
        if position is not None:
            self._desired_position = position.copy()
        if velocity is not None:
            self._desired_velocity = velocity.copy()
        if throttle is not None:
            self._desired_throttle = throttle

    def set_rl_blend(self, blend: float) -> None:
        """Set the blending factor for RL corrections.

        Args:
            blend: Float in [0, 1].  0 = PID only, 1 = full RL correction.
        """
        self._rl_blend = np.clip(blend, 0.0, 1.0)

    def set_domain_randomisation(
        self, mass_scale: float = 1.0, thrust_scale: float = 1.0,
    ) -> None:
        """Forward domain randomisation parameters to the RL controller."""
        self.rl_controller.set_domain_params(mass_scale, thrust_scale)

    # -- main control step ---------------------------------------------------

    def step(self, nav_state: NavigationState) -> ControlCommand:
        """Execute one control cycle.

        Args:
            nav_state: Current navigation state estimate.

        Returns:
            ControlCommand with gimbal angles, throttle, and torque.
        """
        # 1. Attitude error (quaternion error -> vector part is ~half-angle)
        q_err = _quat_error(nav_state.attitude, self._desired_attitude)
        att_err_vec = q_err[1:4]  # vector part ~ rotation error (small angle)
        
        # 2. LQR attitude torque + rate damping combined
        pid_torque = np.zeros(3)
        rate_err = nav_state.angular_rates
        for i in range(3):
            # State vector driving to 0
            state_err = np.array([att_err_vec[i], rate_err[i]])
            pid_torque[i] = -self.attitude_lqr.compute(state_err)[0]
            
        # 4. Position / velocity LQR -> throttle correction
        pos_err = self._desired_position - nav_state.position
        vel_err = self._desired_velocity - nav_state.velocity
        
        state_err_pos_z = np.array([pos_err[2], vel_err[2]])
        # Negative because we computed pos_err = desired - current, 
        # so LQR(x) expects current state relative to target = current - desired = -err
        pid_throttle_correction_z = self.pos_lqr.compute(-state_err_pos_z)[0]
        
        # Map correction to scalar throttle delta 
        pid_throttle_delta = float(np.clip(
            pid_throttle_correction_z * 0.01, -0.2, 0.2,
        ))

        # 5. RL adaptive corrections
        rl_torque_corr, rl_throttle_corr = self.rl_controller.compute(
            quat_error_vec=att_err_vec,
            angular_rate=nav_state.angular_rates,
            position_error=pos_err,
            velocity_error=vel_err,
            throttle=self._desired_throttle,
        )

        # 6. Simplex safety switch
        att_err_norm = np.linalg.norm(att_err_vec)
        final_torque, final_throttle_delta, used_baseline = self.simplex.evaluate(
            rl_torque=pid_torque + self._rl_blend * rl_torque_corr,
            rl_throttle_delta=pid_throttle_delta + self._rl_blend * rl_throttle_corr,
            pid_torque=pid_torque,
            pid_throttle_delta=pid_throttle_delta,
            angular_rate=nav_state.angular_rates,
            attitude_error_norm=att_err_norm,
        )

        # 7. Throttle command
        throttle_cmd = np.clip(
            self._desired_throttle + final_throttle_delta, 0.0, 1.0,
        )

        # 8. Convert torque to gimbal angles (simplified TVC model)
        gimbal_angles = self._torque_to_gimbal(
            final_torque, throttle_cmd, nav_state.mass,
        )

        return ControlCommand(
            gimbal_angles=gimbal_angles,
            throttle=throttle_cmd,
            torque_command=final_torque,
            using_baseline=used_baseline,
        )

    # -- thrust vector control -----------------------------------------------

    def _torque_to_gimbal(
        self,
        torque: np.ndarray,
        throttle: float,
        mass: float,
    ) -> np.ndarray:
        """Convert a body-frame torque command into gimbal deflection angles.

        The gimbal produces torque by deflecting the thrust vector.  For a
        single-nozzle simplified model:

            torque_x ~ F * L * sin(gimbal_pitch)
            torque_y ~ F * L * sin(gimbal_yaw)

        where F is total thrust and L is the moment arm (distance from CG to
        engine gimbal point, approximately vehicle_length / 2).

        Args:
            torque:   Desired body-frame torque [Tx, Ty, Tz] (N-m).
            throttle: Current throttle setting [0, 1].
            mass:     Current vehicle mass (kg).

        Returns:
            Gimbal angles [pitch, yaw] in radians, clamped to max_gimbal_angle.
        """
        max_gimbal = self._vc.max_gimbal_angle
        thrust_force = throttle * self._vc.max_total_thrust
        moment_arm = self._vc.vehicle_length * 0.5

        effective_force = thrust_force * moment_arm
        if effective_force < 1.0:
            return np.zeros(2)

        # Desired gimbal deflection (small-angle: sin(theta) ~ theta)
        gimbal_pitch = torque[0] / effective_force
        gimbal_yaw = torque[1] / effective_force

        gimbal_pitch = np.clip(gimbal_pitch, -max_gimbal, max_gimbal)
        gimbal_yaw = np.clip(gimbal_yaw, -max_gimbal, max_gimbal)

        return np.array([gimbal_pitch, gimbal_yaw])

    # -- reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal controller states."""
        self.simplex.reset()
        self._desired_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self._desired_position = np.zeros(3)
        self._desired_velocity = np.zeros(3)
        self._desired_throttle = 0.0
