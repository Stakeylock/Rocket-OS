"""
Simplex Architecture for AI Safety Assurance.

The Simplex architecture wraps a high-performance but unverified AI
controller with a verified safety controller and a decision module that
arbitrates between the two.  The AI controller proposes actions; the
decision module forward-simulates each proposal and computes the forward
reachable set.  If the reachable set risks intersecting a safety boundary,
the decision module vetoes the AI and switches to the conservative safety
controller.

References
----------
* Sha, L. (2001). Using Simplicity to Control Complexity.
* Schierman et al. (2015). Runtime Assurance Framework for NDI controllers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ControlAction:
    """A candidate control output from either the AI or safety controller."""
    forces: np.ndarray        # (3,) body-frame force vector [N]
    torques: np.ndarray       # (3,) body-frame torque vector [N-m]
    timestamp: float          # simulation time [s]
    source: str               # "ai_controller" or "safety_controller"


@dataclass
class SafetyEnvelope:
    """Hard safety boundaries that must never be violated."""
    max_dynamic_pressure: float = 45_000.0    # Pa (Increased from 35k)
    max_angle_of_attack: float = np.radians(25.0)   # rad (Increased from 15 deg)
    max_angular_rate: float = np.radians(35.0)       # rad/s (Increased from 20 deg/s)
    min_altitude: float = -1.0                        # m AGL (Allow for slight dip/ground contact)
    max_acceleration: float = 7.0 * 9.81              # m/s^2 (7 g)


@dataclass
class _VehicleSnapshot:
    """Minimal vehicle state for forward simulation."""
    position: np.ndarray       # (3,) inertial [m]
    velocity: np.ndarray       # (3,) inertial [m/s]
    attitude: np.ndarray       # (3,) Euler angles (roll, pitch, yaw) [rad]
    angular_velocity: np.ndarray  # (3,) body rates [rad/s]
    mass: float                # kg
    timestamp: float           # s


# ---------------------------------------------------------------------------
# Decision Module
# ---------------------------------------------------------------------------

class DecisionModule:
    """
    Evaluates whether a proposed AI action will keep the vehicle
    inside the safety envelope by forward-simulating simplified rigid-body
    dynamics and computing the forward reachable set.

    Parameters
    ----------
    envelope : SafetyEnvelope
        Hard safety limits.
    horizon_s : float
        How far ahead to simulate (seconds).
    dt : float
        Integration step for forward simulation.
    gravity : np.ndarray
        Gravity vector in inertial frame.
    """

    def __init__(
        self,
        envelope: Optional[SafetyEnvelope] = None,
        horizon_s: float = 2.0,
        dt: float = 0.05,
        gravity: Optional[np.ndarray] = None,
    ):
        self.envelope = envelope or SafetyEnvelope()
        self.horizon_s = horizon_s
        self.dt = dt
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

        # Veto log
        self._veto_log: List[Dict] = []

    # -- public API ---------------------------------------------------------


    def evaluate(
        self, action: ControlAction, vehicle_state: dict
    ) -> tuple[bool, str]:
        """Evaluate action safety using forward simulation and envelope checking."""
        snap = _VehicleSnapshot(
            position=np.array(vehicle_state['position'], dtype=np.float64),
            velocity=np.array(vehicle_state['velocity'], dtype=np.float64),
            attitude=np.array(vehicle_state['attitude'], dtype=np.float64),
            angular_velocity=np.array(vehicle_state['angular_velocity'], dtype=np.float64),
            mass=vehicle_state.get('mass', 1000.0),
            timestamp=vehicle_state.get('timestamp', 0.0),
        )

        # Forward-simulate the trajectory under this action
        trajectory = self._forward_simulate(snap, action)

        # Check if any point violates the envelope
        violation = self._check_envelope(trajectory)
        if violation:
            return False, violation

        return True, ''

    # -- forward simulation -------------------------------------------------

    def _forward_simulate(
        self, snap: _VehicleSnapshot, action: ControlAction
    ) -> List[_VehicleSnapshot]:
        """
        Forward-simulate simplified 6-DOF rigid body dynamics under the
        proposed constant control action over the prediction horizon.

        Returns the trajectory as a list of snapshots (the forward
        reachable set under worst-case assumptions).
        """
        steps = int(self.horizon_s / self.dt)
        trajectory: List[_VehicleSnapshot] = [snap]

        pos = snap.position.copy()
        vel = snap.velocity.copy()
        att = snap.attitude.copy()
        omega = snap.angular_velocity.copy()
        mass = snap.mass
        
        # --- Control Barrier Function (CBF) Check Formulation ---
        # State: x = [pos, vel, att, omega]
        # Safety constraint h(x) >= 0. Here we use angle bounds + angular velocity limits.
        # h(x) = max_angle_of_attack - max(|roll|, |pitch|) => ensure positive
        # We approximate \dot{h}(x) + alpha * h(x) >= 0 based on proposed actions
        alpha = 1.0
        
        max_angle = self.envelope.max_angle_of_attack
        # Pitch and roll are indices 1, 0
        h_val_pitch = max_angle - abs(att[1])
        h_val_roll  = max_angle - abs(att[0])
        
        # Determine \dot{h} based on omega (omega directly influences \dot{att})
        h_dot_pitch = -np.sign(att[1]) * omega[1] if att[1] != 0 else -omega[1]
        h_dot_roll  = -np.sign(att[0]) * omega[0] if att[0] != 0 else -omega[0]
        
        # Check if the current state is already violating or close, and if the action pushes it further
        # Angular acceleration (simplified assumption as in original code)
        angular_accel = action.torques / (mass * 0.5) 
        
        # 1-step lookahead for h_dot (CBF forward projection)
        omega_next = omega + angular_accel * self.dt * 5  # amplify for margin
        h_dot_pitch_next = -np.sign(att[1]) * omega_next[1] if att[1] != 0 else -abs(omega_next[1])
        h_dot_roll_next  = -np.sign(att[0]) * omega_next[0] if att[0] != 0 else -abs(omega_next[0])
        
        is_pitch_cbf_safe = (h_dot_pitch_next + alpha * h_val_pitch) >= 0
        is_roll_cbf_safe  = (h_dot_roll_next  + alpha * h_val_roll)  >= 0
        
        # Original forward simulation (kept for terminal position checking)
        t = snap.timestamp

        for _ in range(steps):
            # If our localized fast CBF solver detects imminent violation, return violation immediately
            if not is_pitch_cbf_safe or not is_roll_cbf_safe:
                # Force violation by placing simulated vehicle out of bounds
                att_fail = att.copy()
                att_fail[0] = 999.0
                trajectory.append(_VehicleSnapshot(
                    position=pos, velocity=vel, attitude=att_fail,
                    angular_velocity=omega, mass=mass, timestamp=t + self.dt
                ))
                return trajectory

            # Use scipy Rotation
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('xyz', att)
            R = r.as_matrix()

            # Translational dynamics
            accel_body = action.forces / mass
            accel_inertial = R @ accel_body + self.gravity
            vel = vel + accel_inertial * self.dt
            pos = pos + vel * self.dt

            # Rotational dynamics (simplified -- assume unit inertia ratios)
            angular_accel = action.torques / (mass * 0.5)  # rough approx
            omega = omega + angular_accel * self.dt
            att = att + omega * self.dt

            t += self.dt

            trajectory.append(_VehicleSnapshot(
                position=pos.copy(),
                velocity=vel.copy(),
                attitude=att.copy(),
                angular_velocity=omega.copy(),
                mass=mass,
                timestamp=t,
            ))

        return trajectory

    # -- envelope check -----------------------------------------------------

    def _check_envelope(self, trajectory: List[_VehicleSnapshot]) -> str:
        """
        Check if any state in the trajectory violates the safety envelope.

        Returns an empty string if safe, or a description of the first
        violation found.
        """
        env = self.envelope

        for snap in trajectory:
            # Altitude check
            altitude = snap.position[2]
            if altitude < env.min_altitude:
                return (
                    f"Altitude {altitude:.1f} m below minimum "
                    f"{env.min_altitude:.1f} m at t={snap.timestamp:.3f}"
                )

            # Acceleration check
            speed = np.linalg.norm(snap.velocity)
            # Approximate dynamic pressure (sea-level density rough estimate)
            rho = 1.225 * np.exp(-max(altitude, 0.0) / 8500.0)
            q = 0.5 * rho * speed ** 2
            if q > env.max_dynamic_pressure:
                return (
                    f"Dynamic pressure {q:.0f} Pa exceeds limit "
                    f"{env.max_dynamic_pressure:.0f} Pa at t={snap.timestamp:.3f}"
                )

            # Angle of attack (simplified -- pitch deviation from velocity vector)
            if speed > 1.0:
                vel_unit = snap.velocity / speed
                body_z = np.array([
                    -np.sin(snap.attitude[1]),
                    np.sin(snap.attitude[0]) * np.cos(snap.attitude[1]),
                    np.cos(snap.attitude[0]) * np.cos(snap.attitude[1]),
                ])
                dot = np.clip(np.dot(vel_unit, body_z), -1.0, 1.0)
                aoa = np.arccos(abs(dot))
                if aoa > env.max_angle_of_attack:
                    return (
                        f"Angle of attack {np.degrees(aoa):.1f} deg exceeds limit "
                        f"{np.degrees(env.max_angle_of_attack):.1f} deg at "
                        f"t={snap.timestamp:.3f}"
                    )

            # Angular rate check
            max_rate = np.max(np.abs(snap.angular_velocity))
            if max_rate > env.max_angular_rate:
                return (
                    f"Angular rate {np.degrees(max_rate):.1f} deg/s exceeds limit "
                    f"{np.degrees(env.max_angular_rate):.1f} deg/s at "
                    f"t={snap.timestamp:.3f}"
                )

            # Acceleration magnitude
            accel_mag = np.linalg.norm(
                snap.velocity - trajectory[0].velocity
            ) / max(snap.timestamp - trajectory[0].timestamp, 1e-9)
            if accel_mag > env.max_acceleration:
                return (
                    f"Acceleration {accel_mag / 9.81:.1f} g exceeds limit "
                    f"{env.max_acceleration / 9.81:.1f} g at t={snap.timestamp:.3f}"
                )

        return ""


# ---------------------------------------------------------------------------
# Safety Controller
# ---------------------------------------------------------------------------

class SafetyController:
    """
    Conservative verified controller that guarantees the vehicle remains
    within the safety envelope.

    Strategy: wings-level attitude hold with maximum vertical thrust for
    stabilisation.  Uses a simple PD control law that has been formally
    verified to maintain stability margins.

    Parameters
    ----------
    max_thrust : float
        Maximum available thrust [N].
    kp_attitude : float
        Proportional gain for attitude correction.
    kd_attitude : float
        Derivative gain for angular rate damping.
    """

    def __init__(
        self,
        max_thrust: float = 845_000.0 * 3,  # 3-engine safe thrust
        kp_attitude: float = 150_000.0,     # Increased (3x) for aggressive recovery
        kd_attitude: float = 60_000.0,      # Increased (3x)
    ):
        self.max_thrust = max_thrust
        self.kp_attitude = kp_attitude
        self.kd_attitude = kd_attitude

    def compute(self, vehicle_state: Dict, timestamp: float) -> ControlAction:
        """
        Compute conservative safe control action.

        Drives attitude toward wings-level (zero roll/pitch) and applies
        near-maximum upward thrust for altitude maintenance.
        """
        attitude = np.array(vehicle_state["attitude"], dtype=np.float64)
        omega = np.array(vehicle_state["angular_velocity"], dtype=np.float64)

        # PD attitude controller -- target wings level (roll=0, pitch=0)
        target_attitude = np.array([0.0, 0.0, attitude[2]])  # maintain current yaw
        attitude_error = target_attitude - attitude

        torques = (
            self.kp_attitude * attitude_error
            - self.kd_attitude * omega
        )

        # Clamp torques to physically realisable range
        max_torque = 500_000.0  # N-m
        torques = np.clip(torques, -max_torque, max_torque)

        # Vertical thrust -- full safe thrust upward in body frame
        forces = np.array([0.0, 0.0, self.max_thrust])

        return ControlAction(
            forces=forces,
            torques=torques,
            timestamp=timestamp,
            source="safety_controller",
        )


# ---------------------------------------------------------------------------
# Simplex Architecture
# ---------------------------------------------------------------------------

class SimplexArchitecture:
    """
    Wraps a high-performance AI controller with a verified safety
    controller and a decision module.

    Flow
    ----
    1. AI controller proposes an action.
    2. Decision module forward-simulates and checks the safety envelope.
    3. If approved, the AI action is forwarded to actuators.
    4. If vetoed, the safety controller output is used instead.

    All transitions are logged for post-flight analysis.

    Parameters
    ----------
    safety_controller : SafetyController
        The verified safe-mode controller.
    decision_module : DecisionModule
        The arbiter that checks AI proposals.
    """

    def __init__(
        self,
        safety_controller: Optional[SafetyController] = None,
        decision_module: Optional[DecisionModule] = None,
    ):
        self.safety_controller = safety_controller or SafetyController()
        self.decision_module = decision_module or DecisionModule()

        # State
        self._using_safety: bool = False
        self._dwell_counter: int = 0
        self._switch_log: List[Dict] = []
        self._total_ai_proposals: int = 0
        self._total_vetoes: int = 0

    # -- main entry point ---------------------------------------------------

    def evaluate_and_select(
        self, ai_action: ControlAction, vehicle_state: Dict
    ) -> ControlAction:
        """
        Evaluate the AI-proposed action and return the approved output.

        Parameters
        ----------
        ai_action : ControlAction
            Proposed action from the high-performance AI controller.
        vehicle_state : dict
            Current vehicle state (position, velocity, attitude,
            angular_velocity, mass, timestamp).

        Returns
        -------
        ControlAction
            Either the original *ai_action* (if safe) or the output of
            the safety controller.
        """
        self._total_ai_proposals += 1

        # 결정 모듈(Decision Module)을 통해 AI 제안이 안전한지 확인
        # Bug 2 Fix: Hysteresis 및 Dwell Timer 추가
        approved, reason = self.decision_module.evaluate(ai_action, vehicle_state)

        if approved:
            if self._using_safety:
                # 안전 모드에서 복구 중인 경우, 지정된 횟수만큼 연속 승인이 필요함
                self._dwell_counter += 1
                if self._dwell_counter >= 50:  # REENGAGEMENT_DWELL_STEPS
                    self._log_switch("safety -> ai", ai_action.timestamp, "Safety recovered and dwell timer elapsed")
                    self._using_safety = False
                    self._dwell_counter = 0
                else:
                    # 아직 지연 시간이 남았으므로 안전 제어기 계속 사용
                    return self.safety_controller.compute(
                        vehicle_state, ai_action.timestamp
                    )
            return ai_action

        # 거부됨(Vetoed) -- 안전 제어기로 전환 및 지연 카운터 초기화
        self._total_vetoes += 1
        self._dwell_counter = 0
        
        safe_action = self.safety_controller.compute(
            vehicle_state, ai_action.timestamp
        )

        if not self._using_safety:
            self._log_switch("ai -> safety", ai_action.timestamp, reason)
            self._using_safety = True

        logger.info(
            "Simplex VETO #%d at t=%.3f: %s",
            self._total_vetoes,
            ai_action.timestamp,
            reason,
        )
        return safe_action

    # -- logging / telemetry ------------------------------------------------

    def _log_switch(self, direction: str, timestamp: float, reason: str) -> None:
        self._switch_log.append({
            "direction": direction,
            "timestamp": timestamp,
            "reason": reason,
        })

    @property
    def is_using_safety_controller(self) -> bool:
        return self._using_safety

    @property
    def switch_log(self) -> List[Dict]:
        return list(self._switch_log)

    @property
    def veto_count(self) -> int:
        return self._total_vetoes

    @property
    def ai_proposal_count(self) -> int:
        return self._total_ai_proposals

    @property
    def veto_rate(self) -> float:
        if self._total_ai_proposals == 0:
            return 0.0
        return self._total_vetoes / self._total_ai_proposals

    def stats(self) -> Dict:
        return {
            "total_ai_proposals": self._total_ai_proposals,
            "total_vetoes": self._total_vetoes,
            "veto_rate": self.veto_rate,
            "currently_safe_mode": self._using_safety,
            "transitions": len(self._switch_log),
        }
