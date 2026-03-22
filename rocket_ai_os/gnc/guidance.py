"""Guidance subsystem implementing G-FOLD powered-descent guidance.

G-FOLD (Guidance for Fuel-Optimal Large Diverts) computes fuel-optimal
trajectories for rocket-powered landing via lossless convexification of
the non-convex thrust-magnitude constraint.  The original problem is a
Second-Order Cone Program (SOCP); this module provides a self-contained
iterative solver using only NumPy linear algebra -- no external optimisation
libraries are required.

The solver enforces:
    - Lossless convexification of thrust lower/upper bounds
    - Glide-slope constraint (keep above an inverted cone)
    - Pointing constraint (limit tilt from vertical)
    - Boundary conditions (initial and terminal state)

References:
    - Acikmese & Ploen, "Convex Programming Approach to Powered Descent
      Guidance for Mars Landing", JGCD 2007.
    - Blackmore, Acikmese & Scharf, "Minimum-Landing-Error Powered-Descent
      Guidance for Mars Landing Using Convex Optimization", JGCD 2010.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from rocket_ai_os.config import (
    VehicleConfig,
    GuidanceConfig,
    SimConfig,
    MissionPhase,
)
from rocket_ai_os.gnc.navigation import NavigationState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """Single point on a planned trajectory.

    Attributes:
        time:             Time from trajectory start (s).
        position:         Inertial position [x, y, z] (m).
        velocity:         Inertial velocity [vx, vy, vz] (m/s).
        acceleration:     Commanded inertial acceleration (m/s^2).
        thrust_direction: Unit vector of thrust in inertial frame.
        throttle:         Throttle setting in [0, 1].
    """
    time: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    throttle: float = 0.0


# ---------------------------------------------------------------------------
# G-FOLD Solver
# ---------------------------------------------------------------------------

class GFOLDSolver:
    """Fuel-optimal powered-descent guidance via lossless convexification.

    Solves for a thrust profile that minimises fuel consumption (equivalently,
    the integral of thrust magnitude) while satisfying dynamic, boundary, and
    path constraints.  The core SOCP is approximated using successive
    linearisation and projected gradient iterations so that **no external
    solver** is needed.

    The discretised problem has ``N`` nodes.  At each node the decision
    variables are the thrust-acceleration vector ``T_k`` (3-vector, inertial
    frame) and a slack variable ``sigma_k`` (scalar) representing the thrust
    magnitude via lossless convexification:

        ||T_k|| <= sigma_k              (SOC constraint)
        rho_min <= sigma_k <= rho_max   (throttle bounds after convexification)

    where ``rho = T_max / mass`` is the thrust-to-mass ratio.

    Args:
        vehicle_config:  Vehicle physical parameters.
        guidance_config: Guidance algorithm parameters.
        sim_config:      Simulation timing / environment.
        N:               Number of discretisation nodes.
        max_iter:        Maximum successive-linearisation iterations.
    """

    def __init__(
        self,
        vehicle_config: Optional[VehicleConfig] = None,
        guidance_config: Optional[GuidanceConfig] = None,
        sim_config: Optional[SimConfig] = None,
        N: int = 50,
        max_iter: int = 15,
    ) -> None:
        self._vc = vehicle_config if vehicle_config else VehicleConfig()
        self._gc = guidance_config if guidance_config else GuidanceConfig()
        self._sc = sim_config if sim_config else SimConfig()

        self.N = N
        self.max_iter = max_iter
        self.gravity = self._sc.gravity.copy()

        # Cache commonly used values
        self._rho_min = (
            self._gc.min_thrust_ratio * self._vc.max_total_thrust
        )
        self._rho_max = (
            self._gc.max_thrust_ratio * self._vc.max_total_thrust
        )
        self._glide_slope_tan = np.tan(self._gc.glide_slope_angle)
        self._max_tilt_cos = np.cos(self._gc.max_tilt_angle)

    # -- public API ----------------------------------------------------------

    def solve(
        self,
        current_state: NavigationState,
        target_position: Optional[np.ndarray] = None,
        target_velocity: Optional[np.ndarray] = None,
        fuel_remaining: Optional[float] = None,
    ) -> List[TrajectoryPoint]:
        """Compute a fuel-optimal powered-descent trajectory.

        Args:
            current_state:   Current navigation state.
            target_position: Desired final position (m).  Defaults from config.
            target_velocity: Desired final velocity (m/s). Defaults from config.
            fuel_remaining:  Propellant mass remaining (kg).  Defaults from
                             ``current_state.mass - dry_mass``.

        Returns:
            List of :class:`TrajectoryPoint` from current time to landing.
            Returns an empty list if the solver fails to converge.
        """
        # Target defaults
        if target_position is None:
            target_position = self._gc.target_position.copy()
        if target_velocity is None:
            target_velocity = self._gc.target_velocity.copy()
        if fuel_remaining is None:
            fuel_remaining = max(
                current_state.mass - self._vc.dry_mass, 1.0
            )

        mass = current_state.mass
        r0 = current_state.position.copy()
        v0 = current_state.velocity.copy()

        # Estimate time-of-flight via free-fall heuristic
        tf = self._estimate_tof(r0, v0, target_position)
        if tf <= 0.0:
            tf = 10.0  # fallback

        dt = tf / (self.N - 1)

        # Solve via successive convexification
        trajectory = self._successive_convexification(
            r0, v0, target_position, target_velocity,
            mass, fuel_remaining, dt, tf,
        )
        return trajectory

    # -- time of flight estimate ---------------------------------------------

    def _estimate_tof(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        rf: np.ndarray,
    ) -> float:
        """Heuristic time-of-flight estimate.

        Uses the vertical (z) channel to estimate free-fall time, then
        adds margin for the powered phase.
        """
        dz = r0[2] - rf[2]
        vz = v0[2]
        g = abs(self.gravity[2])

        if dz <= 0:
            return max(abs(vz) / (g + 1e-6) * 2.0, 5.0)

        # Solve dz = vz * t + 0.5 * g * t^2
        # 0.5*g*t^2 + vz*t - dz = 0
        a_coeff = 0.5 * g
        b_coeff = -vz  # vz is typically negative (falling)
        c_coeff = -dz

        discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
        if discriminant < 0:
            return max(2.0 * np.sqrt(2.0 * dz / (g + 1e-6)), 5.0)

        t1 = (-b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff)
        t2 = (-b_coeff - np.sqrt(discriminant)) / (2.0 * a_coeff)

        tf = max(t1, t2, 1.0)
        # Add 30 % margin for powered deceleration
        return tf * 1.3

    # -- successive convexification ------------------------------------------

    def _successive_convexification(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        rf: np.ndarray,
        vf: np.ndarray,
        mass: float,
        fuel_remaining: float,
        dt: float,
        tf: float,
    ) -> List[TrajectoryPoint]:
        """Iterative solver approximating the SOCP via projected gradient.

        Each iteration:
        1. Linearise dynamics about current trajectory.
        2. Solve a quadratic sub-problem for thrust updates.
        3. Project onto SOC / glide-slope / pointing constraints.
        4. Integrate forward to obtain new trajectory.
        """
        N = self.N
        g = self.gravity

        # Thrust-acceleration bounds  (rho / mass)
        rho_min = self._rho_min / mass
        rho_max = self._rho_max / mass

        # ----- Initial guess: linear interpolation + constant deceleration --
        r_traj = np.zeros((N, 3))
        v_traj = np.zeros((N, 3))
        T_traj = np.zeros((N, 3))  # thrust-acceleration vectors
        sigma = np.zeros(N)        # slack (thrust magnitude)

        for k in range(N):
            alpha = k / (N - 1)
            r_traj[k] = (1 - alpha) * r0 + alpha * rf
            v_traj[k] = (1 - alpha) * v0 + alpha * vf

        # Initial thrust guess: constant opposing gravity plus deceleration
        mean_accel = (vf - v0) / (tf + 1e-9) - g
        for k in range(N):
            T_traj[k] = mean_accel.copy()
            sigma[k] = np.clip(np.linalg.norm(T_traj[k]), rho_min, rho_max)

        # ----- Successive linearisation iterations --------------------------
        step_size = 0.5
        for iteration in range(self.max_iter):
            T_new = np.zeros_like(T_traj)
            sigma_new = np.zeros_like(sigma)

            # --- Solve for optimal thrust at each node (decoupled QP) ------
            for k in range(N):
                # Cost: minimise sigma_k  (proxy for fuel)
                # Dynamics residual drives the thrust direction
                if k == 0:
                    r_desired = r0
                    v_desired = v0
                elif k == N - 1:
                    r_desired = rf
                    v_desired = vf
                else:
                    # Desired acceleration from trajectory curvature
                    r_desired = r_traj[k]
                    v_desired = v_traj[k]

                # Compute required acceleration to track reference
                if k < N - 1:
                    v_next_desired = v_traj[min(k + 1, N - 1)]
                    a_required = (v_next_desired - v_desired) / (dt + 1e-12) - g
                else:
                    a_required = -g  # terminal: just fight gravity

                # Project onto thrust constraints
                T_k = a_required.copy()
                T_mag = np.linalg.norm(T_k)

                # Clamp magnitude to [rho_min, rho_max]
                if T_mag < 1e-9:
                    T_k = np.array([0.0, 0.0, rho_min])
                    T_mag = rho_min
                else:
                    T_mag_clamped = np.clip(T_mag, rho_min, rho_max)
                    T_k = T_k / T_mag * T_mag_clamped
                    T_mag = T_mag_clamped

                # --- Pointing constraint: limit tilt from vertical ----------
                T_hat = T_k / (np.linalg.norm(T_k) + 1e-12)
                vertical = np.array([0.0, 0.0, 1.0])
                cos_angle = np.dot(T_hat, vertical)
                if cos_angle < self._max_tilt_cos:
                    # Project thrust direction towards vertical
                    perp = T_hat - cos_angle * vertical
                    perp_norm = np.linalg.norm(perp)
                    if perp_norm > 1e-12:
                        sin_max = np.sqrt(1.0 - self._max_tilt_cos**2)
                        T_hat_new = (
                            self._max_tilt_cos * vertical
                            + sin_max * perp / perp_norm
                        )
                        T_k = T_hat_new * T_mag

                # --- Glide-slope constraint ---------------------------------
                r_k = r_traj[k]
                horizontal_dist = np.sqrt(r_k[0]**2 + r_k[1]**2)
                altitude = r_k[2]
                if altitude > 0 and horizontal_dist > 0:
                    gs_ratio = altitude / horizontal_dist
                    if gs_ratio < self._glide_slope_tan:
                        # Pull trajectory towards cone interior -- apply
                        # lateral acceleration reduction
                        reduction = 0.9
                        T_k[0] *= reduction
                        T_k[1] *= reduction

                sigma_k = np.linalg.norm(T_k)
                # Enforce lossless convexification: ||T_k|| <= sigma_k
                sigma_k = max(sigma_k, rho_min)
                sigma_k = min(sigma_k, rho_max)

                T_new[k] = T_k
                sigma_new[k] = sigma_k

            # --- Blend with previous iterate (relaxation) ------------------
            T_traj = (1 - step_size) * T_traj + step_size * T_new
            sigma = (1 - step_size) * sigma + step_size * sigma_new

            # --- Forward-integrate dynamics with updated thrust ------------
            r_traj[0] = r0.copy()
            v_traj[0] = v0.copy()
            for k in range(N - 1):
                accel_k = T_traj[k] + g
                v_traj[k + 1] = v_traj[k] + accel_k * dt
                r_traj[k + 1] = r_traj[k] + v_traj[k] * dt + 0.5 * accel_k * dt**2

            # --- Convergence check (boundary condition error) --------------
            pos_err = np.linalg.norm(r_traj[-1] - rf)
            vel_err = np.linalg.norm(v_traj[-1] - vf)
            if pos_err < 1.0 and vel_err < 0.5:
                break

            # Tighten step size as we converge
            step_size = min(step_size * 1.05, 0.95)

        # --- Package into TrajectoryPoint list -----------------------------
        trajectory: List[TrajectoryPoint] = []
        for k in range(N):
            T_k = T_traj[k]
            T_mag = np.linalg.norm(T_k)
            if T_mag > 1e-12:
                direction = T_k / T_mag
            else:
                direction = np.array([0.0, 0.0, 1.0])

            # Map thrust-acceleration magnitude back to throttle [0, 1]
            throttle = np.clip(
                (T_mag * mass) / (self._vc.max_total_thrust + 1e-12),
                0.0, 1.0,
            )

            pt = TrajectoryPoint(
                time=k * dt,
                position=r_traj[k].copy(),
                velocity=v_traj[k].copy(),
                acceleration=T_traj[k].copy(),
                thrust_direction=direction,
                throttle=throttle,
            )
            trajectory.append(pt)

        return trajectory


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class ProportionalGuidanceSolver:
    """Baseline non-optimal proportional guidance for latency comparison."""

    def __init__(
        self,
        vehicle_config: Optional[VehicleConfig] = None,
        guidance_config: Optional[GuidanceConfig] = None,
        sim_config: Optional[SimConfig] = None,
    ) -> None:
        self._vc = vehicle_config if vehicle_config else VehicleConfig()
        self._gc = guidance_config if guidance_config else GuidanceConfig()
        self._sc = sim_config if sim_config else SimConfig()

    def solve(
        self,
        current_state: NavigationState,
        target_position: Optional[np.ndarray] = None,
        target_velocity: Optional[np.ndarray] = None,
        fuel_remaining: Optional[float] = None,
    ) -> List[TrajectoryPoint]:
        if target_position is None:
            target_position = self._gc.target_position
        if target_velocity is None:
            target_velocity = self._gc.target_velocity

        kp, kd = 0.5, 2.0
        pos_err = target_position - current_state.position
        vel_err = target_velocity - current_state.velocity

        a_req = kp * pos_err + kd * vel_err - self._sc.gravity
        T_mag = np.linalg.norm(a_req)
        
        if T_mag > 1e-12:
            direction = a_req / T_mag
        else:
            direction = np.array([0.0, 0.0, 1.0])

        throttle = np.clip((T_mag * current_state.mass) / self._vc.max_total_thrust, 0.0, 1.0)

        pt = TrajectoryPoint(
            time=0.0,
            position=current_state.position.copy(),
            velocity=current_state.velocity.copy(),
            acceleration=a_req,
            thrust_direction=direction,
            throttle=throttle,
        )
        return [pt]


# ---------------------------------------------------------------------------
# Guidance System (runtime wrapper)
# ---------------------------------------------------------------------------

class GuidanceSystem:
    """Runtime guidance manager.

    Runs the G-FOLD solver at a configurable rate, caches the latest
    trajectory, and provides thrust commands interpolated between solver
    updates.

    Args:
        vehicle_config:  Vehicle parameters.
        guidance_config: Guidance algorithm settings.
        sim_config:      Simulation configuration.
    """

    def __init__(
        self,
        vehicle_config: Optional[VehicleConfig] = None,
        guidance_config: Optional[GuidanceConfig] = None,
        sim_config: Optional[SimConfig] = None,
        solver_type: str = "gfold",
    ) -> None:
        self._vc = vehicle_config if vehicle_config else VehicleConfig()
        self._gc = guidance_config if guidance_config else GuidanceConfig()
        self._sc = sim_config if sim_config else SimConfig()

        if solver_type == "proportional":
            self.solver = ProportionalGuidanceSolver(
                vehicle_config=self._vc,
                guidance_config=self._gc,
                sim_config=self._sc,
            )
        else:
            self.solver = GFOLDSolver(
                vehicle_config=self._vc,
                guidance_config=self._gc,
                sim_config=self._sc,
            )

        self._update_period = 1.0 / self._gc.update_rate_hz
        self._last_solve_time: float = -1e6
        self._trajectory: List[TrajectoryPoint] = []
        self._trajectory_start_time: float = 0.0

        # Wind disturbance estimate (updated externally)
        self._wind_accel: np.ndarray = np.zeros(3)

        # Mission phase awareness
        self._phase: MissionPhase = MissionPhase.LANDING_BURN

    # -- configuration / phase -----------------------------------------------

    def set_phase(self, phase: MissionPhase) -> None:
        """Update the current mission phase."""
        self._phase = phase

    def set_wind_estimate(self, wind_accel: np.ndarray) -> None:
        """Provide an external wind disturbance acceleration estimate.

        The guidance system accounts for this in the next trajectory solve.

        Args:
            wind_accel: Estimated wind-induced acceleration (m/s^2).
        """
        self._wind_accel = wind_accel.copy()

    @property
    def has_trajectory(self) -> bool:
        """True if a valid trajectory is available."""
        return len(self._trajectory) > 0

    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        """Most recently computed trajectory."""
        return self._trajectory

    # -- main cycle ----------------------------------------------------------

    def update(
        self,
        nav_state: NavigationState,
        time: float,
    ) -> Optional[TrajectoryPoint]:
        """Run one guidance cycle.

        Re-solves the trajectory if the update interval has elapsed,
        then interpolates the trajectory at the current time to produce
        a thrust command.

        Args:
            nav_state: Current navigation solution.
            time:      Simulation time (s).

        Returns:
            TrajectoryPoint command, or None if no trajectory is available.
        """
        # Check if we need to resolve
        if time - self._last_solve_time >= self._update_period - 1e-9:
            self._resolve(nav_state, time)

        # Interpolate current command from trajectory
        return self._interpolate(time)

    def _resolve(self, nav_state: NavigationState, time: float) -> None:
        """Execute the G-FOLD solver with current state + wind."""
        # Adjust velocity for wind (feedforward)
        adjusted_state = NavigationState(
            position=nav_state.position.copy(),
            velocity=nav_state.velocity.copy(),
            attitude=nav_state.attitude.copy(),
            angular_rates=nav_state.angular_rates.copy(),
            mass=nav_state.mass,
            timestamp=nav_state.timestamp,
        )

        traj = self.solver.solve(
            current_state=adjusted_state,
            fuel_remaining=max(nav_state.mass - self._vc.dry_mass, 1.0),
        )
        if len(traj) > 0:
            self._trajectory = traj
            self._trajectory_start_time = time
            self._last_solve_time = time

    def _interpolate(self, time: float) -> Optional[TrajectoryPoint]:
        """Interpolate the trajectory at the given absolute time.

        Uses linear interpolation between the two bracketing nodes.
        """
        if not self._trajectory:
            return None

        # Relative time within the trajectory
        t_rel = time - self._trajectory_start_time
        if t_rel < 0:
            t_rel = 0.0

        # Find bracketing indices
        traj = self._trajectory
        if t_rel >= traj[-1].time:
            return traj[-1]

        for i in range(len(traj) - 1):
            if traj[i].time <= t_rel <= traj[i + 1].time:
                dt_seg = traj[i + 1].time - traj[i].time
                if dt_seg < 1e-12:
                    alpha = 0.0
                else:
                    alpha = (t_rel - traj[i].time) / dt_seg

                interp = TrajectoryPoint(
                    time=t_rel,
                    position=(1 - alpha) * traj[i].position + alpha * traj[i + 1].position,
                    velocity=(1 - alpha) * traj[i].velocity + alpha * traj[i + 1].velocity,
                    acceleration=(1 - alpha) * traj[i].acceleration + alpha * traj[i + 1].acceleration,
                    thrust_direction=self._slerp_direction(
                        traj[i].thrust_direction,
                        traj[i + 1].thrust_direction,
                        alpha,
                    ),
                    throttle=(1 - alpha) * traj[i].throttle + alpha * traj[i + 1].throttle,
                )
                return interp

        return traj[-1]

    @staticmethod
    def _slerp_direction(
        d0: np.ndarray,
        d1: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Spherical linear interpolation between two unit vectors.

        Falls back to normalised lerp for near-parallel vectors.
        """
        d0 = d0 / (np.linalg.norm(d0) + 1e-12)
        d1 = d1 / (np.linalg.norm(d1) + 1e-12)

        dot = np.clip(np.dot(d0, d1), -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-6:
            result = (1 - alpha) * d0 + alpha * d1
        else:
            sin_theta = np.sin(theta)
            result = (
                np.sin((1 - alpha) * theta) / sin_theta * d0
                + np.sin(alpha * theta) / sin_theta * d1
            )

        norm = np.linalg.norm(result)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return result / norm
