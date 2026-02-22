"""
Debris Detection and Collision Avoidance subsystem.

Tracks space debris against the star background using simulated optical
sensors, propagates orbits with simplified Keplerian mechanics, assesses
collision probability via Monte-Carlo-lite sampling, and generates
avoidance manoeuvre burn plans when threat thresholds are exceeded.

The CollisionAssessment class implements an Estimated Line-Of-Variance
(ELVO) approach for close-approach prediction with covariance-aware
probability calculation.

References:
    - Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    - Alfano, "A Numerical Implementation of Spherical Object Collision
      Probability", Journal of the Astronautical Sciences, 2005.
    - ESA Space Debris Office, "Collision Avoidance Operations Handbook".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SimConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU_EARTH: float = 3.986004418e14   # Earth gravitational parameter (m^3/s^2)
R_EARTH: float = 6.371e6           # Earth mean radius (m)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Observed debris object being tracked by the optical sensor.

    Attributes:
        obj_id:        Unique tracking identifier.
        position:      Inertial position [x, y, z] (m).
        velocity:      Inertial velocity [vx, vy, vz] (m/s).
        size_estimate: Estimated diameter (m).
        threat_level:  Classified threat in [0, 1] (1 = highest threat).
        covariance:    Position uncertainty covariance (3x3) (m^2).
        last_updated:  Time of last observation (s).
    """
    obj_id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size_estimate: float = 0.1
    threat_level: float = 0.0
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 100.0
    )
    last_updated: float = 0.0


@dataclass
class CollisionPrediction:
    """Result of a collision assessment for one tracked object.

    Attributes:
        obj_id:                Object identifier (matches TrackedObject).
        time_to_closest_approach: Time until closest approach (s).
        miss_distance:         Predicted closest approach distance (m).
        collision_probability: Estimated probability of collision [0, 1].
        relative_velocity:     Relative speed at closest approach (m/s).
        needs_maneuver:        True if a manoeuvre is recommended.
    """
    obj_id: int
    time_to_closest_approach: float
    miss_distance: float
    collision_probability: float
    relative_velocity: float = 0.0
    needs_maneuver: bool = False


@dataclass
class BurnPlan:
    """Collision avoidance manoeuvre plan.

    Attributes:
        delta_v:          Delta-v vector in inertial frame (m/s).
        delta_v_magnitude: Scalar delta-v magnitude (m/s).
        burn_direction:   Unit vector burn direction.
        burn_start_time:  Planned ignition time (s).
        burn_duration:    Estimated burn duration (s).
        fuel_required:    Estimated propellant mass (kg).
        target_obj_id:    ID of the threat being avoided.
        post_miss_distance: Predicted miss distance after manoeuvre (m).
    """
    delta_v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_v_magnitude: float = 0.0
    burn_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    burn_start_time: float = 0.0
    burn_duration: float = 0.0
    fuel_required: float = 0.0
    target_obj_id: int = -1
    post_miss_distance: float = 0.0


# ---------------------------------------------------------------------------
# Keplerian orbit propagation helpers
# ---------------------------------------------------------------------------

def _propagate_kepler(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    mu: float = MU_EARTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate a state vector using a simplified Keplerian two-body model.

    Uses the f-and-g series expansion truncated to second order, suitable
    for short-duration propagation intervals (~minutes).

    Args:
        position: Inertial position (3,) in metres.
        velocity: Inertial velocity (3,) in m/s.
        dt:       Propagation interval (s).
        mu:       Gravitational parameter (m^3/s^2).

    Returns:
        Tuple of (new_position, new_velocity).
    """
    r = np.linalg.norm(position)
    if r < 1.0:
        # Degenerate -- return unchanged
        return position.copy(), velocity.copy()

    r3 = r ** 3

    # f-and-g series (second order)
    f = 1.0 - 0.5 * mu / r3 * dt ** 2
    g = dt - (1.0 / 6.0) * mu / r3 * dt ** 3
    f_dot = -mu / r3 * dt
    g_dot = 1.0 - 0.5 * mu / r3 * dt ** 2

    new_pos = f * position + g * velocity
    new_vel = f_dot * position + g_dot * velocity

    return new_pos, new_vel


def _time_to_closest_approach(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
) -> float:
    """Estimate time to closest approach along a linear trajectory.

    Minimises |r + v*t|^2 analytically.

    Args:
        rel_pos: Relative position (3,) in m.
        rel_vel: Relative velocity (3,) in m/s.

    Returns:
        Time to closest approach (s), clamped to [0, inf).
    """
    v2 = np.dot(rel_vel, rel_vel)
    if v2 < 1e-12:
        return 0.0
    t_ca = -np.dot(rel_pos, rel_vel) / v2
    return float(max(t_ca, 0.0))


# ---------------------------------------------------------------------------
# Debris Tracker (optical sensor simulation)
# ---------------------------------------------------------------------------

class DebrisTracker:
    """Track debris objects against the star background using a simulated
    optical sensor.

    The tracker maintains a catalogue of tracked objects, propagates their
    states between observations, and processes new sensor detections to
    update or initialise tracks.

    Args:
        fov_half_angle:    Sensor field of view half-angle (rad).
        detection_limit:   Minimum detectable object size (m).
        max_range:         Maximum tracking range (m).
        position_noise_std: 1-sigma position measurement noise (m).
        velocity_noise_std: 1-sigma velocity measurement noise (m/s).
        max_coast_time:    Drop a track after this many seconds without update.
        seed:              RNG seed.
    """

    def __init__(
        self,
        fov_half_angle: float = np.radians(15.0),
        detection_limit: float = 0.01,
        max_range: float = 50_000.0,
        position_noise_std: float = 50.0,
        velocity_noise_std: float = 5.0,
        max_coast_time: float = 300.0,
        seed: int = 42,
    ) -> None:
        self.fov_half_angle = fov_half_angle
        self.detection_limit = detection_limit
        self.max_range = max_range
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.max_coast_time = max_coast_time
        self._rng = np.random.default_rng(seed)

        self._catalogue: Dict[int, TrackedObject] = {}
        self._next_id: int = 0

    @property
    def catalogue(self) -> Dict[int, TrackedObject]:
        """Current debris catalogue."""
        return dict(self._catalogue)

    # -- orbit propagation of catalogue --------------------------------------

    def _propagate_catalogue(self, dt: float) -> None:
        """Propagate all tracked objects forward using Keplerian mechanics.

        Args:
            dt: Time step (s).
        """
        for obj in self._catalogue.values():
            new_pos, new_vel = _propagate_kepler(obj.position, obj.velocity, dt)
            obj.position = new_pos
            obj.velocity = new_vel
            # Inflate covariance over coast period
            obj.covariance += np.eye(3) * (self.position_noise_std ** 2) * (dt / 60.0)

    # -- sensor detection simulation -----------------------------------------

    def _simulate_detection(
        self,
        true_position: np.ndarray,
        true_velocity: np.ndarray,
        size: float,
        vehicle_position: np.ndarray,
        vehicle_attitude_z: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Determine if an object is detectable and return noisy measurement.

        Args:
            true_position:      Object true inertial position (m).
            true_velocity:      Object true inertial velocity (m/s).
            size:               Object diameter (m).
            vehicle_position:   Vehicle inertial position (m).
            vehicle_attitude_z: Unit vector of sensor boresight.

        Returns:
            Tuple (measured_position, measured_velocity) or None.
        """
        rel_pos = true_position - vehicle_position
        distance = np.linalg.norm(rel_pos)

        if distance < 1.0 or distance > self.max_range:
            return None

        if size < self.detection_limit:
            return None

        # Check field of view
        los_dir = rel_pos / distance
        boresight_norm = np.linalg.norm(vehicle_attitude_z)
        if boresight_norm < 1e-12:
            return None
        cos_angle = np.dot(los_dir, vehicle_attitude_z / boresight_norm)
        if cos_angle < np.cos(self.fov_half_angle):
            return None

        # Detection probability: decreases with range and size
        snr = (size / self.detection_limit) * (self.max_range / max(distance, 1.0)) ** 2
        p_detect = float(np.clip(1.0 - np.exp(-snr / 2.0), 0.0, 1.0))
        if self._rng.random() > p_detect:
            return None

        # Noisy measurement (range-dependent noise scaling)
        range_factor = distance / self.max_range
        meas_pos = true_position + self._rng.normal(
            0, self.position_noise_std * (1.0 + range_factor), 3
        )
        meas_vel = true_velocity + self._rng.normal(
            0, self.velocity_noise_std * (1.0 + range_factor), 3
        )

        return meas_pos, meas_vel

    # -- track management ----------------------------------------------------

    def _associate_or_create(
        self,
        meas_pos: np.ndarray,
        meas_vel: np.ndarray,
        size: float,
        time: float,
    ) -> TrackedObject:
        """Associate a detection with an existing track or create a new one.

        Uses nearest-neighbour gating in the position domain.

        Args:
            meas_pos: Measured position (m).
            meas_vel: Measured velocity (m/s).
            size:     Object size estimate (m).
            time:     Current simulation time (s).

        Returns:
            The updated or newly created TrackedObject.
        """
        gate_threshold = 3.0 * self.position_noise_std * 10.0  # generous gate

        best_id: Optional[int] = None
        best_dist = gate_threshold

        for obj_id, obj in self._catalogue.items():
            dist = np.linalg.norm(meas_pos - obj.position)
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id

        if best_id is not None:
            obj = self._catalogue[best_id]
            # Simple weighted update (alpha-beta filter flavour)
            alpha = 0.4
            obj.position = (1 - alpha) * obj.position + alpha * meas_pos
            obj.velocity = (1 - alpha) * obj.velocity + alpha * meas_vel
            obj.size_estimate = (1 - alpha) * obj.size_estimate + alpha * size
            obj.covariance *= (1 - alpha)  # Shrink covariance on update
            obj.last_updated = time
            return obj
        else:
            new_obj = TrackedObject(
                obj_id=self._next_id,
                position=meas_pos.copy(),
                velocity=meas_vel.copy(),
                size_estimate=size,
                covariance=np.eye(3) * (self.position_noise_std ** 2),
                last_updated=time,
            )
            self._catalogue[self._next_id] = new_obj
            self._next_id += 1
            return new_obj

    def _prune_stale_tracks(self, time: float) -> None:
        """Remove tracks that have not been updated within max_coast_time.

        Args:
            time: Current simulation time (s).
        """
        stale_ids = [
            obj_id for obj_id, obj in self._catalogue.items()
            if (time - obj.last_updated) > self.max_coast_time
        ]
        for obj_id in stale_ids:
            del self._catalogue[obj_id]

    # -- public API ----------------------------------------------------------

    def track(
        self,
        sensor_data: List[Dict[str, object]],
        vehicle_position: np.ndarray,
        vehicle_attitude_z: np.ndarray,
        time: float,
        dt: float = 1.0,
    ) -> List[TrackedObject]:
        """Process a batch of sensor detections and return updated catalogue.

        Args:
            sensor_data:  List of dicts with keys:
                            "position": (3,) true inertial position (m),
                            "velocity": (3,) true inertial velocity (m/s),
                            "size":     float diameter (m).
            vehicle_position:   Vehicle inertial position (m).
            vehicle_attitude_z: Vehicle sensor boresight direction.
            time:               Current simulation time (s).
            dt:                 Time since last track update (s).

        Returns:
            List of currently tracked objects.
        """
        # Propagate existing tracks
        self._propagate_catalogue(dt)

        # Process each detection
        for item in sensor_data:
            true_pos = np.asarray(item["position"], dtype=np.float64)
            true_vel = np.asarray(item["velocity"], dtype=np.float64)
            size = float(item["size"])

            result = self._simulate_detection(
                true_pos, true_vel, size,
                vehicle_position, vehicle_attitude_z,
            )
            if result is not None:
                meas_pos, meas_vel = result
                self._associate_or_create(meas_pos, meas_vel, size, time)

        # Prune stale tracks
        self._prune_stale_tracks(time)

        return list(self._catalogue.values())


# ---------------------------------------------------------------------------
# Collision Assessment (ELVO-based)
# ---------------------------------------------------------------------------

class CollisionAssessment:
    """Assess collision risk using Estimated Line-Of-Variance (ELVO) analysis.

    For each tracked object, propagates both vehicle and debris states
    forward to find the time of closest approach, then estimates collision
    probability using Monte-Carlo-lite sampling of the combined position
    uncertainty ellipsoid.

    Args:
        combined_radius:     Sum of vehicle and debris effective radii (m).
        probability_threshold: Probability above which a manoeuvre is flagged.
        propagation_horizon:   Maximum look-ahead time (s).
        propagation_step:      Time step for closest-approach search (s).
        n_monte_carlo:         Number of MC samples for probability estimate.
        seed:                  RNG seed for MC sampling.
    """

    def __init__(
        self,
        combined_radius: float = 10.0,
        probability_threshold: float = 1e-4,
        propagation_horizon: float = 3600.0,
        propagation_step: float = 10.0,
        n_monte_carlo: int = 1000,
        seed: int = 42,
    ) -> None:
        self.combined_radius = combined_radius
        self.probability_threshold = probability_threshold
        self.propagation_horizon = propagation_horizon
        self.propagation_step = propagation_step
        self.n_monte_carlo = n_monte_carlo
        self._rng = np.random.default_rng(seed)

    def _find_closest_approach(
        self,
        vehicle_pos: np.ndarray,
        vehicle_vel: np.ndarray,
        debris_pos: np.ndarray,
        debris_vel: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Find time, distance, and relative speed at closest approach.

        Two-phase search:
        1. Coarse scan over propagation horizon.
        2. Refinement around the coarse minimum using linear TCA estimate.

        Args:
            vehicle_pos: Vehicle position (3,) (m).
            vehicle_vel: Vehicle velocity (3,) (m/s).
            debris_pos:  Debris position (3,) (m).
            debris_vel:  Debris velocity (3,) (m/s).

        Returns:
            Tuple (time_to_ca, miss_distance, relative_speed).
        """
        best_t = 0.0
        best_dist = np.inf
        best_rel_speed = 0.0

        v_pos, v_vel = vehicle_pos.copy(), vehicle_vel.copy()
        d_pos, d_vel = debris_pos.copy(), debris_vel.copy()

        # Phase 1: Coarse scan
        n_steps = int(self.propagation_horizon / self.propagation_step)
        prev_dist = np.linalg.norm(d_pos - v_pos)

        for step in range(1, n_steps + 1):
            t = step * self.propagation_step

            v_pos_t, v_vel_t = _propagate_kepler(vehicle_pos, vehicle_vel, t)
            d_pos_t, d_vel_t = _propagate_kepler(debris_pos, debris_vel, t)

            rel = d_pos_t - v_pos_t
            dist = np.linalg.norm(rel)

            if dist < best_dist:
                best_dist = dist
                best_t = t
                best_rel_speed = np.linalg.norm(d_vel_t - v_vel_t)

            # Early termination if distance is increasing after a minimum
            if dist > prev_dist and prev_dist <= best_dist + 1.0:
                # Already past minimum -- refine below
                pass
            prev_dist = dist

        # Phase 2: Refine with linear TCA around best_t
        v_pos_r, v_vel_r = _propagate_kepler(vehicle_pos, vehicle_vel, best_t)
        d_pos_r, d_vel_r = _propagate_kepler(debris_pos, debris_vel, best_t)

        rel_pos = d_pos_r - v_pos_r
        rel_vel = d_vel_r - v_vel_r
        dt_refine = _time_to_closest_approach(rel_pos, rel_vel)
        t_refined = best_t + dt_refine

        if 0 <= t_refined <= self.propagation_horizon:
            v_pos_f, v_vel_f = _propagate_kepler(vehicle_pos, vehicle_vel, t_refined)
            d_pos_f, d_vel_f = _propagate_kepler(debris_pos, debris_vel, t_refined)
            dist_refined = np.linalg.norm(d_pos_f - v_pos_f)
            if dist_refined < best_dist:
                best_dist = dist_refined
                best_t = t_refined
                best_rel_speed = np.linalg.norm(d_vel_f - v_vel_f)

        return float(best_t), float(best_dist), float(best_rel_speed)

    def _estimate_collision_probability(
        self,
        miss_distance: float,
        combined_covariance: np.ndarray,
    ) -> float:
        """Estimate collision probability via Monte-Carlo sampling.

        Draws position offset samples from the combined uncertainty
        distribution and counts what fraction fall within the combined
        collision radius.

        Args:
            miss_distance:      Nominal miss distance (m).
            combined_covariance: Combined position covariance (3x3) (m^2).

        Returns:
            Estimated collision probability in [0, 1].
        """
        # Ensure covariance is positive semi-definite
        eigvals = np.linalg.eigvalsh(combined_covariance)
        if np.any(eigvals < 0):
            combined_covariance = combined_covariance + np.eye(3) * abs(min(eigvals)) * 1.1

        # Cholesky decomposition for sampling
        try:
            L = np.linalg.cholesky(combined_covariance)
        except np.linalg.LinAlgError:
            # Fallback: diagonal approximation
            L = np.diag(np.sqrt(np.maximum(np.diag(combined_covariance), 1e-6)))

        # Miss distance as a nominal offset vector (along x-axis by convention)
        nominal_offset = np.array([miss_distance, 0.0, 0.0])

        # Monte-Carlo sampling
        samples = self._rng.standard_normal((self.n_monte_carlo, 3))
        offsets = samples @ L.T + nominal_offset

        distances = np.linalg.norm(offsets, axis=1)
        n_hits = np.sum(distances < self.combined_radius)

        probability = float(n_hits / self.n_monte_carlo)

        # Apply analytic lower bound for very low probability events
        # Mahalanobis distance-based Gaussian approximation
        cov_inv = np.linalg.inv(combined_covariance + np.eye(3) * 1e-6)
        mahal2 = nominal_offset @ cov_inv @ nominal_offset
        analytic_approx = float(np.exp(-0.5 * mahal2))

        # Use the more conservative (higher) estimate
        return max(probability, analytic_approx * 0.01)

    # -- public API ----------------------------------------------------------

    def assess(
        self,
        vehicle_state: Dict[str, np.ndarray],
        tracked_objects: List[TrackedObject],
    ) -> List[CollisionPrediction]:
        """Assess collision risk for all tracked objects.

        Args:
            vehicle_state:   Dict with "position" (3,) and "velocity" (3,).
            tracked_objects: List of currently tracked debris objects.

        Returns:
            List of CollisionPrediction, one per tracked object, sorted
            by collision probability (highest first).
        """
        vehicle_pos = np.asarray(vehicle_state["position"], dtype=np.float64)
        vehicle_vel = np.asarray(vehicle_state["velocity"], dtype=np.float64)

        # Vehicle position covariance (assumed)
        vehicle_cov = np.eye(3) * 25.0  # 5 m 1-sigma per axis

        predictions: List[CollisionPrediction] = []

        for obj in tracked_objects:
            t_ca, miss_dist, rel_speed = self._find_closest_approach(
                vehicle_pos, vehicle_vel,
                obj.position, obj.velocity,
            )

            # Combined covariance at closest approach
            combined_cov = vehicle_cov + obj.covariance

            col_prob = self._estimate_collision_probability(miss_dist, combined_cov)

            needs_maneuver = col_prob > self.probability_threshold

            predictions.append(CollisionPrediction(
                obj_id=obj.obj_id,
                time_to_closest_approach=t_ca,
                miss_distance=miss_dist,
                collision_probability=col_prob,
                relative_velocity=rel_speed,
                needs_maneuver=needs_maneuver,
            ))

        # Sort by probability (highest first)
        predictions.sort(key=lambda p: p.collision_probability, reverse=True)

        return predictions


# ---------------------------------------------------------------------------
# Collision Avoidance Manoeuvre Planner
# ---------------------------------------------------------------------------

class CollisionAvoidanceManeuver:
    """Plan delta-v manoeuvres to avoid predicted collisions.

    Computes the optimal avoidance burn perpendicular to the relative
    velocity vector (B-plane targeting), checks fuel constraints, and
    generates a time-tagged burn plan.

    Args:
        min_safe_distance:  Target post-manoeuvre miss distance (m).
        max_delta_v:        Maximum allowable delta-v per manoeuvre (m/s).
        isp:                Engine specific impulse (s).
        vehicle_dry_mass:   Dry mass of vehicle (kg).
        min_lead_time:      Minimum time before TCA to execute burn (s).
    """

    def __init__(
        self,
        min_safe_distance: float = 1000.0,
        max_delta_v: float = 20.0,
        isp: float = 311.0,
        vehicle_dry_mass: float = 22_200.0,
        min_lead_time: float = 60.0,
    ) -> None:
        self.min_safe_distance = min_safe_distance
        self.max_delta_v = max_delta_v
        self.isp = isp
        self.vehicle_dry_mass = vehicle_dry_mass
        self.min_lead_time = min_lead_time

    def _compute_avoidance_dv(
        self,
        vehicle_pos: np.ndarray,
        vehicle_vel: np.ndarray,
        debris_pos: np.ndarray,
        debris_vel: np.ndarray,
        t_ca: float,
    ) -> Tuple[np.ndarray, float]:
        """Compute the avoidance delta-v vector via B-plane targeting.

        The manoeuvre is applied perpendicular to both the vehicle velocity
        and the relative position vector at TCA, which maximises the change
        in miss distance per unit delta-v.

        Args:
            vehicle_pos: Current vehicle position (m).
            vehicle_vel: Current vehicle velocity (m/s).
            debris_pos:  Current debris position (m).
            debris_vel:  Current debris velocity (m/s).
            t_ca:        Time to closest approach (s).

        Returns:
            Tuple (delta_v_vector, required_magnitude).
        """
        # Propagate to TCA
        v_pos_ca, v_vel_ca = _propagate_kepler(vehicle_pos, vehicle_vel, t_ca)
        d_pos_ca, d_vel_ca = _propagate_kepler(debris_pos, debris_vel, t_ca)

        rel_pos = d_pos_ca - v_pos_ca
        rel_vel = d_vel_ca - v_vel_ca
        miss_dist = np.linalg.norm(rel_pos)

        # Need to increase miss distance to min_safe_distance
        if miss_dist >= self.min_safe_distance:
            return np.zeros(3), 0.0

        deficit = self.min_safe_distance - miss_dist

        # B-plane normal: perpendicular to vehicle velocity and relative position
        v_hat = vehicle_vel / max(np.linalg.norm(vehicle_vel), 1e-12)
        r_hat = rel_pos / max(np.linalg.norm(rel_pos), 1e-12)

        # Manoeuvre direction: perpendicular to approach geometry
        maneuver_dir = np.cross(v_hat, r_hat)
        maneuver_norm = np.linalg.norm(maneuver_dir)

        if maneuver_norm < 1e-12:
            # Colinear -- use perpendicular to velocity
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v_hat, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            maneuver_dir = np.cross(v_hat, perp)
            maneuver_norm = np.linalg.norm(maneuver_dir)

        maneuver_dir = maneuver_dir / maneuver_norm

        # Delta-v magnitude: roughly deficit / t_ca (impulse approximation)
        # with safety factor
        t_eff = max(t_ca, 1.0)
        dv_magnitude = float(deficit / t_eff * 1.5)

        return maneuver_dir * dv_magnitude, dv_magnitude

    def _check_fuel(
        self,
        dv_magnitude: float,
        fuel_remaining: float,
    ) -> Tuple[bool, float]:
        """Check if sufficient fuel exists for the planned manoeuvre.

        Uses the Tsiolkovsky rocket equation to determine required
        propellant mass.

        Args:
            dv_magnitude:   Required delta-v (m/s).
            fuel_remaining: Available propellant mass (kg).

        Returns:
            Tuple (is_feasible, fuel_required_kg).
        """
        g0 = 9.81  # m/s^2
        ve = self.isp * g0
        total_mass = self.vehicle_dry_mass + fuel_remaining

        # Tsiolkovsky: dv = ve * ln(m0 / mf)
        # -> mf = m0 * exp(-dv / ve)
        # -> fuel_used = m0 - mf = m0 * (1 - exp(-dv / ve))
        mass_ratio = np.exp(-dv_magnitude / ve)
        fuel_required = float(total_mass * (1.0 - mass_ratio))

        is_feasible = fuel_required <= fuel_remaining
        return is_feasible, fuel_required

    def _estimate_burn_duration(
        self,
        dv_magnitude: float,
        fuel_remaining: float,
        thrust: float = 845_000.0,
    ) -> float:
        """Estimate the burn duration for a given delta-v.

        Args:
            dv_magnitude:   Delta-v magnitude (m/s).
            fuel_remaining: Available propellant (kg).
            thrust:         Engine thrust (N).

        Returns:
            Estimated burn duration (s).
        """
        total_mass = self.vehicle_dry_mass + fuel_remaining
        # F = m * a  ->  a = F / m  ->  dt = dv / a
        if thrust < 1.0:
            return 0.0
        accel = thrust / total_mass
        return float(dv_magnitude / accel)

    def _verify_post_maneuver(
        self,
        vehicle_pos: np.ndarray,
        vehicle_vel: np.ndarray,
        delta_v: np.ndarray,
        debris_pos: np.ndarray,
        debris_vel: np.ndarray,
        t_ca: float,
    ) -> float:
        """Predict post-manoeuvre miss distance at TCA.

        Args:
            vehicle_pos: Vehicle position (m).
            vehicle_vel: Vehicle velocity before manoeuvre (m/s).
            delta_v:     Applied delta-v vector (m/s).
            debris_pos:  Debris position (m).
            debris_vel:  Debris velocity (m/s).
            t_ca:        Time to closest approach (s).

        Returns:
            Post-manoeuvre miss distance (m).
        """
        new_vel = vehicle_vel + delta_v
        v_pos_ca, _ = _propagate_kepler(vehicle_pos, new_vel, t_ca)
        d_pos_ca, _ = _propagate_kepler(debris_pos, debris_vel, t_ca)
        return float(np.linalg.norm(d_pos_ca - v_pos_ca))

    # -- public API ----------------------------------------------------------

    def plan_maneuver(
        self,
        vehicle_state: Dict[str, np.ndarray],
        threat: CollisionPrediction,
        debris: TrackedObject,
        fuel_remaining: float,
        current_time: float = 0.0,
    ) -> Optional[BurnPlan]:
        """Generate a collision avoidance burn plan for a specific threat.

        Returns None if:
        - The miss distance already exceeds the safety threshold.
        - Insufficient fuel for the required delta-v.
        - Insufficient lead time before closest approach.
        - The delta-v exceeds the maximum allowable.

        Args:
            vehicle_state:  Dict with "position" (3,) and "velocity" (3,).
            threat:         CollisionPrediction for the object to avoid.
            debris:         TrackedObject state of the threatening debris.
            fuel_remaining: Available propellant mass (kg).
            current_time:   Current simulation time (s).

        Returns:
            BurnPlan or None if manoeuvre is infeasible or unnecessary.
        """
        vehicle_pos = np.asarray(vehicle_state["position"], dtype=np.float64)
        vehicle_vel = np.asarray(vehicle_state["velocity"], dtype=np.float64)

        # Check lead time
        if threat.time_to_closest_approach < self.min_lead_time:
            return None

        # Compute avoidance delta-v
        delta_v, dv_mag = self._compute_avoidance_dv(
            vehicle_pos, vehicle_vel,
            debris.position, debris.velocity,
            threat.time_to_closest_approach,
        )

        if dv_mag < 1e-6:
            # No manoeuvre needed -- already safe
            return None

        # Clamp to maximum allowable
        if dv_mag > self.max_delta_v:
            delta_v = delta_v / dv_mag * self.max_delta_v
            dv_mag = self.max_delta_v

        # Check fuel
        is_feasible, fuel_required = self._check_fuel(dv_mag, fuel_remaining)
        if not is_feasible:
            return None

        # Burn timing and duration
        burn_duration = self._estimate_burn_duration(dv_mag, fuel_remaining)
        burn_start = current_time + max(
            threat.time_to_closest_approach - burn_duration - self.min_lead_time,
            0.0,
        )

        # Verify post-manoeuvre miss distance
        post_miss = self._verify_post_maneuver(
            vehicle_pos, vehicle_vel, delta_v,
            debris.position, debris.velocity,
            threat.time_to_closest_approach,
        )

        # Burn direction unit vector
        burn_dir = delta_v / max(np.linalg.norm(delta_v), 1e-12)

        return BurnPlan(
            delta_v=delta_v,
            delta_v_magnitude=dv_mag,
            burn_direction=burn_dir,
            burn_start_time=burn_start,
            burn_duration=burn_duration,
            fuel_required=fuel_required,
            target_obj_id=threat.obj_id,
            post_miss_distance=post_miss,
        )
