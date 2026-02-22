"""
Physics environment models for the rocket simulation.

Implements four loosely-coupled models that collectively define the
external forces acting on the vehicle:

* **Atmosphere** -- exponential density/pressure/temperature profiles.
* **AerodynamicModel** -- drag (and basic lift) with Mach-dependent Cd.
* **GravityModel** -- inverse-square law with optional J2 perturbation.
* **WindModel** -- altitude-dependent mean wind with gust/turbulence.

All models are intentionally lightweight (numpy only, no external
solvers) to support real-time Monte-Carlo use inside the simulation
engine.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from rocket_ai_os.config import SimConfig


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_R_EARTH = 6_371_000.0             # Mean Earth radius (m)
_MU_EARTH = 3.986004418e14         # Gravitational parameter (m^3/s^2)
_J2 = 1.08263e-3                   # J2 zonal harmonic coefficient
_R_GAS = 287.058                   # Specific gas constant for dry air (J/(kg*K))
_GAMMA_AIR = 1.4                   # Ratio of specific heats for air
_SEA_LEVEL_PRESSURE = 101_325.0    # Pa
_SEA_LEVEL_TEMPERATURE = 288.15    # K
_SEA_LEVEL_DENSITY = 1.225         # kg/m^3
_SCALE_HEIGHT = 8_500.0            # m (tropospheric approximation)
_G0 = 9.80665                      # Standard gravitational acceleration (m/s^2)


# ---------------------------------------------------------------------------
# Atmosphere model
# ---------------------------------------------------------------------------

class Atmosphere:
    """Exponential atmosphere model for altitudes up to ~85 km.

    Uses an exponential pressure/density profile with a linear
    temperature lapse rate below 11 km and isothermal above.

    Args:
        scale_height:      Pressure scale height in metres.
        sea_level_density: Air density at sea level in kg/m^3.
        sea_level_temp:    Temperature at sea level in Kelvin.
        sea_level_pressure: Pressure at sea level in Pascals.
    """

    # Temperature lapse rate in the troposphere (K/m)
    _LAPSE_RATE: float = 0.0065  # valid 0 -- 11 km
    _TROPOPAUSE_ALT: float = 11_000.0  # m
    _TROPOPAUSE_TEMP: float = _SEA_LEVEL_TEMPERATURE - 0.0065 * 11_000.0  # ~216.65 K

    def __init__(
        self,
        scale_height: float = _SCALE_HEIGHT,
        sea_level_density: float = _SEA_LEVEL_DENSITY,
        sea_level_temp: float = _SEA_LEVEL_TEMPERATURE,
        sea_level_pressure: float = _SEA_LEVEL_PRESSURE,
    ) -> None:
        self.scale_height = scale_height
        self.rho0 = sea_level_density
        self.T0 = sea_level_temp
        self.P0 = sea_level_pressure

    def get_density(self, altitude: float) -> float:
        """Return atmospheric density at the given altitude.

        Uses an exponential decay model:  rho = rho0 * exp(-h / H).

        Args:
            altitude: Geometric altitude in metres above sea level.

        Returns:
            Air density in kg/m^3.
        """
        altitude = max(altitude, 0.0)
        return float(self.rho0 * np.exp(-altitude / self.scale_height))

    def get_pressure(self, altitude: float) -> float:
        """Return atmospheric pressure at the given altitude.

        Below the tropopause the barometric formula with a constant lapse
        rate is used.  Above, an isothermal exponential decay applies.

        Args:
            altitude: Geometric altitude in metres.

        Returns:
            Air pressure in Pascals.
        """
        altitude = max(altitude, 0.0)

        if altitude <= self._TROPOPAUSE_ALT:
            # Tropospheric barometric formula
            T = self.T0 - self._LAPSE_RATE * altitude
            exponent = _G0 / (_R_GAS * self._LAPSE_RATE)
            return float(self.P0 * (T / self.T0) ** exponent)
        else:
            # Pressure at tropopause
            T_tp = self._TROPOPAUSE_TEMP
            exponent = _G0 / (_R_GAS * self._LAPSE_RATE)
            P_tp = self.P0 * (T_tp / self.T0) ** exponent
            # Isothermal region above tropopause
            dh = altitude - self._TROPOPAUSE_ALT
            return float(P_tp * np.exp(-_G0 * dh / (_R_GAS * T_tp)))

    def get_temperature(self, altitude: float) -> float:
        """Return atmospheric temperature at the given altitude.

        Linear lapse below the tropopause, constant above.

        Args:
            altitude: Geometric altitude in metres.

        Returns:
            Temperature in Kelvin.
        """
        altitude = max(altitude, 0.0)

        if altitude <= self._TROPOPAUSE_ALT:
            return float(self.T0 - self._LAPSE_RATE * altitude)
        else:
            return float(self._TROPOPAUSE_TEMP)

    def get_speed_of_sound(self, altitude: float) -> float:
        """Return the local speed of sound at the given altitude.

        Computed from the ideal-gas relation: a = sqrt(gamma * R * T).

        Args:
            altitude: Geometric altitude in metres.

        Returns:
            Speed of sound in m/s.
        """
        T = self.get_temperature(altitude)
        return float(np.sqrt(_GAMMA_AIR * _R_GAS * T))


# ---------------------------------------------------------------------------
# Aerodynamic model
# ---------------------------------------------------------------------------

class AerodynamicModel:
    """Aerodynamic force and torque model with Mach-dependent drag.

    Computes drag (and optionally a simplified normal / side force) on
    the vehicle using the standard drag equation with a Mach-dependent
    drag coefficient.

    The Mach-Cd profile captures the transonic drag rise:
        - Subsonic   (M < 0.8):  Cd = Cd0
        - Transonic  (0.8 <= M <= 1.2):  Cd rises to Cd0 * drag_rise_factor
        - Supersonic (M > 1.2):  Cd decays back toward Cd0

    Args:
        Cd0:               Zero-lift drag coefficient at subsonic speeds.
        reference_area:    Aerodynamic reference area in m^2.
        vehicle_length:    Vehicle length in m (for torque moment arm).
        drag_rise_factor:  Peak Cd / Cd0 ratio in the transonic regime.
        cp_offset_fraction: Centre-of-pressure offset from nose as a
                           fraction of vehicle length (used for torque).
    """

    def __init__(
        self,
        Cd0: float = 0.3,
        reference_area: float = 10.52,
        vehicle_length: float = 47.7,
        drag_rise_factor: float = 1.6,
        cp_offset_fraction: float = 0.6,
    ) -> None:
        self.Cd0 = Cd0
        self.ref_area = reference_area
        self.vehicle_length = vehicle_length
        self.drag_rise_factor = drag_rise_factor
        self.cp_offset = cp_offset_fraction * vehicle_length

    def _get_cd(self, mach: float) -> float:
        """Return the drag coefficient for the given Mach number.

        Implements a piece-wise Mach-Cd curve:
            - M < 0.8:  Cd = Cd0
            - 0.8 <= M <= 1.2:  sinusoidal rise to Cd0 * drag_rise_factor
            - M > 1.2:  gradual decay back toward Cd0

        Args:
            mach: Freestream Mach number (non-negative).

        Returns:
            Drag coefficient (dimensionless).
        """
        Cd0 = self.Cd0
        if mach < 0.8:
            return Cd0
        elif mach <= 1.2:
            # Smooth transonic rise using a half-sine
            t = (mach - 0.8) / 0.4  # 0..1 through the transonic band
            return Cd0 * (1.0 + (self.drag_rise_factor - 1.0) * np.sin(t * np.pi / 2.0))
        else:
            # Supersonic: decay as 1/sqrt(M^2 - 1)  (Prandtl-Glauert analogy)
            Cd_peak = Cd0 * self.drag_rise_factor
            decay = 1.0 / np.sqrt(mach * mach - 1.0 + 0.01)
            return float(max(Cd0, Cd_peak * decay))

    @staticmethod
    def _angle_of_attack(velocity: np.ndarray, attitude_dcm: np.ndarray) -> float:
        """Compute the aerodynamic angle of attack in radians.

        The angle of attack is the angle between the freestream velocity
        vector and the vehicle body Z-axis (longitudinal axis).

        Args:
            velocity:      Inertial-frame velocity vector (m/s).
            attitude_dcm:  3x3 body-to-inertial DCM.

        Returns:
            Angle of attack in radians [0, pi].
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-3:
            return 0.0
        body_z_inertial = attitude_dcm @ np.array([0.0, 0.0, 1.0])
        cos_aoa = np.clip(np.dot(velocity, body_z_inertial) / speed, -1.0, 1.0)
        return float(np.arccos(cos_aoa))

    def compute_aero_forces(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude_dcm: np.ndarray,
        atmosphere: Atmosphere,
        wind_velocity: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute aerodynamic force and torque in the inertial frame.

        The drag force opposes the velocity-relative-to-air vector.  A
        simplified normal-force component is added when the angle of
        attack is nonzero, producing a restoring (weathercock) torque.

        Args:
            position:      Vehicle position [x, y, z] in metres (ENU).
            velocity:      Vehicle inertial velocity [vx, vy, vz] in m/s.
            attitude_dcm:  3x3 body-to-inertial DCM.
            atmosphere:    Atmosphere model instance.
            wind_velocity: Optional wind velocity in the inertial frame (m/s).

        Returns:
            Tuple of (force, torque) both as 3-element numpy arrays in
            the inertial frame (N, N*m).
        """
        altitude = max(float(position[2]), 0.0)

        # Airspeed = velocity - wind
        if wind_velocity is not None:
            v_air = velocity - wind_velocity
        else:
            v_air = velocity.copy()

        airspeed = np.linalg.norm(v_air)

        if airspeed < 1e-3:
            return np.zeros(3), np.zeros(3)

        rho = atmosphere.get_density(altitude)
        a_sound = atmosphere.get_speed_of_sound(altitude)
        mach = airspeed / a_sound if a_sound > 1.0 else 0.0

        Cd = self._get_cd(mach)
        q = 0.5 * rho * airspeed * airspeed  # dynamic pressure

        # --- Drag: opposes airspeed direction ---
        v_hat = v_air / airspeed
        drag_magnitude = Cd * q * self.ref_area
        drag_force = -drag_magnitude * v_hat

        # --- Angle of attack and normal force ---
        aoa = self._angle_of_attack(v_air, attitude_dcm)

        # Simplified normal-force coefficient: Cn ~ Cn_alpha * sin(aoa)
        # Cn_alpha ~ 2 for a slender body
        Cn_alpha = 2.0
        normal_coeff = Cn_alpha * np.sin(aoa)
        normal_mag = normal_coeff * q * self.ref_area

        # Normal force direction: perpendicular to airspeed in the plane
        # defined by the body axis and the airspeed vector.
        body_z = attitude_dcm @ np.array([0.0, 0.0, 1.0])
        n_perp = body_z - np.dot(body_z, v_hat) * v_hat
        n_perp_norm = np.linalg.norm(n_perp)
        if n_perp_norm > 1e-6:
            normal_force = normal_mag * (n_perp / n_perp_norm)
        else:
            normal_force = np.zeros(3)

        total_force = drag_force + normal_force

        # --- Aerodynamic torque ---
        # Centre-of-pressure is offset from centre-of-mass along the body
        # Z-axis.  Torque = r_cp x F_aero (both in inertial frame).
        cp_body = np.array([0.0, 0.0, self.cp_offset])
        r_cp_inertial = attitude_dcm @ cp_body
        torque = np.cross(r_cp_inertial, total_force)

        return total_force, torque


# ---------------------------------------------------------------------------
# Gravity model
# ---------------------------------------------------------------------------

class GravityModel:
    """Gravitational acceleration model with optional J2 oblateness.

    For altitudes typical of powered-descent (< 100 km) a simple
    constant downward gravity is usually sufficient.  For longer-duration
    coast phases the full inverse-square law with J2 provides better
    accuracy.

    Args:
        use_j2:       Enable J2 perturbation term.
        flat_gravity:  If provided, use a constant gravity vector instead
                      of the inverse-square model.  Useful for short
                      landing-burn scenarios to reduce computation.
    """

    def __init__(
        self,
        use_j2: bool = False,
        flat_gravity: Optional[np.ndarray] = None,
    ) -> None:
        self.use_j2 = use_j2
        self.flat_gravity = (
            np.asarray(flat_gravity, dtype=np.float64)
            if flat_gravity is not None
            else None
        )

    def compute_gravity(
        self,
        position: np.ndarray,
        mass: float,
    ) -> np.ndarray:
        """Compute gravitational force on the vehicle.

        When ``flat_gravity`` is configured the position argument is
        ignored and a constant body force is returned.  Otherwise the
        inverse-square law is evaluated with the vehicle at an ENU
        offset from a point on Earth's surface.

        Args:
            position: Vehicle position in ENU frame [x, y, z] (m).
            mass:     Vehicle mass in kg.

        Returns:
            Gravitational force vector in the inertial (ENU) frame (N).
        """
        if self.flat_gravity is not None:
            return self.flat_gravity * mass

        # Position relative to Earth centre (approximate: launch site at
        # equator, ENU z = altitude).
        altitude = float(position[2])
        r = _R_EARTH + altitude

        if r < 1.0:
            r = 1.0  # safety clamp

        # Basic inverse-square (radial, pointing downward in ENU)
        g_mag = _MU_EARTH / (r * r)
        g_vector = np.array([0.0, 0.0, -g_mag])

        if self.use_j2:
            # J2 perturbation in a simplified ENU approximation.
            # The dominant J2 effect at low inclination is a slight
            # increase in the radial gravity component.
            sin_lat = 0.0  # equatorial launch assumption
            cos_lat = 1.0
            r_ratio = (_R_EARTH / r) ** 2
            j2_radial = (3.0 / 2.0) * _J2 * r_ratio * (3.0 * sin_lat ** 2 - 1.0)
            g_vector[2] *= (1.0 + j2_radial)

        return g_vector * mass


# ---------------------------------------------------------------------------
# Wind model
# ---------------------------------------------------------------------------

class WindModel:
    """Altitude- and time-dependent wind model with gusts and turbulence.

    The mean wind profile increases logarithmically with altitude up to
    the jet-stream layer.  Discrete gusts and continuous turbulence
    (simplified Dryden model) are superimposed.

    Args:
        base_wind:  Mean surface wind vector [east, north, up] in m/s.
        gust_intensity: Peak gust magnitude in m/s.
        turbulence_intensity: Dryden turbulence sigma in m/s.
        seed: RNG seed for reproducible turbulence.
    """

    # Altitude at which the log profile transitions to constant (m)
    _BOUNDARY_LAYER_TOP: float = 2_000.0
    # Characteristic turbulence time constant (s)
    _TAU_TURB: float = 5.0

    def __init__(
        self,
        base_wind: Optional[np.ndarray] = None,
        gust_intensity: float = 10.0,
        turbulence_intensity: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.base_wind = (
            np.asarray(base_wind, dtype=np.float64)
            if base_wind is not None
            else np.array([3.0, 1.0, 0.0])  # mild easterly + slight northerly
        )
        self.gust_intensity = gust_intensity
        self.turb_sigma = turbulence_intensity
        self._rng = np.random.default_rng(seed)

        # Dryden turbulence state (first-order Markov per axis)
        self._turb_state = np.zeros(3)

        # Pre-generate discrete gust events (altitude, direction, magnitude)
        self._gust_events = self._generate_gust_schedule()

    def _generate_gust_schedule(self) -> list:
        """Create a set of random discrete gusts.

        Each gust event is a tuple (altitude_centre, thickness,
        direction_unit, magnitude) that produces a Gaussian-shaped
        velocity pulse when the vehicle passes through the altitude band.
        """
        n_gusts = self._rng.integers(3, 8)
        events = []
        for _ in range(n_gusts):
            alt_centre = self._rng.uniform(200.0, 10_000.0)
            thickness = self._rng.uniform(100.0, 500.0)
            direction = self._rng.normal(size=3)
            direction[2] *= 0.3  # vertical gusts are weaker
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            magnitude = self._rng.uniform(0.3, 1.0) * self.gust_intensity
            events.append((alt_centre, thickness, direction, magnitude))
        return events

    def _mean_wind(self, altitude: float) -> np.ndarray:
        """Compute the mean wind vector at a given altitude.

        Uses a logarithmic boundary-layer profile below 2 km and a
        roughly constant wind aloft above that.

        Args:
            altitude: Geometric altitude in metres.

        Returns:
            Mean wind vector [east, north, up] in m/s.
        """
        altitude = max(altitude, 1.0)  # avoid log(0)

        if altitude <= self._BOUNDARY_LAYER_TOP:
            # Logarithmic profile: V(h) = V_ref * ln(h/z0) / ln(h_ref/z0)
            z0 = 0.01       # surface roughness (m)
            h_ref = 10.0    # reference height (m)
            factor = np.log(altitude / z0) / np.log(h_ref / z0)
            factor = max(factor, 0.0)
            return self.base_wind * factor
        else:
            # Above boundary layer: roughly 2-3x surface wind
            factor_top = np.log(self._BOUNDARY_LAYER_TOP / 0.01) / np.log(10.0 / 0.01)
            return self.base_wind * factor_top

    def _gust_contribution(self, altitude: float) -> np.ndarray:
        """Sum the contributions of all discrete gust events.

        Each gust has a Gaussian envelope centred at a specific altitude.

        Args:
            altitude: Vehicle altitude in metres.

        Returns:
            Gust velocity vector in m/s.
        """
        gust = np.zeros(3)
        for alt_c, thickness, direction, mag in self._gust_events:
            envelope = np.exp(-0.5 * ((altitude - alt_c) / thickness) ** 2)
            gust += mag * direction * envelope
        return gust

    def _update_turbulence(self, dt: float) -> np.ndarray:
        """Advance the Dryden turbulence state by one time step.

        Implements a first-order Gauss-Markov process per axis:
            x_new = x * exp(-dt/tau) + sigma * sqrt(1 - exp(-2*dt/tau)) * noise

        Args:
            dt: Time step in seconds.

        Returns:
            Turbulence velocity perturbation in m/s.
        """
        tau = self._TAU_TURB
        exp_dt = np.exp(-dt / tau)
        noise_scale = self.turb_sigma * np.sqrt(1.0 - exp_dt * exp_dt)

        self._turb_state = (
            self._turb_state * exp_dt
            + noise_scale * self._rng.standard_normal(3)
        )
        # Vertical turbulence is typically weaker
        self._turb_state[2] *= 0.5
        return self._turb_state.copy()

    def get_wind(self, altitude: float, time: float, dt: float = 0.01) -> np.ndarray:
        """Return the total wind velocity at a given altitude and time.

        Combines the mean wind profile, discrete gusts, and continuous
        Dryden turbulence.

        Args:
            altitude: Vehicle altitude in metres.
            time:     Current simulation time in seconds (used for
                      turbulence seeding consistency).
            dt:       Time step for turbulence state update (s).

        Returns:
            Wind velocity vector [east, north, up] in m/s, expressed in
            the inertial ENU frame.
        """
        mean = self._mean_wind(altitude)
        gust = self._gust_contribution(altitude)
        turb = self._update_turbulence(dt)
        return mean + gust + turb
