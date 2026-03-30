"""
Space Weather Prediction and Radiation Shielding subsystem.

Monitors simulated solar particle flux, predicts Solar Particle Events (SPE)
using LSTM-inspired rolling-statistics trend analysis, and manages radiation
shielding through vehicle attitude reorientation and electronics power-down
sequences.

References:
    - Schwadron et al., "Does the Worsening Galactic Cosmic Ray Environment
      Observe a Global Trend?", Space Weather, 2018.
    - Cucinotta et al., "Space Radiation Risk Limits and Earth-Moon-Mars
      Environmental Models", Space Weather, 2006.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from rocket_ai_os.config import SimConfig


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class SpaceWeatherCondition(Enum):
    """Overall space-weather severity classification."""
    QUIET = auto()          # Background flux -- nominal operations
    MODERATE = auto()       # Elevated flux -- increased monitoring
    STORM = auto()          # Geomagnetic / proton storm -- restrict EVA
    SEVERE_STORM = auto()   # Extreme event -- safe mode mandatory


class ShieldingMode(Enum):
    """Radiation shield manager operating modes."""
    NOMINAL = auto()     # Normal flight -- no special orientation
    ALERT = auto()       # Elevated flux detected -- prepare shielding
    SAFE_MODE = auto()   # Active shielding -- electronics powered down
    RECOVERY = auto()    # Post-event -- gradually restoring systems


@dataclass
class SolarParticleEvent:
    """Descriptor for a predicted or detected Solar Particle Event.

    Attributes:
        intensity:      Peak proton flux estimate (particles/cm^2/s/sr).
        onset_time:     Estimated event onset (simulation seconds).
        duration:       Predicted event duration (s).
        predicted_peak: Time of predicted flux maximum (s).
        condition:      Severity classification.
        confidence:     Prediction confidence in [0, 1].
    """
    intensity: float
    onset_time: float
    duration: float
    predicted_peak: float
    condition: SpaceWeatherCondition = SpaceWeatherCondition.QUIET
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Space Weather Monitor (LSTM-inspired predictor)
# ---------------------------------------------------------------------------

class SpaceWeatherMonitor:
    """Monitor solar particle flux and predict SPE events.

    Uses an LSTM-inspired architecture simplified to pure numpy:
    - Rolling mean / variance (hidden state analogue).
    - Exponential-weighted trend (cell state analogue).
    - Threshold gates that trigger alerts based on flux rate-of-change.

    The ``forget gate`` is an exponential decay on the trend accumulator,
    and the ``input gate`` scales new flux observations by their deviation
    from the rolling mean.

    Args:
        window_size:   Rolling statistics window (number of samples).
        ema_alpha:     Exponential moving average smoothing factor (0, 1).
        quiet_threshold:    Flux below this -> QUIET (particles/cm^2/s/sr).
        moderate_threshold: Flux above this -> MODERATE.
        storm_threshold:    Flux above this -> STORM.
        severe_threshold:   Flux above this -> SEVERE_STORM.
        trend_sensitivity:  Multiplier for rate-of-change detection.
        seed:               RNG seed.
    """

    def __init__(
        self,
        window_size: int = 64,
        ema_alpha: float = 0.1,
        quiet_threshold: float = 10.0,
        moderate_threshold: float = 100.0,
        storm_threshold: float = 1_000.0,
        severe_threshold: float = 10_000.0,
        trend_sensitivity: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.ema_alpha = ema_alpha

        self.quiet_threshold = quiet_threshold
        self.moderate_threshold = moderate_threshold
        self.storm_threshold = storm_threshold
        self.severe_threshold = severe_threshold
        self.trend_sensitivity = trend_sensitivity

        self._rng = np.random.default_rng(seed)

        # Rolling buffer (flux history)
        self._flux_buffer: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self._buffer_idx: int = 0
        self._buffer_count: int = 0

        # LSTM-inspired internal states
        self._cell_state: float = 0.0       # trend accumulator (cell state)
        self._hidden_state: float = 0.0     # smoothed flux (hidden state)
        self._ema_flux: float = 0.0         # exponential moving average
        self._ema_variance: float = 0.0     # variance tracker

        # Event state
        self._current_condition = SpaceWeatherCondition.QUIET
        self._last_time: float = 0.0

    # -- internal LSTM-like gates --------------------------------------------

    def _forget_gate(self, dt: float) -> float:
        """Exponential decay on trend state -- forgets old trends.

        Args:
            dt: Time since last update (s).

        Returns:
            Forget factor in (0, 1].
        """
        tau = 60.0  # trend memory time constant (s)
        return float(np.exp(-dt / tau))

    def _input_gate(self, flux: float) -> float:
        """Scale new observation by its deviation from the mean.

        Large deviations (potential SPE onset) receive higher weight.

        Args:
            flux: New flux measurement.

        Returns:
            Gate value in [0, 1].
        """
        if self._buffer_count < 2:
            return 0.5
        mean = np.mean(self._flux_buffer[:self._buffer_count])
        std = max(np.std(self._flux_buffer[:self._buffer_count]), 1e-6)
        z_score = abs(flux - mean) / std
        return float(np.clip(z_score / 4.0, 0.0, 1.0))

    def _output_gate(self) -> float:
        """Confidence-like gate based on buffer fullness and variance stability.

        Returns:
            Gate value in [0, 1].
        """
        fullness = min(self._buffer_count / self.window_size, 1.0)
        # Variance stability: low variance relative to mean -> high confidence
        if self._ema_flux > 1e-6:
            cv = np.sqrt(max(self._ema_variance, 0.0)) / self._ema_flux
            stability = float(np.exp(-cv))
        else:
            stability = 0.5
        return float(0.5 * fullness + 0.5 * stability)

    # -- update --------------------------------------------------------------

    def update(self, flux: float, time: float) -> SpaceWeatherCondition:
        """Ingest a new flux measurement and classify current conditions.

        Args:
            flux: Measured proton flux (particles/cm^2/s/sr).
            time: Current simulation time (s).

        Returns:
            Updated SpaceWeatherCondition.
        """
        dt = max(time - self._last_time, 1e-3)
        self._last_time = time

        # --- Push into rolling buffer ---
        self._flux_buffer[self._buffer_idx] = flux
        self._buffer_idx = (self._buffer_idx + 1) % self.window_size
        self._buffer_count = min(self._buffer_count + 1, self.window_size)

        # --- EMA update ---
        alpha = self.ema_alpha
        self._ema_flux = alpha * flux + (1 - alpha) * self._ema_flux
        diff2 = (flux - self._ema_flux) ** 2
        self._ema_variance = alpha * diff2 + (1 - alpha) * self._ema_variance

        # --- LSTM-inspired state update ---
        fg = self._forget_gate(dt)
        ig = self._input_gate(flux)

        # Cell state: accumulate trend weighted by input gate
        trend_input = (flux - self._ema_flux)  # deviation from mean
        self._cell_state = fg * self._cell_state + ig * trend_input

        # Hidden state: smoothed flux modulated by cell state
        og = self._output_gate()
        self._hidden_state = og * np.tanh(self._cell_state) + self._ema_flux

        # --- Classification ---
        effective_flux = max(self._hidden_state, flux)

        if effective_flux >= self.severe_threshold:
            self._current_condition = SpaceWeatherCondition.SEVERE_STORM
        elif effective_flux >= self.storm_threshold:
            self._current_condition = SpaceWeatherCondition.STORM
        elif effective_flux >= self.moderate_threshold:
            self._current_condition = SpaceWeatherCondition.MODERATE
        else:
            self._current_condition = SpaceWeatherCondition.QUIET

        return self._current_condition

    # -- prediction ----------------------------------------------------------

    def predict_spe(
        self,
        flux_history: np.ndarray,
        current_time: float = 0.0,
    ) -> Optional[SolarParticleEvent]:
        """Predict whether a Solar Particle Event is imminent.

        Analyses the flux history using:
        1. Linear trend (least-squares slope).
        2. Acceleration (second derivative).
        3. Rolling maximum vs. rolling mean ratio (precursor signature).

        An SPE prediction is issued when the trend slope exceeds a
        sensitivity-adjusted threshold.

        Args:
            flux_history: 1-D array of recent flux measurements (newest last).
            current_time: Current simulation time (s).

        Returns:
            SolarParticleEvent if an event is predicted, else None.
        """
        n = len(flux_history)
        if n < 4:
            return None

        # --- Linear trend via least-squares ---
        t = np.arange(n, dtype=np.float64)
        mean_t = np.mean(t)
        mean_f = np.mean(flux_history)
        cov_tf = np.sum((t - mean_t) * (flux_history - mean_f))
        var_t = np.sum((t - mean_t) ** 2)
        slope = cov_tf / max(var_t, 1e-12)

        # --- Second derivative (acceleration) ---
        if n >= 3:
            d2 = np.diff(flux_history, n=2)
            acceleration = float(np.mean(d2[-min(5, len(d2)):]))
        else:
            acceleration = 0.0

        # --- Precursor ratio: recent max / rolling mean ---
        recent = flux_history[-min(8, n):]
        precursor_ratio = float(np.max(recent) / max(mean_f, 1e-6))

        # --- Decision logic ---
        threshold = self.moderate_threshold * 0.5 / self.trend_sensitivity

        is_rising = slope > threshold
        is_accelerating = acceleration > 0.0
        is_precursor = precursor_ratio > 2.0

        if not (is_rising and (is_accelerating or is_precursor)):
            return None

        # --- Estimate event parameters ---
        # Predicted peak intensity: extrapolate current trend
        time_to_peak = max(abs(mean_f / slope) if slope > 0 else 30.0, 5.0)
        peak_intensity = float(flux_history[-1] + slope * time_to_peak)

        # Duration estimate: empirical scaling
        duration = float(np.clip(time_to_peak * 3.0, 60.0, 7200.0))

        # Classify expected severity
        if peak_intensity >= self.severe_threshold:
            condition = SpaceWeatherCondition.SEVERE_STORM
        elif peak_intensity >= self.storm_threshold:
            condition = SpaceWeatherCondition.STORM
        elif peak_intensity >= self.moderate_threshold:
            condition = SpaceWeatherCondition.MODERATE
        else:
            condition = SpaceWeatherCondition.QUIET

        confidence = float(np.clip(
            0.3 * min(slope / max(threshold, 1e-6), 1.0)
            + 0.3 * min(precursor_ratio / 5.0, 1.0)
            + 0.2 * (1.0 if is_accelerating else 0.0)
            + 0.2 * min(n / self.window_size, 1.0),
            0.0,
            1.0,
        ))

        return SolarParticleEvent(
            intensity=peak_intensity,
            onset_time=current_time,
            duration=duration,
            predicted_peak=current_time + time_to_peak,
            condition=condition,
            confidence=confidence,
        )

    # -- shielding recommendation --------------------------------------------

    def recommend_shielding_orientation(
        self,
        particle_direction: np.ndarray,
    ) -> np.ndarray:
        """Compute an attitude quaternion that orients the vehicle to use
        the fuel tanks (bottom) as a radiation shadow shield.

        The desired orientation points the vehicle's -Z axis (engine bell /
        fuel tank direction) towards the incoming particle flux.

        Args:
            particle_direction: Unit vector of incoming particle flux in
                                the inertial frame (3,).

        Returns:
            Attitude quaternion [w, x, y, z] (scalar-first) that aligns
            the vehicle's -Z axis with the particle direction.
        """
        particle_dir = np.asarray(particle_direction, dtype=np.float64)
        norm = np.linalg.norm(particle_dir)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        particle_dir = particle_dir / norm

        # We want body -Z to face particle_direction, so body +Z faces away.
        # Target +Z in inertial frame:
        target_z = -particle_dir

        # Build rotation from [0, 0, 1] to target_z using axis-angle
        z_body = np.array([0.0, 0.0, 1.0])
        dot = float(np.dot(z_body, target_z))

        if dot > 0.9999:
            # Already aligned
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif dot < -0.9999:
            # 180 degree rotation about any perpendicular axis
            return np.array([0.0, 1.0, 0.0, 0.0])

        axis = np.cross(z_body, target_z)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        # Axis-angle to quaternion
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        quat = np.array([w, xyz[0], xyz[1], xyz[2]])

        return quat / np.linalg.norm(quat)

    @property
    def current_condition(self) -> SpaceWeatherCondition:
        """Return the latest classified space weather condition."""
        return self._current_condition


# ---------------------------------------------------------------------------
# Radiation Shield Manager
# ---------------------------------------------------------------------------

class RadiationShieldManager:
    """Manage vehicle radiation shielding through mode transitions and
    electronics power management.

    Mode transitions:
        NOMINAL -> ALERT -> SAFE_MODE -> RECOVERY -> NOMINAL

    In SAFE_MODE the manager:
    - Powers down sensitive electronics (specified in the sensitivity list).
    - Commands a shielding attitude via the SpaceWeatherMonitor.
    - Maintains minimum-power operations for GNC.

    Args:
        monitor:              SpaceWeatherMonitor instance for predictions.
        sensitive_systems:    List of system names to power down in SAFE_MODE.
        alert_dwell_time:     Minimum time in ALERT before escalating (s).
        recovery_dwell_time:  Minimum time in RECOVERY before returning to NOMINAL (s).
    """

    def __init__(
        self,
        monitor: SpaceWeatherMonitor,
        sensitive_systems: Optional[List[str]] = None,
        alert_dwell_time: float = 30.0,
        recovery_dwell_time: float = 120.0,
    ) -> None:
        self.monitor = monitor
        self.sensitive_systems: List[str] = sensitive_systems or [
            "star_tracker",
            "optical_sensor",
            "lidar",
            "high_gain_antenna",
            "payload_computer",
        ]

        self.alert_dwell_time = alert_dwell_time
        self.recovery_dwell_time = recovery_dwell_time

        # State
        self._mode: ShieldingMode = ShieldingMode.NOMINAL
        self._mode_entry_time: float = 0.0
        self._powered_down: List[str] = []
        self._shielding_quaternion: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._time: float = 0.0

    # -- properties ----------------------------------------------------------

    @property
    def mode(self) -> ShieldingMode:
        """Current shielding mode."""
        return self._mode

    @property
    def powered_down_systems(self) -> List[str]:
        """List of currently powered-down system names."""
        return list(self._powered_down)

    @property
    def shielding_quaternion(self) -> np.ndarray:
        """Target attitude quaternion for radiation shielding."""
        return self._shielding_quaternion.copy()

    # -- mode transition logic -----------------------------------------------

    def _transition_to(self, new_mode: ShieldingMode, time: float) -> None:
        """Execute mode transition with entry/exit actions.

        Args:
            new_mode: Target shielding mode.
            time:     Current simulation time (s).
        """
        old_mode = self._mode

        # Exit actions
        if old_mode == ShieldingMode.SAFE_MODE and new_mode != ShieldingMode.SAFE_MODE:
            # Power up electronics on leaving safe mode
            self._powered_down.clear()

        # Enter actions
        if new_mode == ShieldingMode.SAFE_MODE and old_mode != ShieldingMode.SAFE_MODE:
            # Power down sensitive electronics
            self._powered_down = list(self.sensitive_systems)

        self._mode = new_mode
        self._mode_entry_time = time

    # -- update --------------------------------------------------------------

    def update(
        self,
        flux: float,
        time: float,
        particle_direction: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Run one update cycle of the radiation shield manager.

        1. Feed flux to the SpaceWeatherMonitor.
        2. Evaluate mode transition conditions.
        3. Update shielding attitude recommendation if needed.

        Args:
            flux:                Measured proton flux (particles/cm^2/s/sr).
            time:                Current simulation time (s).
            particle_direction:  Unit vector of incoming flux (inertial frame).

        Returns:
            Dict with keys:
                "mode": current ShieldingMode,
                "condition": SpaceWeatherCondition,
                "powered_down": list of powered-down system names,
                "shielding_quat": recommended attitude quaternion (4,),
                "spe_prediction": SolarParticleEvent or None.
        """
        self._time = time
        dwell = time - self._mode_entry_time

        # Step 1: Update monitor
        condition = self.monitor.update(flux, time)

        # Step 2: SPE prediction from recent buffer
        count = self.monitor._buffer_count
        if count > 4:
            idx = self.monitor._buffer_idx
            if count >= self.monitor.window_size:
                # Reconstruct ordered history from circular buffer
                history = np.concatenate([
                    self.monitor._flux_buffer[idx:],
                    self.monitor._flux_buffer[:idx],
                ])
            else:
                history = self.monitor._flux_buffer[:count]
            spe = self.monitor.predict_spe(history, current_time=time)
        else:
            spe = None

        # Step 3: Mode state machine
        if self._mode == ShieldingMode.NOMINAL:
            if condition in (SpaceWeatherCondition.STORM,
                             SpaceWeatherCondition.SEVERE_STORM):
                self._transition_to(ShieldingMode.SAFE_MODE, time)
            elif condition == SpaceWeatherCondition.MODERATE:
                self._transition_to(ShieldingMode.ALERT, time)
            elif spe is not None and spe.condition in (
                SpaceWeatherCondition.STORM,
                SpaceWeatherCondition.SEVERE_STORM,
            ):
                self._transition_to(ShieldingMode.ALERT, time)

        elif self._mode == ShieldingMode.ALERT:
            if condition in (SpaceWeatherCondition.STORM,
                             SpaceWeatherCondition.SEVERE_STORM):
                self._transition_to(ShieldingMode.SAFE_MODE, time)
            elif condition == SpaceWeatherCondition.QUIET and dwell > self.alert_dwell_time:
                self._transition_to(ShieldingMode.NOMINAL, time)

        elif self._mode == ShieldingMode.SAFE_MODE:
            if condition in (SpaceWeatherCondition.QUIET,
                             SpaceWeatherCondition.MODERATE):
                if dwell > self.alert_dwell_time:
                    self._transition_to(ShieldingMode.RECOVERY, time)

        elif self._mode == ShieldingMode.RECOVERY:
            if condition in (SpaceWeatherCondition.STORM,
                             SpaceWeatherCondition.SEVERE_STORM):
                # Re-enter safe mode
                self._transition_to(ShieldingMode.SAFE_MODE, time)
            elif condition == SpaceWeatherCondition.QUIET and dwell > self.recovery_dwell_time:
                self._transition_to(ShieldingMode.NOMINAL, time)

        # Step 4: Shielding attitude
        if (
            self._mode in (ShieldingMode.ALERT, ShieldingMode.SAFE_MODE)
            and particle_direction is not None
        ):
            self._shielding_quaternion = (
                self.monitor.recommend_shielding_orientation(particle_direction)
            )

        return {
            "mode": self._mode,
            "condition": condition,
            "powered_down": list(self._powered_down),
            "shielding_quat": self._shielding_quaternion.copy(),
            "spe_prediction": spe,
        }
