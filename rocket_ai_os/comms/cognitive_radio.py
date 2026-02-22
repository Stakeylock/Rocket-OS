"""
Cognitive Radio Engine and Link Recovery System.

Implements software-defined radio concepts for adaptive communication:
- Spectrum sensing with interference and jamming detection
- Policy-based dynamic spectrum access and modulation switching
- Link budget calculation with rain fade and atmospheric attenuation
- Automatic modulation degradation under falling SNR
- Hierarchical lost-link recovery (sidelobe search, attitude recovery, handshake)

References:
    - NASA SCaN Cognitive Radio architecture
    - CCSDS Proximity-1 Space Link Protocol
    - ITU-R P.618 rain attenuation model (simplified)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Band(Enum):
    """RF / optical communication bands used by the vehicle."""
    X_BAND = auto()    # 8-12 GHz  (deep-space / TT&C)
    KA_BAND = auto()   # 26-40 GHz (high-throughput)
    S_BAND = auto()    # 2-4 GHz   (legacy TT&C)
    OPTICAL = auto()   # Free-space optical link


class ModulationScheme(Enum):
    """Supported digital modulation schemes, ordered by spectral efficiency."""
    BPSK = auto()    # 1 bit/symbol – most robust
    QPSK = auto()    # 2 bits/symbol
    QAM16 = auto()   # 4 bits/symbol
    QAM64 = auto()   # 6 bits/symbol – highest throughput


class RecoveryPhase(Enum):
    """State-machine phases for hierarchical lost-link recovery."""
    NOMINAL = auto()
    SIDELOBE_SEARCH = auto()
    ATTITUDE_RECOVERY = auto()
    PROTOCOL_HANDSHAKE = auto()
    RECOVERED = auto()
    FAILED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LinkState:
    """Snapshot of the current communication link parameters."""
    band: Band
    modulation: ModulationScheme
    signal_strength_dbm: float
    bit_rate_bps: float
    bit_error_rate: float
    is_active: bool


# ---------------------------------------------------------------------------
# Constants / band characteristics
# ---------------------------------------------------------------------------

# Nominal centre frequencies (Hz)
_BAND_FREQ_HZ: Dict[Band, float] = {
    Band.S_BAND: 2.2e9,
    Band.X_BAND: 8.4e9,
    Band.KA_BAND: 32.0e9,
    Band.OPTICAL: 2.8e14,  # ~1064 nm laser
}

# Typical free-space noise figure per band (dB)
_BAND_NOISE_FIGURE_DB: Dict[Band, float] = {
    Band.S_BAND: 1.5,
    Band.X_BAND: 2.0,
    Band.KA_BAND: 3.5,
    Band.OPTICAL: 6.0,
}

# Maximum achievable data rate per band (bps) – hardware limit
_BAND_MAX_RATE_BPS: Dict[Band, float] = {
    Band.S_BAND: 2e6,
    Band.X_BAND: 50e6,
    Band.KA_BAND: 1e9,
    Band.OPTICAL: 10e9,
}

# Rain-fade susceptibility factor (dB per mm/h) – simplified ITU-R P.618
_RAIN_FADE_FACTOR: Dict[Band, float] = {
    Band.S_BAND: 0.01,
    Band.X_BAND: 0.05,
    Band.KA_BAND: 0.25,
    Band.OPTICAL: 1.0,   # very susceptible to weather
}

# Bits per symbol for each modulation scheme
_BITS_PER_SYMBOL: Dict[ModulationScheme, int] = {
    ModulationScheme.BPSK: 1,
    ModulationScheme.QPSK: 2,
    ModulationScheme.QAM16: 4,
    ModulationScheme.QAM64: 6,
}

# Approximate Eb/N0 required (dB) for BER < 1e-6 (AWGN)
_REQUIRED_SNR_DB: Dict[ModulationScheme, float] = {
    ModulationScheme.BPSK: 10.5,
    ModulationScheme.QPSK: 10.5,
    ModulationScheme.QAM16: 14.5,
    ModulationScheme.QAM64: 18.5,
}

# Ordered modulation degradation path (highest -> lowest throughput)
_MODULATION_LADDER: List[ModulationScheme] = [
    ModulationScheme.QAM64,
    ModulationScheme.QAM16,
    ModulationScheme.QPSK,
    ModulationScheme.BPSK,
]

# Speed of light (m/s)
_C = 299_792_458.0
# Boltzmann constant (J/K)
_K_BOLTZ = 1.380649e-23


# ---------------------------------------------------------------------------
# Cognitive Radio Engine
# ---------------------------------------------------------------------------

class CognitiveRadioEngine:
    """
    Software-defined radio engine with cognitive spectrum management.

    Performs cyclic sense-decide-act:
      1. Sense  – energy detection across bands, interference mapping.
      2. Decide – policy engine selects best band & modulation.
      3. Act    – hardware reconfiguration (simulated).

    Parameters
    ----------
    rng_seed : int
        Seed for reproducible random interference/fade simulation.
    transmit_power_dbm : float
        Transmitter output power in dBm.
    antenna_gain_dbi : float
        Antenna gain in dBi (both Tx and Rx assumed equal).
    system_noise_temp_k : float
        System noise temperature in Kelvin.
    symbol_rate : float
        Base symbol rate in symbols/sec.
    """

    def __init__(
        self,
        rng_seed: int = 42,
        transmit_power_dbm: float = 40.0,
        antenna_gain_dbi: float = 35.0,
        system_noise_temp_k: float = 290.0,
        symbol_rate: float = 1e6,
    ) -> None:
        self._rng = np.random.default_rng(rng_seed)
        self._tx_power_dbm = transmit_power_dbm
        self._antenna_gain_dbi = antenna_gain_dbi
        self._noise_temp_k = system_noise_temp_k
        self._symbol_rate = symbol_rate

        # Current link state
        self._band = Band.X_BAND
        self._modulation = ModulationScheme.QPSK
        self._is_active = True

        # Simulated environment conditions
        self._rain_rate_mm_h: float = 0.0          # mm/h
        self._jamming_power_dbm: Dict[Band, float] = {b: -np.inf for b in Band}
        self._distance_m: float = 500_000.0         # default 500 km slant range

        # History
        self._reconfiguration_log: List[Dict] = []

    # -- public properties ---------------------------------------------------

    @property
    def band(self) -> Band:
        return self._band

    @property
    def modulation(self) -> ModulationScheme:
        return self._modulation

    @property
    def is_active(self) -> bool:
        return self._is_active

    # -- environment setters (for simulation) --------------------------------

    def set_rain_rate(self, rate_mm_h: float) -> None:
        """Set current rain rate for fade simulation."""
        self._rain_rate_mm_h = max(0.0, rate_mm_h)

    def set_jamming(self, band: Band, power_dbm: float) -> None:
        """Inject a jamming signal on a specific band."""
        self._jamming_power_dbm[band] = power_dbm

    def clear_jamming(self, band: Optional[Band] = None) -> None:
        """Remove jamming (on one band or all bands)."""
        if band is not None:
            self._jamming_power_dbm[band] = -np.inf
        else:
            self._jamming_power_dbm = {b: -np.inf for b in Band}

    def set_distance(self, distance_m: float) -> None:
        """Set the Tx-Rx slant range in metres."""
        self._distance_m = max(1.0, distance_m)

    # -- link budget ---------------------------------------------------------

    def link_budget(self, band: Optional[Band] = None) -> Dict[str, float]:
        """
        Compute the link budget for the specified (or current) band.

        Returns a dictionary with all link-budget terms in dB/dBm.
        """
        band = band or self._band
        freq_hz = _BAND_FREQ_HZ[band]
        wavelength_m = _C / freq_hz

        # Free-space path loss (dB)
        if self._distance_m > 0 and wavelength_m > 0:
            fspl_db = 20.0 * np.log10(4.0 * np.pi * self._distance_m / wavelength_m)
        else:
            fspl_db = 0.0

        # Rain attenuation (dB)
        rain_atten_db = _RAIN_FADE_FACTOR[band] * self._rain_rate_mm_h

        # Noise floor (dBm) in symbol-rate bandwidth
        noise_power_w = _K_BOLTZ * self._noise_temp_k * self._symbol_rate
        noise_floor_dbm = 10.0 * np.log10(noise_power_w) + 30.0  # W -> dBm
        noise_floor_dbm += _BAND_NOISE_FIGURE_DB[band]

        # Received signal power (dBm)
        rx_power_dbm = (
            self._tx_power_dbm
            + 2.0 * self._antenna_gain_dbi      # Tx + Rx gain
            - fspl_db
            - rain_atten_db
        )

        # Jamming contribution
        jam_dbm = self._jamming_power_dbm[band]
        effective_noise_dbm = noise_floor_dbm
        if np.isfinite(jam_dbm):
            # Add jamming power to noise floor (linear sum)
            noise_w = 10.0 ** ((noise_floor_dbm - 30.0) / 10.0)
            jam_w = 10.0 ** ((jam_dbm - 30.0) / 10.0)
            effective_noise_dbm = 10.0 * np.log10(noise_w + jam_w) + 30.0

        # SNR (dB)
        snr_db = rx_power_dbm - effective_noise_dbm

        return {
            "tx_power_dbm": self._tx_power_dbm,
            "antenna_gain_dbi": self._antenna_gain_dbi,
            "fspl_db": fspl_db,
            "rain_atten_db": rain_atten_db,
            "rx_power_dbm": rx_power_dbm,
            "noise_floor_dbm": noise_floor_dbm,
            "effective_noise_dbm": effective_noise_dbm,
            "jamming_dbm": jam_dbm if np.isfinite(jam_dbm) else None,
            "snr_db": snr_db,
        }

    # -- spectrum sensing ----------------------------------------------------

    def sense_spectrum(self) -> Dict[Band, Dict[str, float]]:
        """
        Perform energy-detection spectrum sensing across all bands.

        Returns
        -------
        interference_map : dict
            Per-band dictionary with keys:
            - ``snr_db``        : measured signal-to-noise ratio
            - ``interference_dbm`` : detected interference / jamming level
            - ``usable``        : bool flag, True if band is operationally usable
            - ``rain_atten_db`` : estimated rain fade
        """
        interference_map: Dict[Band, Dict[str, float]] = {}
        for band in Band:
            budget = self.link_budget(band)
            snr_db = budget["snr_db"]
            jam = self._jamming_power_dbm[band]
            rain_atten = budget["rain_atten_db"]

            # Band is usable if SNR exceeds the lowest-modulation requirement
            min_snr_required = _REQUIRED_SNR_DB[ModulationScheme.BPSK]
            usable = bool(snr_db >= min_snr_required)

            interference_map[band] = {
                "snr_db": float(snr_db),
                "interference_dbm": float(jam) if np.isfinite(jam) else None,
                "usable": usable,
                "rain_atten_db": float(rain_atten),
            }
        return interference_map

    # -- BER estimation ------------------------------------------------------

    @staticmethod
    def _estimate_ber(modulation: ModulationScheme, snr_db: float) -> float:
        """
        Approximate BER for *modulation* at *snr_db* (AWGN channel).

        Uses closed-form Q-function approximations.
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        if snr_linear <= 0:
            return 0.5

        if modulation == ModulationScheme.BPSK:
            # BER = Q(sqrt(2*Eb/N0))
            arg = np.sqrt(2.0 * snr_linear)
            ber = 0.5 * np.exp(-arg ** 2 / 2.0)  # upper-bound approx
        elif modulation == ModulationScheme.QPSK:
            arg = np.sqrt(2.0 * snr_linear)
            ber = 0.5 * np.exp(-arg ** 2 / 2.0)
        elif modulation == ModulationScheme.QAM16:
            arg = np.sqrt(0.8 * snr_linear)
            ber = 0.75 * np.exp(-arg ** 2 / 2.0)
        elif modulation == ModulationScheme.QAM64:
            arg = np.sqrt(2.0 / 7.0 * snr_linear)
            ber = 7.0 / 12.0 * np.exp(-arg ** 2 / 2.0)
        else:
            ber = 0.5

        return float(np.clip(ber, 0.0, 0.5))

    # -- reconfiguration -----------------------------------------------------

    def _select_best_modulation(self, snr_db: float, target_ber: float) -> ModulationScheme:
        """Pick the highest-throughput modulation whose BER <= target_ber."""
        for mod in _MODULATION_LADDER:
            ber = self._estimate_ber(mod, snr_db)
            if ber <= target_ber:
                return mod
        # Fallback to BPSK if nothing meets threshold
        return ModulationScheme.BPSK

    def _select_best_band(self, interference_map: Dict[Band, Dict]) -> Band:
        """
        Pick the band with the highest usable SNR.

        Prefers the current band if it is still usable (hysteresis).
        """
        best_band = self._band
        best_snr = -np.inf

        current_info = interference_map.get(self._band)
        if current_info and current_info["usable"]:
            best_snr = current_info["snr_db"]
            best_band = self._band

        for band, info in interference_map.items():
            if info["usable"] and info["snr_db"] > best_snr + 3.0:
                # 3 dB hysteresis before switching away from current band
                best_snr = info["snr_db"]
                best_band = band

        return best_band

    def reconfigure(self, target_ber: float = 1e-6) -> LinkState:
        """
        Sense the spectrum and reconfigure band / modulation to meet *target_ber*.

        Returns the new :class:`LinkState`.
        """
        interference_map = self.sense_spectrum()

        # Select best band
        new_band = self._select_best_band(interference_map)
        snr_db = interference_map[new_band]["snr_db"]

        # Select best modulation for that SNR
        new_mod = self._select_best_modulation(snr_db, target_ber)

        # Compute achieved BER and data rate
        ber = self._estimate_ber(new_mod, snr_db)
        bits_per_symbol = _BITS_PER_SYMBOL[new_mod]
        bit_rate = self._symbol_rate * bits_per_symbol
        bit_rate = min(bit_rate, _BAND_MAX_RATE_BPS[new_band])

        # Received power from budget
        budget = self.link_budget(new_band)
        rx_power_dbm = budget["rx_power_dbm"]

        # Determine if the link is viable at all
        is_active = bool(snr_db >= _REQUIRED_SNR_DB[ModulationScheme.BPSK])

        # Apply changes
        old_band, old_mod = self._band, self._modulation
        self._band = new_band
        self._modulation = new_mod
        self._is_active = is_active

        # Log the event
        self._reconfiguration_log.append({
            "timestamp": time.monotonic(),
            "old_band": old_band,
            "new_band": new_band,
            "old_modulation": old_mod,
            "new_modulation": new_mod,
            "snr_db": snr_db,
            "ber": ber,
            "bit_rate_bps": bit_rate,
            "is_active": is_active,
        })

        return LinkState(
            band=new_band,
            modulation=new_mod,
            signal_strength_dbm=rx_power_dbm,
            bit_rate_bps=bit_rate,
            bit_error_rate=ber,
            is_active=is_active,
        )

    def get_link_state(self) -> LinkState:
        """Return the current link state without reconfiguration."""
        budget = self.link_budget()
        snr_db = budget["snr_db"]
        ber = self._estimate_ber(self._modulation, snr_db)
        bits_per_symbol = _BITS_PER_SYMBOL[self._modulation]
        bit_rate = min(
            self._symbol_rate * bits_per_symbol,
            _BAND_MAX_RATE_BPS[self._band],
        )
        return LinkState(
            band=self._band,
            modulation=self._modulation,
            signal_strength_dbm=budget["rx_power_dbm"],
            bit_rate_bps=bit_rate,
            bit_error_rate=ber,
            is_active=self._is_active,
        )

    @property
    def reconfiguration_log(self) -> List[Dict]:
        """Return a copy of the reconfiguration history."""
        return list(self._reconfiguration_log)


# ---------------------------------------------------------------------------
# Link Recovery System
# ---------------------------------------------------------------------------

class LinkRecoverySystem:
    """
    Hierarchical lost-link recovery procedure.

    Recovery phases (executed in order until success or timeout):
      1. **Sidelobe search** – sweep phased-array beam in a spiral pattern.
      2. **Attitude recovery** – physically reorient the vehicle antenna boresight.
      3. **Protocol handshake** – reinitiate CCSDS session negotiation.

    The system operates as a state machine driven by ``attempt_recovery()``.

    Parameters
    ----------
    radio : CognitiveRadioEngine
        Reference to the radio engine whose link we are recovering.
    sidelobe_sweep_steps : int
        Number of beam-steering positions to try during sidelobe search.
    attitude_search_cone_deg : float
        Half-cone angle for the attitude search pattern.
    max_handshake_attempts : int
        Maximum protocol handshake retries.
    rng_seed : int
        Seed for reproducible simulation.
    """

    def __init__(
        self,
        radio: CognitiveRadioEngine,
        sidelobe_sweep_steps: int = 36,
        attitude_search_cone_deg: float = 15.0,
        max_handshake_attempts: int = 5,
        rng_seed: int = 42,
    ) -> None:
        self._radio = radio
        self._sweep_steps = sidelobe_sweep_steps
        self._attitude_cone_deg = attitude_search_cone_deg
        self._max_handshake = max_handshake_attempts
        self._rng = np.random.default_rng(rng_seed)

        self._phase = RecoveryPhase.NOMINAL
        self._phase_history: List[Dict] = []

        # Internal counters
        self._sweep_index: int = 0
        self._handshake_attempts: int = 0

        # Simulated beam offsets (az, el) in degrees
        self._beam_offset_deg: np.ndarray = np.zeros(2)
        # Simulated attitude correction (roll, pitch, yaw) in degrees
        self._attitude_correction_deg: np.ndarray = np.zeros(3)

    # -- properties ----------------------------------------------------------

    @property
    def phase(self) -> RecoveryPhase:
        return self._phase

    @property
    def phase_history(self) -> List[Dict]:
        return list(self._phase_history)

    # -- internal helpers ----------------------------------------------------

    def _log_transition(self, from_phase: RecoveryPhase, to_phase: RecoveryPhase,
                        detail: str = "") -> None:
        self._phase_history.append({
            "timestamp": time.monotonic(),
            "from": from_phase,
            "to": to_phase,
            "detail": detail,
        })

    def _sidelobe_search(self) -> bool:
        """
        Sweep the phased-array beam across a spiral pattern.

        Returns True if a viable link is found during the sweep.
        """
        for i in range(self._sweep_index, self._sweep_steps):
            # Spiral scan: increasing radius, rotating azimuth
            radius_deg = (i / self._sweep_steps) * self._attitude_cone_deg
            azimuth_deg = (i * 137.508) % 360.0  # golden-angle spiral

            self._beam_offset_deg = np.array([
                radius_deg * np.cos(np.radians(azimuth_deg)),
                radius_deg * np.sin(np.radians(azimuth_deg)),
            ])

            # Simulate: pointing offset reduces effective antenna gain
            offset_mag = np.linalg.norm(self._beam_offset_deg)
            # Gaussian beam pattern: gain drops with offset^2
            gain_loss_db = 0.1 * offset_mag ** 2

            # Temporarily adjust the radio's effective gain
            original_gain = self._radio._antenna_gain_dbi
            self._radio._antenna_gain_dbi = original_gain - gain_loss_db

            link = self._radio.reconfigure()

            # Restore gain
            self._radio._antenna_gain_dbi = original_gain

            self._sweep_index = i + 1

            if link.is_active and link.bit_error_rate < 1e-4:
                return True

        return False

    def _attitude_recovery(self) -> bool:
        """
        Simulate vehicle reorientation towards the ground station.

        Uses a simple Monte Carlo search within the attitude cone.
        """
        n_attempts = 12
        for _ in range(n_attempts):
            # Random attitude correction within the search cone
            az = self._rng.uniform(0.0, 360.0)
            el = self._rng.uniform(0.0, self._attitude_cone_deg)

            correction = np.array([
                el * np.cos(np.radians(az)),
                el * np.sin(np.radians(az)),
                self._rng.uniform(-5.0, 5.0),  # slight roll
            ])
            self._attitude_correction_deg = correction

            # Pointing improvement reduces effective distance (simplified model)
            pointing_error = np.linalg.norm(correction[:2])
            gain_recovery_db = max(0.0, self._attitude_cone_deg - pointing_error)

            original_gain = self._radio._antenna_gain_dbi
            self._radio._antenna_gain_dbi = original_gain + gain_recovery_db * 0.5

            link = self._radio.reconfigure()

            self._radio._antenna_gain_dbi = original_gain

            if link.is_active and link.bit_error_rate < 1e-4:
                return True

        return False

    def _protocol_handshake(self) -> bool:
        """
        Attempt CCSDS-style session negotiation.

        Simulates handshake success probability based on current link quality.
        """
        for attempt in range(self._handshake_attempts, self._max_handshake):
            self._handshake_attempts = attempt + 1

            link = self._radio.get_link_state()
            if not link.is_active:
                continue

            # Handshake success probability rises with SNR
            budget = self._radio.link_budget()
            snr_db = budget["snr_db"]
            # Sigmoid success probability centred at 8 dB
            p_success = 1.0 / (1.0 + np.exp(-(snr_db - 8.0)))

            if self._rng.random() < p_success:
                return True

        return False

    # -- public API ----------------------------------------------------------

    def reset(self) -> None:
        """Reset the recovery state machine to NOMINAL."""
        self._phase = RecoveryPhase.NOMINAL
        self._sweep_index = 0
        self._handshake_attempts = 0
        self._beam_offset_deg = np.zeros(2)
        self._attitude_correction_deg = np.zeros(3)

    def attempt_recovery(self) -> Dict[str, object]:
        """
        Execute one cycle of the hierarchical lost-link recovery.

        The method walks through the recovery phases in order.  Call
        repeatedly (or once) from the comms partition scheduler.

        Returns
        -------
        recovery_status : dict
            ``phase``           : current :class:`RecoveryPhase`
            ``success``         : bool – True if link recovered
            ``beam_offset_deg`` : array – current beam steering offset
            ``attitude_correction_deg`` : array – applied attitude correction
            ``handshake_attempts``      : int – number of handshake retries used
            ``detail``          : str – human-readable status message
        """

        def _status(success: bool, detail: str) -> Dict[str, object]:
            return {
                "phase": self._phase,
                "success": success,
                "beam_offset_deg": self._beam_offset_deg.copy(),
                "attitude_correction_deg": self._attitude_correction_deg.copy(),
                "handshake_attempts": self._handshake_attempts,
                "detail": detail,
            }

        # -- Phase 1: Sidelobe Search ----------------------------------------
        if self._phase in (RecoveryPhase.NOMINAL, RecoveryPhase.SIDELOBE_SEARCH):
            prev = self._phase
            self._phase = RecoveryPhase.SIDELOBE_SEARCH
            if prev != RecoveryPhase.SIDELOBE_SEARCH:
                self._log_transition(prev, self._phase, "Starting sidelobe search")

            if self._sidelobe_search():
                self._phase = RecoveryPhase.RECOVERED
                self._log_transition(RecoveryPhase.SIDELOBE_SEARCH,
                                     RecoveryPhase.RECOVERED,
                                     "Link recovered via sidelobe search")
                return _status(True, "Recovered via sidelobe search")

            # Move to next phase
            self._log_transition(RecoveryPhase.SIDELOBE_SEARCH,
                                 RecoveryPhase.ATTITUDE_RECOVERY,
                                 "Sidelobe search exhausted")

        # -- Phase 2: Attitude Recovery --------------------------------------
        if self._phase in (RecoveryPhase.SIDELOBE_SEARCH, RecoveryPhase.ATTITUDE_RECOVERY):
            prev = self._phase
            self._phase = RecoveryPhase.ATTITUDE_RECOVERY
            if prev != RecoveryPhase.ATTITUDE_RECOVERY:
                self._log_transition(prev, self._phase, "Starting attitude recovery")

            if self._attitude_recovery():
                self._phase = RecoveryPhase.RECOVERED
                self._log_transition(RecoveryPhase.ATTITUDE_RECOVERY,
                                     RecoveryPhase.RECOVERED,
                                     "Link recovered via attitude reorientation")
                return _status(True, "Recovered via attitude recovery")

            self._log_transition(RecoveryPhase.ATTITUDE_RECOVERY,
                                 RecoveryPhase.PROTOCOL_HANDSHAKE,
                                 "Attitude recovery failed")

        # -- Phase 3: Protocol Handshake -------------------------------------
        if self._phase in (RecoveryPhase.ATTITUDE_RECOVERY, RecoveryPhase.PROTOCOL_HANDSHAKE):
            prev = self._phase
            self._phase = RecoveryPhase.PROTOCOL_HANDSHAKE
            if prev != RecoveryPhase.PROTOCOL_HANDSHAKE:
                self._log_transition(prev, self._phase, "Starting protocol handshake")

            if self._protocol_handshake():
                self._phase = RecoveryPhase.RECOVERED
                self._log_transition(RecoveryPhase.PROTOCOL_HANDSHAKE,
                                     RecoveryPhase.RECOVERED,
                                     "Link recovered via handshake")
                return _status(True, "Recovered via protocol handshake")

            self._phase = RecoveryPhase.FAILED
            self._log_transition(RecoveryPhase.PROTOCOL_HANDSHAKE,
                                 RecoveryPhase.FAILED,
                                 "All recovery phases exhausted")

        return _status(False, "Recovery failed – all phases exhausted")
