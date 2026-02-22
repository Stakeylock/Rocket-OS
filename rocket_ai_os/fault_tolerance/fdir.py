"""
Fault Detection, Isolation, and Recovery (FDIR) System.

Implements the standard ECSS-E-ST-70-41C FDIR hierarchy for launch
vehicles:

1. **Detection** -- limit checking, trend analysis, model-based residuals
2. **Isolation** -- identify the specific failed component
3. **Recovery** -- switch to backup, reconfigure, or enter safe mode

Fault types are based on real launch-vehicle failure modes (engine out,
sensor failure, gimbal stuck, fuel leak, etc.).  The FDIR integrates
with the Fault-Tolerant Control Allocation (FTCA) subsystem for engine
faults and uses watchdog timers to monitor partition health.

References
----------
* ECSS-E-ST-70-41C. Telemetry and Telecommand Packet Utilization.
* Wertz et al. (2011). Space Mission Engineering: The New SMAD.
* SpaceX Falcon 9 engine-out capability documentation.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class FaultType(Enum):
    """Categorisation of avionics and vehicle faults."""
    ENGINE_OUT = auto()
    SENSOR_FAILURE = auto()
    COMMS_LOSS = auto()
    POWER_ANOMALY = auto()
    THERMAL_LIMIT = auto()
    STRUCTURAL = auto()
    GIMBAL_STUCK = auto()
    FUEL_LEAK = auto()


class _Severity(Enum):
    """Escalation severity levels."""
    WARNING = 1
    CAUTION = 2
    CRITICAL = 3
    CATASTROPHIC = 4


class _RecoveryAction(Enum):
    """Standard recovery strategies."""
    SWITCH_TO_BACKUP = auto()
    RECONFIGURE = auto()
    REDUCE_THRUST = auto()
    SAFE_MODE = auto()
    ENGINE_SHUTDOWN = auto()
    ABORT = auto()
    NONE = auto()


@dataclass
class FaultRecord:
    """Immutable record of a detected fault."""
    fault_type: FaultType
    severity: int                    # 1=warning .. 4=catastrophic
    timestamp: float                 # mission elapsed time [s]
    subsystem: str                   # e.g. "engine_3", "imu_primary"
    description: str
    isolated: bool = False
    recovered: bool = False


# ---------------------------------------------------------------------------
# Internal helpers -- limit check, trend analysis, model-based detection
# ---------------------------------------------------------------------------

@dataclass
class _LimitSpec:
    """Defines warning/critical limits for a single telemetry channel."""
    channel: str
    warn_low: Optional[float] = None
    warn_high: Optional[float] = None
    crit_low: Optional[float] = None
    crit_high: Optional[float] = None


@dataclass
class _TrendBuffer:
    """Sliding window for trend analysis on a single channel."""
    channel: str
    window_size: int = 50
    _values: list = field(default_factory=list)
    _timestamps: list = field(default_factory=list)

    def push(self, value: float, timestamp: float) -> None:
        self._values.append(value)
        self._timestamps.append(timestamp)
        if len(self._values) > self.window_size:
            self._values.pop(0)
            self._timestamps.pop(0)

    def slope(self) -> Optional[float]:
        """Return linear slope via least-squares, or None if not enough data."""
        n = len(self._values)
        if n < 5:
            return None
        t = np.array(self._timestamps[-n:])
        v = np.array(self._values[-n:])
        t_norm = t - t[0]
        if t_norm[-1] == 0.0:
            return None
        # Least-squares slope
        A = np.vstack([t_norm, np.ones(n)]).T
        result = np.linalg.lstsq(A, v, rcond=None)
        return float(result[0][0])


@dataclass
class _WatchdogTimer:
    """Watchdog for a partition -- must be kicked before deadline."""
    partition: str
    deadline_ms: float
    _last_kick: float = 0.0

    def kick(self, current_time_ms: float) -> None:
        self._last_kick = current_time_ms

    def expired(self, current_time_ms: float) -> bool:
        return (current_time_ms - self._last_kick) > self.deadline_ms


# ---------------------------------------------------------------------------
# Fault tree & escalation
# ---------------------------------------------------------------------------

_FAULT_TREE: Dict[FaultType, Dict] = {
    FaultType.ENGINE_OUT: {
        "severity": _Severity.CRITICAL,
        "affected": ["propulsion", "guidance", "flight_control"],
        "recovery": _RecoveryAction.RECONFIGURE,
        "escalates_to": FaultType.STRUCTURAL,  # if multiple engines
        "description": "Engine has ceased producing thrust",
    },
    FaultType.SENSOR_FAILURE: {
        "severity": _Severity.CAUTION,
        "affected": ["navigation"],
        "recovery": _RecoveryAction.SWITCH_TO_BACKUP,
        "escalates_to": None,
        "description": "Sensor output out of range or stuck",
    },
    FaultType.COMMS_LOSS: {
        "severity": _Severity.WARNING,
        "affected": ["comms", "telemetry"],
        "recovery": _RecoveryAction.SWITCH_TO_BACKUP,
        "escalates_to": None,
        "description": "Communication link lost",
    },
    FaultType.POWER_ANOMALY: {
        "severity": _Severity.CAUTION,
        "affected": ["power", "avionics"],
        "recovery": _RecoveryAction.RECONFIGURE,
        "escalates_to": FaultType.STRUCTURAL,
        "description": "Bus voltage or current anomaly",
    },
    FaultType.THERMAL_LIMIT: {
        "severity": _Severity.CAUTION,
        "affected": ["thermal"],
        "recovery": _RecoveryAction.REDUCE_THRUST,
        "escalates_to": FaultType.ENGINE_OUT,
        "description": "Temperature exceeds operational limit",
    },
    FaultType.STRUCTURAL: {
        "severity": _Severity.CATASTROPHIC,
        "affected": ["structure", "propulsion", "all"],
        "recovery": _RecoveryAction.ABORT,
        "escalates_to": None,
        "description": "Structural anomaly detected",
    },
    FaultType.GIMBAL_STUCK: {
        "severity": _Severity.CRITICAL,
        "affected": ["flight_control", "propulsion"],
        "recovery": _RecoveryAction.RECONFIGURE,
        "escalates_to": FaultType.ENGINE_OUT,
        "description": "Gimbal actuator stuck or unresponsive",
    },
    FaultType.FUEL_LEAK: {
        "severity": _Severity.CRITICAL,
        "affected": ["propulsion", "structure"],
        "recovery": _RecoveryAction.SAFE_MODE,
        "escalates_to": FaultType.STRUCTURAL,
        "description": "Propellant leak detected (pressure drop / mass anomaly)",
    },
}


# ---------------------------------------------------------------------------
# FDIR System
# ---------------------------------------------------------------------------

class FDIRSystem:
    """
    Hierarchical Fault Detection, Isolation, and Recovery system.

    Layers
    ------
    1. **Limit checking**: compares telemetry channels against predefined
       warning / critical thresholds.
    2. **Trend analysis**: detects slowly drifting parameters via slope
       estimation on a sliding window.
    3. **Model-based detection**: compares expected vs. measured outputs
       using physics models (residual analysis).
    4. **Watchdog timers**: monitors partition heartbeats; triggers
       fault if a partition fails to respond within its deadline.

    Parameters
    ----------
    ftca_callback : callable, optional
        Called with ``(engine_id: str, action: str)`` when an engine
        fault is detected and the FTCA should reconfigure thrust
        allocation.
    """

    def __init__(self, ftca_callback: Optional[Callable] = None):
        self._ftca_callback = ftca_callback

        # Limit specifications per channel
        self._limits: Dict[str, _LimitSpec] = {}

        # Trend buffers per channel
        self._trends: Dict[str, _TrendBuffer] = {}

        # Slope thresholds (channel -> max absolute slope before warning)
        self._trend_thresholds: Dict[str, float] = {}

        # Model-based residual thresholds (channel -> threshold)
        self._model_thresholds: Dict[str, float] = {}
        self._model_predictors: Dict[str, Callable] = {}

        # Watchdog timers
        self._watchdogs: Dict[str, _WatchdogTimer] = {}

        # Fault log
        self._fault_log: List[FaultRecord] = []

        # Active (un-recovered) faults
        self._active_faults: List[FaultRecord] = []

        # Engine health map for FTCA integration
        self._engine_health: Dict[str, bool] = {}

        # Escalation counter per fault type
        self._escalation_counts: Dict[FaultType, int] = {ft: 0 for ft in FaultType}

        # Install default limits for common channels
        self._install_defaults()

    # -- configuration API --------------------------------------------------

    def add_limit(
        self,
        channel: str,
        warn_low: Optional[float] = None,
        warn_high: Optional[float] = None,
        crit_low: Optional[float] = None,
        crit_high: Optional[float] = None,
    ) -> None:
        """Register limit-check thresholds for a telemetry channel."""
        self._limits[channel] = _LimitSpec(
            channel=channel,
            warn_low=warn_low,
            warn_high=warn_high,
            crit_low=crit_low,
            crit_high=crit_high,
        )
        # Also create a trend buffer
        if channel not in self._trends:
            self._trends[channel] = _TrendBuffer(channel=channel)

    def add_trend_threshold(self, channel: str, max_slope: float) -> None:
        """If the slope of *channel* exceeds *max_slope* a warning is raised."""
        self._trend_thresholds[channel] = max_slope
        if channel not in self._trends:
            self._trends[channel] = _TrendBuffer(channel=channel)

    def add_model_predictor(
        self, channel: str, predictor: Callable, threshold: float
    ) -> None:
        """
        Register a model-based detector.

        *predictor* is called with the full telemetry dict and should
        return the expected value for *channel*.  If the residual
        ``|measured - expected|`` exceeds *threshold* a fault is raised.
        """
        self._model_predictors[channel] = predictor
        self._model_thresholds[channel] = threshold

    def add_watchdog(self, partition: str, deadline_ms: float) -> None:
        """Add a watchdog timer for *partition*."""
        self._watchdogs[partition] = _WatchdogTimer(
            partition=partition, deadline_ms=deadline_ms
        )

    def kick_watchdog(self, partition: str, current_time_ms: float) -> None:
        """Kick (reset) the watchdog for *partition*."""
        wd = self._watchdogs.get(partition)
        if wd is not None:
            wd.kick(current_time_ms)

    def register_engine(self, engine_id: str) -> None:
        """Register an engine for health tracking."""
        self._engine_health[engine_id] = True

    # -- default limits (realistic launch vehicle values) -------------------

    def _install_defaults(self) -> None:
        """Populate sensible defaults for common telemetry channels."""
        # Chamber pressure per engine (Pa) -- typical LOX/RP-1
        for i in range(9):
            eid = f"engine_{i}"
            self.add_limit(
                f"{eid}_chamber_pressure",
                warn_low=6.0e6, warn_high=10.5e6,
                crit_low=4.0e6, crit_high=11.0e6,
            )
            self.add_limit(
                f"{eid}_turbopump_rpm",
                warn_low=20_000.0, warn_high=38_000.0,
                crit_low=15_000.0, crit_high=40_000.0,
            )
            self.register_engine(eid)

        # IMU channels
        for axis in ("x", "y", "z"):
            self.add_limit(
                f"imu_accel_{axis}",
                crit_low=-80.0, crit_high=80.0,
            )
            self.add_limit(
                f"imu_gyro_{axis}",
                warn_low=-np.radians(30), warn_high=np.radians(30),
                crit_low=-np.radians(60), crit_high=np.radians(60),
            )

        # Tank pressure
        self.add_limit("lox_tank_pressure", warn_low=1.5e5, crit_low=1.0e5, crit_high=5.0e5)
        self.add_limit("rp1_tank_pressure", warn_low=1.5e5, crit_low=1.0e5, crit_high=5.0e5)

        # Battery
        self.add_limit("bus_voltage", warn_low=26.0, warn_high=30.0, crit_low=24.0, crit_high=32.0)

        # Thermal
        self.add_limit("avionics_temp", warn_high=70.0, crit_high=85.0)

        # Trend thresholds
        self.add_trend_threshold("lox_tank_pressure", max_slope=-500.0)  # rapid drop = leak
        self.add_trend_threshold("rp1_tank_pressure", max_slope=-500.0)
        self.add_trend_threshold("avionics_temp", max_slope=2.0)         # deg/s

    # -- detection ----------------------------------------------------------

    def detect(self, telemetry: Dict[str, float], timestamp: float) -> List[FaultRecord]:
        """
        Run all three detection layers on the current telemetry snapshot.

        Parameters
        ----------
        telemetry : dict
            Channel name -> numeric value.
        timestamp : float
            Mission elapsed time [s].

        Returns
        -------
        faults : list of FaultRecord
            Newly detected faults this cycle.
        """
        new_faults: List[FaultRecord] = []

        # Layer 1: limit checks
        new_faults.extend(self._check_limits(telemetry, timestamp))

        # Layer 2: trend analysis
        new_faults.extend(self._check_trends(telemetry, timestamp))

        # Layer 3: model-based residuals
        new_faults.extend(self._check_models(telemetry, timestamp))

        # Layer 4: watchdog timers
        current_time_ms = timestamp * 1000.0
        new_faults.extend(self._check_watchdogs(current_time_ms, timestamp))

        # Record all new faults
        for f in new_faults:
            self._fault_log.append(f)
            self._active_faults.append(f)

        return new_faults

    def _check_limits(
        self, telemetry: Dict[str, float], timestamp: float
    ) -> List[FaultRecord]:
        faults: List[FaultRecord] = []
        for channel, value in telemetry.items():
            spec = self._limits.get(channel)
            if spec is None:
                continue

            severity = 0
            direction = ""

            if spec.crit_high is not None and value > spec.crit_high:
                severity = _Severity.CRITICAL.value
                direction = "above critical high"
            elif spec.crit_low is not None and value < spec.crit_low:
                severity = _Severity.CRITICAL.value
                direction = "below critical low"
            elif spec.warn_high is not None and value > spec.warn_high:
                severity = _Severity.WARNING.value
                direction = "above warning high"
            elif spec.warn_low is not None and value < spec.warn_low:
                severity = _Severity.WARNING.value
                direction = "below warning low"

            if severity > 0:
                fault_type = self._classify_channel(channel)
                faults.append(FaultRecord(
                    fault_type=fault_type,
                    severity=severity,
                    timestamp=timestamp,
                    subsystem=channel,
                    description=f"{channel}={value:.4g} {direction}",
                ))
        return faults

    def _check_trends(
        self, telemetry: Dict[str, float], timestamp: float
    ) -> List[FaultRecord]:
        faults: List[FaultRecord] = []

        # Feed trend buffers
        for channel, value in telemetry.items():
            buf = self._trends.get(channel)
            if buf is not None:
                buf.push(value, timestamp)

        # Check slopes
        for channel, max_slope in self._trend_thresholds.items():
            buf = self._trends.get(channel)
            if buf is None:
                continue
            slope = buf.slope()
            if slope is None:
                continue
            if abs(slope) > abs(max_slope):
                fault_type = self._classify_channel(channel)
                faults.append(FaultRecord(
                    fault_type=fault_type,
                    severity=_Severity.CAUTION.value,
                    timestamp=timestamp,
                    subsystem=channel,
                    description=(
                        f"Trend anomaly on {channel}: slope={slope:.4g} "
                        f"(limit {max_slope:.4g})"
                    ),
                ))
        return faults

    def _check_models(
        self, telemetry: Dict[str, float], timestamp: float
    ) -> List[FaultRecord]:
        faults: List[FaultRecord] = []
        for channel, predictor in self._model_predictors.items():
            measured = telemetry.get(channel)
            if measured is None:
                continue
            try:
                expected = predictor(telemetry)
            except Exception:
                continue
            residual = abs(measured - expected)
            threshold = self._model_thresholds[channel]
            if residual > threshold:
                fault_type = self._classify_channel(channel)
                faults.append(FaultRecord(
                    fault_type=fault_type,
                    severity=_Severity.CAUTION.value,
                    timestamp=timestamp,
                    subsystem=channel,
                    description=(
                        f"Model-based residual on {channel}: "
                        f"|{measured:.4g} - {expected:.4g}| = {residual:.4g} "
                        f"> {threshold:.4g}"
                    ),
                ))
        return faults

    def _check_watchdogs(
        self, current_time_ms: float, timestamp: float
    ) -> List[FaultRecord]:
        faults: List[FaultRecord] = []
        for partition, wd in self._watchdogs.items():
            if wd.expired(current_time_ms):
                faults.append(FaultRecord(
                    fault_type=FaultType.COMMS_LOSS,
                    severity=_Severity.CRITICAL.value,
                    timestamp=timestamp,
                    subsystem=partition,
                    description=(
                        f"Watchdog expired for partition '{partition}' "
                        f"(deadline {wd.deadline_ms:.0f} ms)"
                    ),
                ))
        return faults

    # -- isolation ----------------------------------------------------------

    def isolate(self, fault: FaultRecord) -> List[str]:
        """
        Identify which subsystems are affected by *fault*.

        Uses the fault tree to determine propagation paths.

        Returns
        -------
        affected : list of str
            Names of affected subsystems.
        """
        tree_entry = _FAULT_TREE.get(fault.fault_type, {})
        affected = list(tree_entry.get("affected", [fault.subsystem]))

        fault.isolated = True
        logger.info(
            "Fault isolated -- type=%s, subsystem=%s, affected=%s",
            fault.fault_type.name,
            fault.subsystem,
            affected,
        )
        return affected

    # -- recovery -----------------------------------------------------------

    def recover(self, fault: FaultRecord) -> str:
        """
        Execute recovery action for *fault*.

        Returns
        -------
        action_name : str
            A human-readable description of the recovery action taken.
        """
        tree_entry = _FAULT_TREE.get(fault.fault_type, {})
        action = tree_entry.get("recovery", _RecoveryAction.NONE)

        action_desc = self._execute_recovery(fault, action)

        # Check escalation
        self._escalation_counts[fault.fault_type] += 1
        escalates_to = tree_entry.get("escalates_to")
        if (
            escalates_to is not None
            and self._escalation_counts[fault.fault_type] >= 2
        ):
            logger.critical(
                "Fault %s escalating to %s after %d occurrences",
                fault.fault_type.name,
                escalates_to.name,
                self._escalation_counts[fault.fault_type],
            )
            escalated = FaultRecord(
                fault_type=escalates_to,
                severity=min(fault.severity + 1, _Severity.CATASTROPHIC.value),
                timestamp=fault.timestamp,
                subsystem=fault.subsystem,
                description=f"Escalated from {fault.fault_type.name}",
            )
            self._fault_log.append(escalated)
            self._active_faults.append(escalated)

        fault.recovered = True
        # Remove from active list
        self._active_faults = [f for f in self._active_faults if not f.recovered]

        return action_desc

    def _execute_recovery(
        self, fault: FaultRecord, action: _RecoveryAction
    ) -> str:
        """Dispatch the recovery action."""
        if action == _RecoveryAction.SWITCH_TO_BACKUP:
            desc = f"Switched {fault.subsystem} to backup unit"
            logger.info(desc)
            return desc

        if action == _RecoveryAction.RECONFIGURE:
            # Special handling for engine faults -- notify FTCA
            if fault.fault_type in (FaultType.ENGINE_OUT, FaultType.GIMBAL_STUCK):
                engine_id = self._extract_engine_id(fault.subsystem)
                if engine_id and engine_id in self._engine_health:
                    self._engine_health[engine_id] = False
                    if self._ftca_callback is not None:
                        self._ftca_callback(engine_id, "shutdown")
                    desc = (
                        f"Engine {engine_id} shut down; FTCA notified to "
                        f"reallocate thrust"
                    )
                    logger.warning(desc)
                    return desc
            desc = f"Reconfigured {fault.subsystem} (redundancy management)"
            logger.info(desc)
            return desc

        if action == _RecoveryAction.REDUCE_THRUST:
            desc = f"Reduced thrust to protect {fault.subsystem}"
            logger.info(desc)
            return desc

        if action == _RecoveryAction.SAFE_MODE:
            desc = f"Entering safe mode due to {fault.fault_type.name} on {fault.subsystem}"
            logger.warning(desc)
            return desc

        if action == _RecoveryAction.ENGINE_SHUTDOWN:
            engine_id = self._extract_engine_id(fault.subsystem)
            if engine_id and engine_id in self._engine_health:
                self._engine_health[engine_id] = False
                if self._ftca_callback is not None:
                    self._ftca_callback(engine_id, "shutdown")
            desc = f"Emergency shutdown of {fault.subsystem}"
            logger.critical(desc)
            return desc

        if action == _RecoveryAction.ABORT:
            desc = f"ABORT triggered by {fault.fault_type.name} on {fault.subsystem}"
            logger.critical(desc)
            return desc

        return f"No recovery action for {fault.fault_type.name}"

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_engine_id(subsystem: str) -> Optional[str]:
        """Try to pull an engine_N identifier from a subsystem string."""
        # Handles "engine_3_chamber_pressure" -> "engine_3"
        parts = subsystem.split("_")
        for i, p in enumerate(parts):
            if p == "engine" and i + 1 < len(parts):
                try:
                    int(parts[i + 1])
                    return f"engine_{parts[i + 1]}"
                except ValueError:
                    pass
        return None

    @staticmethod
    def _classify_channel(channel: str) -> FaultType:
        """Map a telemetry channel name to a FaultType."""
        ch = channel.lower()
        if "engine" in ch and ("chamber" in ch or "turbopump" in ch):
            return FaultType.ENGINE_OUT
        if "gimbal" in ch:
            return FaultType.GIMBAL_STUCK
        if "imu" in ch or "gps" in ch or "sensor" in ch:
            return FaultType.SENSOR_FAILURE
        if "voltage" in ch or "current" in ch or "power" in ch or "bus" in ch:
            return FaultType.POWER_ANOMALY
        if "temp" in ch or "thermal" in ch:
            return FaultType.THERMAL_LIMIT
        if "tank" in ch or "fuel" in ch or "lox" in ch or "rp1" in ch:
            return FaultType.FUEL_LEAK
        if "comm" in ch or "link" in ch:
            return FaultType.COMMS_LOSS
        if "struct" in ch or "strain" in ch or "vibration" in ch:
            return FaultType.STRUCTURAL
        return FaultType.SENSOR_FAILURE  # conservative default

    # -- query API ----------------------------------------------------------

    @property
    def fault_log(self) -> List[FaultRecord]:
        """Full history of detected faults."""
        return list(self._fault_log)

    @property
    def active_faults(self) -> List[FaultRecord]:
        """Currently un-recovered faults."""
        return list(self._active_faults)

    @property
    def engine_health(self) -> Dict[str, bool]:
        return dict(self._engine_health)

    def healthy_engine_count(self) -> int:
        return sum(1 for v in self._engine_health.values() if v)

    def stats(self) -> Dict:
        return {
            "total_faults_detected": len(self._fault_log),
            "active_faults": len(self._active_faults),
            "healthy_engines": self.healthy_engine_count(),
            "total_engines": len(self._engine_health),
            "escalation_counts": {
                ft.name: c for ft, c in self._escalation_counts.items() if c > 0
            },
        }
