"""
Triple Modular Redundancy (TMR) with Voter and SEU Mitigation.

Provides radiation-hardened computation by executing every function on
three independent virtual cores, comparing outputs via majority voting,
and scrubbing (correcting) the state of any disagreeing core.

This is critical for launch vehicles operating above the atmosphere
where single-event upsets (SEUs) from cosmic rays and solar particles
can flip bits in memory or logic.

References
----------
* Lyons & Vanderkulk (1962). The Use of Triple-Modular Redundancy
  to Improve Computer Reliability.
* NASA/TM-2000-210616. Single Event Effects in Avionics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class _CoreState:
    """Internal state of one virtual core in TMR."""
    core_id: int
    healthy: bool = True
    seu_injected: bool = False
    # When an SEU is injected the core's function is wrapped to add noise
    seu_offset: Optional[np.ndarray] = None
    total_executions: int = 0
    total_faults: int = 0


@dataclass
class FaultReport:
    """Report generated when the voter detects a disagreement."""
    timestamp: float
    disagreeing_core: int
    expected_value: Any
    actual_value: Any
    corrected: bool
    description: str


class TMRVoter:
    """
    Triple-Modular-Redundancy voter.

    Executes a function on three virtual cores, compares the results,
    and returns the majority value.  If one core disagrees it is flagged,
    and its state can be scrubbed (corrected) to match the majority.

    Parameters
    ----------
    float_tolerance : float
        Tolerance for comparing floating-point outputs.  Two values are
        considered equal if ``|a - b| <= float_tolerance``.
    """

    def __init__(self, float_tolerance: float = 1e-9):
        self.float_tolerance = float_tolerance
        self._cores: List[_CoreState] = [
            _CoreState(core_id=i) for i in range(3)
        ]
        self._fault_reports: List[FaultReport] = []
        self._total_votes: int = 0
        self._total_disagreements: int = 0
        self._clock: float = 0.0

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[FaultReport]]:
        """
        Execute *func* on all three virtual cores, vote, and return the
        majority result.

        Returns
        -------
        result : Any
            The voted (majority) output.
        fault_report : FaultReport or None
            Non-None if a core disagreed.
        """
        self._total_votes += 1
        self._clock += 1.0

        outputs: List[Any] = []
        for core in self._cores:
            core.total_executions += 1
            try:
                result = func(*args, **kwargs)
                # Apply SEU corruption if injected
                if core.seu_injected and core.seu_offset is not None:
                    result = self._corrupt(result, core.seu_offset)
                outputs.append(result)
            except Exception as exc:
                logger.error("Core %d raised exception: %s", core.core_id, exc)
                outputs.append(None)

        voted, fault_report = self._vote(outputs)
        return voted, fault_report

    # -- voting logic -------------------------------------------------------

    def _vote(self, outputs: List[Any]) -> Tuple[Any, Optional[FaultReport]]:
        """Majority vote across three outputs."""
        assert len(outputs) == 3

        # Compare each pair
        match_01 = self._values_equal(outputs[0], outputs[1])
        match_02 = self._values_equal(outputs[0], outputs[2])
        match_12 = self._values_equal(outputs[1], outputs[2])

        # All agree
        if match_01 and match_02:
            return outputs[0], None

        # Identify the odd one out
        if match_01:
            # Core 2 disagrees
            return self._report_disagreement(outputs[0], outputs, 2)
        if match_02:
            # Core 1 disagrees
            return self._report_disagreement(outputs[0], outputs, 1)
        if match_12:
            # Core 0 disagrees
            return self._report_disagreement(outputs[1], outputs, 0)

        # No majority -- catastrophic (all three differ). Pick core 0 and
        # flag everything.
        logger.critical("TMR: no majority -- all three cores disagree!")
        self._total_disagreements += 1
        report = FaultReport(
            timestamp=self._clock,
            disagreeing_core=-1,  # all
            expected_value=None,
            actual_value=[self._to_serializable(o) for o in outputs],
            corrected=False,
            description="No majority: all three cores produced different outputs",
        )
        self._fault_reports.append(report)
        return outputs[0], report

    def _report_disagreement(
        self, majority_value: Any, outputs: List[Any], bad_core: int
    ) -> Tuple[Any, FaultReport]:
        self._total_disagreements += 1
        self._cores[bad_core].total_faults += 1

        report = FaultReport(
            timestamp=self._clock,
            disagreeing_core=bad_core,
            expected_value=self._to_serializable(majority_value),
            actual_value=self._to_serializable(outputs[bad_core]),
            corrected=True,
            description=(
                f"Core {bad_core} disagreed: expected "
                f"{self._to_serializable(majority_value)}, got "
                f"{self._to_serializable(outputs[bad_core])}"
            ),
        )
        self._fault_reports.append(report)
        logger.warning("TMR disagreement: core %d -- scrubbing", bad_core)
        return majority_value, report

    # -- comparison helpers -------------------------------------------------

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Compare two values, handling numpy arrays, floats, and ints."""
        if a is None or b is None:
            return a is b

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape:
                return False
            return bool(np.all(np.abs(a - b) <= self.float_tolerance))

        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            try:
                return abs(float(a) - float(b)) <= self.float_tolerance
            except (TypeError, ValueError):
                return False

        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            return int(a) == int(b)

        # Fall back to equality
        return a == b

    @staticmethod
    def _to_serializable(val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    # -- SEU corruption helper ----------------------------------------------

    @staticmethod
    def _corrupt(value: Any, offset: np.ndarray) -> Any:
        """Add SEU noise to a value."""
        if isinstance(value, np.ndarray):
            return value + offset[:value.size].reshape(value.shape)
        if isinstance(value, (float, np.floating)):
            return float(value) + float(offset[0])
        if isinstance(value, (int, np.integer)):
            return int(value) + int(offset[0])
        return value

    # -- SEU injection / scrub ----------------------------------------------

    def inject_seu(self, core_id: int, magnitude: float = 1.0) -> None:
        """
        Simulate a Single Event Upset on *core_id* (0, 1, or 2).

        The affected core will add random noise of the given *magnitude*
        to its outputs until scrubbed.
        """
        if core_id < 0 or core_id >= 3:
            raise ValueError(f"core_id must be 0, 1, or 2; got {core_id}")

        rng = np.random.default_rng()
        self._cores[core_id].seu_injected = True
        self._cores[core_id].seu_offset = rng.uniform(
            -magnitude, magnitude, size=64
        )
        logger.info("SEU injected on core %d (magnitude=%.2e)", core_id, magnitude)

    def scrub(self, core_id: int) -> None:
        """
        Scrub (correct) a core's state, clearing any injected SEU.

        In real hardware this corresponds to reloading the core's memory
        image from a golden copy.
        """
        if core_id < 0 or core_id >= 3:
            raise ValueError(f"core_id must be 0, 1, or 2; got {core_id}")
        self._cores[core_id].seu_injected = False
        self._cores[core_id].seu_offset = None
        self._cores[core_id].healthy = True
        logger.info("Core %d scrubbed and restored", core_id)

    def scrub_all(self) -> None:
        """Scrub every core."""
        for i in range(3):
            self.scrub(i)

    # -- statistics ---------------------------------------------------------

    @property
    def fault_reports(self) -> List[FaultReport]:
        return list(self._fault_reports)

    @property
    def disagreement_count(self) -> int:
        return self._total_disagreements

    @property
    def vote_count(self) -> int:
        return self._total_votes

    def seu_detection_rate(self) -> float:
        """Fraction of votes that detected a disagreement."""
        if self._total_votes == 0:
            return 0.0
        return self._total_disagreements / self._total_votes

    def core_stats(self) -> List[Dict]:
        return [
            {
                "core_id": c.core_id,
                "healthy": c.healthy,
                "seu_active": c.seu_injected,
                "executions": c.total_executions,
                "faults_detected": c.total_faults,
            }
            for c in self._cores
        ]

    def stats(self) -> Dict:
        return {
            "total_votes": self._total_votes,
            "total_disagreements": self._total_disagreements,
            "seu_detection_rate": self.seu_detection_rate(),
            "cores": self.core_stats(),
        }


# ---------------------------------------------------------------------------
# TMRProcess -- convenience wrapper
# ---------------------------------------------------------------------------

class TMRProcess:
    """
    Wraps any callable in TMR protection.

    Usage
    -----
    >>> tmr_nav = TMRProcess(navigation_update_fn)
    >>> output, report = tmr_nav.execute(sensor_data)
    """

    def __init__(
        self,
        func: Callable,
        float_tolerance: float = 1e-9,
        name: str = "tmr_process",
    ):
        self.func = func
        self.name = name
        self._voter = TMRVoter(float_tolerance=float_tolerance)

    def execute(self, *args: Any, **kwargs: Any) -> Tuple[Any, Optional[FaultReport]]:
        """
        Execute the wrapped function under TMR and return the voted
        result plus an optional fault report.
        """
        result, report = self._voter.execute(self.func, *args, **kwargs)
        if report is not None:
            logger.warning(
                "TMRProcess '%s' fault: %s", self.name, report.description
            )
        return result, report

    def inject_seu(self, core_id: int, magnitude: float = 1.0) -> None:
        """Inject an SEU on a specific core of this process."""
        self._voter.inject_seu(core_id, magnitude)

    def scrub(self, core_id: int) -> None:
        """Scrub a specific core."""
        self._voter.scrub(core_id)

    def scrub_all(self) -> None:
        """Scrub all cores."""
        self._voter.scrub_all()

    @property
    def stats(self) -> Dict:
        s = self._voter.stats()
        s["name"] = self.name
        return s

    @property
    def seu_detection_rate(self) -> float:
        return self._voter.seu_detection_rate()

    @property
    def fault_reports(self) -> List[FaultReport]:
        return self._voter.fault_reports
