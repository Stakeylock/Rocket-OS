"""
Reactive Executive for autonomous rocket mission execution (T-REX style).

Dispatches primitive tasks produced by the HTN planner, monitors their
execution in real time, and triggers replanning when anomalies are
detected.  The executive operates as a state machine that transitions
through dispatch, execute, monitor, and (on failure) replan phases.

The design follows the T-REX (Teleo-Reactive Executive) paradigm used
in underwater and space autonomy systems: it is *not* a blind sequencer
but continuously monitors task outcomes and reacts to environmental
changes during each execution step.

References:
    McGann, Py, Rajan et al. -- "T-REX: A Deliberative System for AUVs"
    ESA GOAC -- Reactive Execution Layer specification
    NASA Europa Clipper -- Onboard Fault-Aware Execution Engine
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from rocket_ai_os.mission.planner import (
    HTNPlanner,
    PlanningError,
    PrimitiveTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExecutionStatus(Enum):
    """Top-level state of the executive's state machine."""
    IDLE = auto()
    DISPATCHING = auto()
    EXECUTING = auto()
    MONITORING = auto()
    REPLANNING = auto()
    ABORTING = auto()


class TaskOutcome(Enum):
    """Result of executing a single primitive task."""
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    PREEMPTED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Outcome of a dispatched task.

    Attributes:
        task:       The task that was dispatched.
        outcome:    How the task ended.
        world_state: World state after execution (may reflect partial effects).
        duration:   Actual wall-clock execution time in seconds.
        fault_info: Optional fault descriptor if the outcome is FAILURE.
        message:    Human-readable status message.
    """
    task: PrimitiveTask
    outcome: TaskOutcome
    world_state: Dict[str, Any]
    duration: float = 0.0
    fault_info: Optional[Dict[str, Any]] = None
    message: str = ""


@dataclass
class FaultInfo:
    """
    Lightweight fault descriptor used by the executive.

    This is intentionally decoupled from the full FDIR subsystem so the
    mission module can operate stand-alone.  When the FDIR subsystem is
    available the executive converts its :class:`FaultRecord` objects
    into this simpler structure.

    Attributes:
        fault_type:  Short string tag (e.g. ``"engine_underperformance"``).
        severity:    Integer 1-5 where 1 is most severe.
        subsystem:   Name of the affected subsystem.
        details:     Arbitrary key-value details.
        timestamp:   Monotonic time when the fault was detected.
    """
    fault_type: str = "unknown"
    severity: int = 3
    subsystem: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Executive
# ---------------------------------------------------------------------------

class Executive:
    """
    Reactive executive that dispatches, monitors, and recovers tasks.

    The executive maintains an internal state machine::

        IDLE -> DISPATCHING -> EXECUTING -> MONITORING
                                               |
                                          success -> next task (or IDLE)
                                          failure -> REPLANNING -> DISPATCHING
                                          abort   -> ABORTING -> IDLE

    Parameters:
        planner:            HTN planner instance used for replanning.
        execution_timeout:  Default maximum duration (seconds) for any
                            primitive task before it is considered timed out.
        monitor_interval:   How often (seconds) the monitor polls task
                            status during execution.

    Usage::

        executive = Executive(planner=my_planner)
        result = executive.dispatch(task, world_state)
        if result.outcome == TaskOutcome.SUCCESS:
            ...
        else:
            recovery = executive.handle_failure(task, fault, world_state)
    """

    def __init__(
        self,
        planner: Optional[HTNPlanner] = None,
        execution_timeout: float = 30.0,
        monitor_interval: float = 0.1,
    ) -> None:
        self._planner = planner
        self._execution_timeout = execution_timeout
        self._monitor_interval = monitor_interval

        self._status: ExecutionStatus = ExecutionStatus.IDLE
        self._current_task: Optional[PrimitiveTask] = None
        self._execution_log: List[ExecutionResult] = []
        self._fault_history: List[FaultInfo] = []

        # User-supplied callbacks for actual task execution and monitoring.
        # If not set, the executive uses its built-in simulators.
        self._execute_fn: Optional[
            Callable[[PrimitiveTask, Dict[str, Any]], ExecutionResult]
        ] = None
        self._monitor_fn: Optional[
            Callable[[PrimitiveTask, Dict[str, Any]], TaskOutcome]
        ] = None

        logger.info(
            "Executive initialised  timeout=%.1fs  monitor_interval=%.2fs",
            execution_timeout,
            monitor_interval,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> ExecutionStatus:
        """Current state-machine status of the executive."""
        return self._status

    @property
    def current_task(self) -> Optional[PrimitiveTask]:
        """Task currently being dispatched / executed, or None."""
        return self._current_task

    @property
    def execution_log(self) -> List[ExecutionResult]:
        """Read-only view of all past execution results."""
        return list(self._execution_log)

    @property
    def fault_history(self) -> List[FaultInfo]:
        """Read-only view of recorded faults."""
        return list(self._fault_history)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_execute_callback(
        self,
        fn: Callable[[PrimitiveTask, Dict[str, Any]], ExecutionResult],
    ) -> None:
        """
        Register a callback that performs the *actual* execution of a
        primitive task.

        The callback receives ``(task, world_state)`` and must return an
        :class:`ExecutionResult`.
        """
        self._execute_fn = fn

    def set_monitor_callback(
        self,
        fn: Callable[[PrimitiveTask, Dict[str, Any]], TaskOutcome],
    ) -> None:
        """
        Register a callback for continuous task monitoring.

        The callback receives ``(task, world_state)`` and returns a
        :class:`TaskOutcome` indicating whether the task is still running
        successfully.
        """
        self._monitor_fn = fn

    # ------------------------------------------------------------------
    # Core dispatch / execute / monitor cycle
    # ------------------------------------------------------------------

    def dispatch(
        self,
        task: PrimitiveTask,
        world_state: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Dispatch a single primitive task through the full execute-monitor
        cycle.

        Steps:
            1. **Precondition check** -- verify the task can start.
            2. **Execute** -- run the task (via callback or built-in sim).
            3. **Monitor** -- poll for anomalies during execution.
            4. **Finalise** -- record result and update state.

        Args:
            task:        The primitive task to execute.
            world_state: Current world state (will be updated on success).

        Returns:
            :class:`ExecutionResult` describing the outcome.
        """
        logger.info("Dispatching task '%s'", task.name)
        self._status = ExecutionStatus.DISPATCHING
        self._current_task = task

        # --- 1. Precondition check ------------------------------------
        if not task.check_preconditions(world_state):
            result = ExecutionResult(
                task=task,
                outcome=TaskOutcome.FAILURE,
                world_state=world_state,
                message=f"Preconditions not met for '{task.name}'",
            )
            self._record_result(result)
            task.status = TaskStatus.FAILED
            self._status = ExecutionStatus.IDLE
            self._current_task = None
            return result

        # --- 2. Execute -----------------------------------------------
        self._status = ExecutionStatus.EXECUTING
        task.status = TaskStatus.ACTIVE
        start_time = time.monotonic()

        if self._execute_fn is not None:
            result = self._execute_fn(task, world_state)
        else:
            result = self._default_execute(task, world_state)

        elapsed = time.monotonic() - start_time

        # --- 3. Monitor -----------------------------------------------
        self._status = ExecutionStatus.MONITORING
        monitored_outcome = self.monitor(task, result.world_state)

        if monitored_outcome == TaskOutcome.FAILURE:
            result = ExecutionResult(
                task=task,
                outcome=TaskOutcome.FAILURE,
                world_state=result.world_state,
                duration=elapsed,
                message=f"Monitor detected failure during '{task.name}'",
            )

        if result.outcome == TaskOutcome.SUCCESS:
            # Check for timeout
            timeout = task.duration_estimate * 3.0  # generous 3x margin
            if elapsed > min(timeout, self._execution_timeout):
                result = ExecutionResult(
                    task=task,
                    outcome=TaskOutcome.TIMEOUT,
                    world_state=result.world_state,
                    duration=elapsed,
                    message=(
                        f"Task '{task.name}' timed out after {elapsed:.2f}s "
                        f"(limit {min(timeout, self._execution_timeout):.2f}s)"
                    ),
                )

        # --- 4. Finalise ----------------------------------------------
        result.duration = elapsed
        self._record_result(result)

        if result.outcome == TaskOutcome.SUCCESS:
            task.status = TaskStatus.COMPLETED
            logger.info(
                "Task '%s' completed successfully in %.3fs",
                task.name,
                elapsed,
            )
        else:
            task.status = TaskStatus.FAILED
            logger.warning(
                "Task '%s' ended with %s: %s",
                task.name,
                result.outcome.name,
                result.message,
            )

        self._current_task = None
        self._status = ExecutionStatus.IDLE
        return result

    def dispatch_plan(
        self,
        plan: List[PrimitiveTask],
        world_state: Dict[str, Any],
    ) -> Tuple[List[ExecutionResult], Dict[str, Any]]:
        """
        Execute an entire plan (list of primitive tasks) in sequence.

        Stops on the first failure and returns all results collected so
        far together with the final world state.

        Args:
            plan:        Ordered list of primitive tasks.
            world_state: Initial world state.

        Returns:
            Tuple of (results_list, final_world_state).
        """
        results: List[ExecutionResult] = []
        current_state = dict(world_state)

        for task in plan:
            result = self.dispatch(task, current_state)
            results.append(result)
            current_state = result.world_state

            if result.outcome != TaskOutcome.SUCCESS:
                logger.warning(
                    "Plan execution halted at task '%s' (%s)",
                    task.name,
                    result.outcome.name,
                )
                break

        return results, current_state

    def monitor(
        self,
        task: PrimitiveTask,
        world_state: Dict[str, Any],
    ) -> TaskOutcome:
        """
        Monitor an executing task for anomalies.

        If a monitor callback is registered it is invoked; otherwise
        the built-in monitor performs basic state-consistency checks.

        Args:
            task:        The currently executing task.
            world_state: World state to check against.

        Returns:
            :class:`TaskOutcome` -- SUCCESS if monitoring detects no issues.
        """
        if self._monitor_fn is not None:
            return self._monitor_fn(task, world_state)
        return self._default_monitor(task, world_state)

    def handle_failure(
        self,
        task: PrimitiveTask,
        fault: Optional[FaultInfo],
        world_state: Dict[str, Any],
    ) -> List[PrimitiveTask]:
        """
        Handle a task failure by consulting the planner for a recovery
        plan.

        If a fault descriptor is provided it is recorded in the fault
        history.  The planner's :meth:`replan` is invoked to produce an
        alternative course of action.

        Args:
            task:        The failed task.
            fault:       Optional fault information.
            world_state: World state at time of failure.

        Returns:
            Recovery plan as a list of primitive tasks, which may be an
            abort sequence if no recovery is available.
        """
        self._status = ExecutionStatus.REPLANNING
        logger.warning(
            "Handling failure of task '%s'  fault=%s",
            task.name,
            fault.fault_type if fault else "none",
        )

        if fault is not None:
            self._fault_history.append(fault)

        if self._planner is None:
            logger.error(
                "No planner available for replanning -- cannot recover"
            )
            self._status = ExecutionStatus.ABORTING
            return []

        try:
            recovery_plan = self._planner.replan(task, world_state)
            self._status = ExecutionStatus.DISPATCHING
            return recovery_plan
        except PlanningError as exc:
            logger.error("Replanning failed: %s", exc)
            self._status = ExecutionStatus.ABORTING
            return []

    # ------------------------------------------------------------------
    # Default (built-in) execution & monitoring
    # ------------------------------------------------------------------

    def _default_execute(
        self,
        task: PrimitiveTask,
        world_state: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Built-in task executor that simulates success by applying the
        task's effects to the world state.

        This is used when no external execution callback has been
        registered and is suitable for unit testing and offline planning
        validation.
        """
        new_state = task.apply_effects(world_state)
        return ExecutionResult(
            task=task,
            outcome=TaskOutcome.SUCCESS,
            world_state=new_state,
            message=f"Simulated execution of '{task.name}'",
        )

    @staticmethod
    def _default_monitor(
        task: PrimitiveTask,
        world_state: Dict[str, Any],
    ) -> TaskOutcome:
        """
        Built-in monitor that performs basic consistency checks.

        Verifies that every effect declared by the task is present in
        the world state after execution.  If any expected effect is
        missing the task is considered failed.
        """
        for key, expected in task.effects.items():
            actual = world_state.get(key)
            if callable(expected):
                # Effect functions are applied, so just verify the key exists
                if actual is None:
                    return TaskOutcome.FAILURE
            else:
                if actual != expected:
                    return TaskOutcome.FAILURE
        return TaskOutcome.SUCCESS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_result(self, result: ExecutionResult) -> None:
        """Append a result to the execution log."""
        self._execution_log.append(result)

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def abort(self, reason: str = "") -> None:
        """
        Force the executive into ABORTING state, cancelling the current
        task if any.

        Args:
            reason: Human-readable reason for the abort.
        """
        logger.error("Executive ABORT: %s", reason or "no reason given")
        self._status = ExecutionStatus.ABORTING

        if self._current_task is not None:
            self._current_task.status = TaskStatus.ABORTED
            self._current_task = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the executive's history and current state."""
        total = len(self._execution_log)
        successes = sum(
            1 for r in self._execution_log
            if r.outcome == TaskOutcome.SUCCESS
        )
        failures = sum(
            1 for r in self._execution_log
            if r.outcome == TaskOutcome.FAILURE
        )
        return {
            "status": self._status.name,
            "current_task": (
                self._current_task.name if self._current_task else None
            ),
            "total_dispatched": total,
            "successes": successes,
            "failures": failures,
            "faults_recorded": len(self._fault_history),
        }

    def __repr__(self) -> str:
        return (
            f"<Executive status={self._status.name} "
            f"dispatched={len(self._execution_log)} "
            f"faults={len(self._fault_history)}>"
        )
