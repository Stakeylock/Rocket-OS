"""
Utility Accrual (UA) Scheduling for autonomous rocket mission management.

Implements a scheduler that maximises the total *accrued utility* of
executed tasks under resource and deadline constraints.  Each task carries
a time-dependent utility function that captures how valuable the task is
at different points in time (e.g., a communication window that closes at
a deadline, or a sensor calibration whose value decays linearly).

The scheduler supports:
* Multiple utility curve shapes: step, linear, exponential, deadline.
* Resource-aware scheduling with multi-dimensional capacity vectors.
* Priority-based preemption for emergency tasks (e.g., solar-flare
  response, collision avoidance).
* Dynamic rescheduling as the mission evolves.

References:
    Jensen, Locke & Tokuda -- "A Time-Driven Scheduling Model for
        Real-Time Systems" (1985)
    Burns & Wellings -- "Utility Accrual Real-Time Scheduling" (2009)
    ESA GOAC -- Resource allocation layer specification
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility curves
# ---------------------------------------------------------------------------

class CurveType(Enum):
    """Supported utility-function curve shapes."""
    STEP = auto()       # Full value until deadline, then zero
    LINEAR = auto()     # Linearly decreasing value as deadline approaches
    EXPONENTIAL = auto()  # Exponentially decaying value
    DEADLINE = auto()   # Binary: value only at or before deadline


class UtilityFunction:
    """
    A time-dependent utility function for a schedulable task.

    The function maps the current time to a scalar utility value that
    represents how valuable it is to execute the owning task *right now*.

    Parameters:
        curve_type:   Shape of the utility curve.
        base_value:   Maximum utility value (at the optimal time).
        deadline:     Absolute time after which utility drops to zero.
        release_time: Earliest time the task may begin execution.
        decay_rate:   Rate parameter for exponential decay (only used
                      when *curve_type* is ``EXPONENTIAL``).

    Usage::

        uf = UtilityFunction(CurveType.LINEAR, base_value=100.0, deadline=50.0)
        print(uf.evaluate(current_time=25.0))  # -> 50.0
    """

    def __init__(
        self,
        curve_type: CurveType = CurveType.STEP,
        base_value: float = 1.0,
        deadline: float = float("inf"),
        release_time: float = 0.0,
        decay_rate: float = 0.1,
    ) -> None:
        self.curve_type = curve_type
        self.base_value = base_value
        self.deadline = deadline
        self.release_time = release_time
        self.decay_rate = decay_rate

    def evaluate(self, current_time: float) -> float:
        """
        Compute the utility of executing the task at *current_time*.

        Returns:
            Non-negative utility value.  Returns 0.0 if the task is past
            its deadline or before its release time.
        """
        if current_time < self.release_time:
            return 0.0
        if current_time > self.deadline:
            return 0.0

        if self.curve_type == CurveType.STEP:
            return self.base_value

        elif self.curve_type == CurveType.LINEAR:
            if self.deadline == float("inf"):
                return self.base_value
            span = self.deadline - self.release_time
            if span <= 0:
                return self.base_value
            remaining = self.deadline - current_time
            return self.base_value * (remaining / span)

        elif self.curve_type == CurveType.EXPONENTIAL:
            elapsed = current_time - self.release_time
            return self.base_value * math.exp(-self.decay_rate * elapsed)

        elif self.curve_type == CurveType.DEADLINE:
            # Binary: full value only if we can still meet the deadline
            return self.base_value

        # Fallback
        return 0.0

    def __repr__(self) -> str:
        return (
            f"<UtilityFunction curve={self.curve_type.name} "
            f"base={self.base_value} deadline={self.deadline}>"
        )


# ---------------------------------------------------------------------------
# Schedulable task
# ---------------------------------------------------------------------------

@dataclass
class SchedulableTask:
    """
    A task that can be scheduled by the UA scheduler.

    Attributes:
        name:                  Unique identifier for the task.
        utility_function:      Time-dependent utility function.
        resource_requirements: Dict mapping resource name -> amount required.
                               E.g. ``{"cpu": 0.3, "memory_kb": 512}``.
        deadline:              Absolute time by which the task must complete.
        duration:              Estimated execution duration in seconds.
        priority:              Static priority (lower = higher priority).
                               Used for tie-breaking and preemption.
        preemptible:           Whether this task can be preempted by a
                               higher-priority task.
        emergency:             If True the task receives unconditional
                               priority and may preempt any running task.
        metadata:              Arbitrary metadata for downstream consumers.
    """
    name: str
    utility_function: UtilityFunction = field(
        default_factory=UtilityFunction,
    )
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    deadline: float = float("inf")
    duration: float = 1.0
    priority: int = 5
    preemptible: bool = True
    emergency: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def utility_density(self) -> float:
        """
        Utility density: utility per unit of total resource demand.

        If no resources are required the density equals the base utility
        to avoid division by zero.
        """
        total_resource = sum(self.resource_requirements.values()) or 1.0
        return self.utility_function.base_value / total_resource


@dataclass
class ScheduleEntry:
    """
    A single entry in the computed schedule.

    Attributes:
        task:        The scheduled task.
        start_time:  Planned absolute start time.
        end_time:    Planned absolute end time.
        utility:     Utility accrued by this entry (evaluated at start_time).
    """
    task: SchedulableTask
    start_time: float
    end_time: float
    utility: float = 0.0


# ---------------------------------------------------------------------------
# UAS Scheduler
# ---------------------------------------------------------------------------

class UASScheduler:
    """
    Utility Accrual Scheduler for mission task scheduling.

    The scheduler constructs an ordered schedule that maximises the total
    accrued utility subject to resource capacity and deadline constraints.
    It supports dynamic rescheduling when emergency tasks arrive.

    Algorithm overview:
        1. Compute the current utility and utility density for every
           candidate task.
        2. Filter out tasks whose deadlines have passed or whose
           resource requirements exceed capacity.
        3. Sort candidates by a composite key: ``(-emergency, priority,
           -utility_density)`` so that emergency tasks always come first,
           followed by high-priority, high-value tasks.
        4. Greedily allocate tasks to time slots, committing resources
           and advancing the clock.
        5. When an emergency task arrives via :meth:`reschedule`, preempt
           the lowest-utility preemptible task to free resources.

    Parameters:
        default_resources: Default resource capacities (can be overridden
                           per call to :meth:`schedule`).

    Usage::

        scheduler = UASScheduler(default_resources={"cpu": 1.0, "mem": 4096})
        schedule = scheduler.schedule(tasks, current_time=0.0)
        for entry in schedule:
            print(entry.task.name, entry.start_time, entry.utility)
    """

    def __init__(
        self,
        default_resources: Optional[Dict[str, float]] = None,
    ) -> None:
        self._default_resources: Dict[str, float] = dict(
            default_resources or {"cpu": 1.0, "memory_kb": 8192}
        )
        self._current_schedule: List[ScheduleEntry] = []
        self._preempted_tasks: List[SchedulableTask] = []
        self._total_utility: float = 0.0

        logger.info(
            "UASScheduler initialised  default_resources=%s",
            self._default_resources,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_schedule(self) -> List[ScheduleEntry]:
        """Return a copy of the current schedule."""
        return list(self._current_schedule)

    @property
    def total_utility(self) -> float:
        """Total utility accrued by the current schedule."""
        return self._total_utility

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        tasks: List[SchedulableTask],
        available_resources: Optional[Dict[str, float]] = None,
        current_time: float = 0.0,
    ) -> List[ScheduleEntry]:
        """
        Build a schedule that maximises total accrued utility.

        Args:
            tasks:               Candidate tasks to schedule.
            available_resources: Resource capacity dict.  If *None* the
                                 scheduler's *default_resources* are used.
            current_time:        Absolute time at which scheduling begins.

        Returns:
            Ordered list of :class:`ScheduleEntry` instances.
        """
        resources = dict(available_resources or self._default_resources)
        logger.info(
            "Scheduling %d tasks at t=%.2f  resources=%s",
            len(tasks),
            current_time,
            resources,
        )

        # Evaluate and filter candidates
        candidates = self._evaluate_candidates(tasks, resources, current_time)

        # Sort by composite key
        candidates.sort(
            key=lambda t: (
                not t.emergency,     # emergencies first
                t.priority,          # lower number = higher priority
                -self._compute_utility_density_at(t, current_time),
            ),
        )

        # Greedy allocation
        schedule: List[ScheduleEntry] = []
        clock = current_time
        remaining = dict(resources)

        for task in candidates:
            # Check deadline feasibility
            if clock + task.duration > task.deadline:
                logger.debug(
                    "Skipping '%s': cannot finish before deadline "
                    "(clock=%.2f  dur=%.2f  dl=%.2f)",
                    task.name,
                    clock,
                    task.duration,
                    task.deadline,
                )
                continue

            # Check resource feasibility
            if not self._resources_available(task, remaining):
                logger.debug(
                    "Skipping '%s': insufficient resources", task.name,
                )
                continue

            # Allocate
            utility = task.utility_function.evaluate(clock)
            entry = ScheduleEntry(
                task=task,
                start_time=clock,
                end_time=clock + task.duration,
                utility=utility,
            )
            schedule.append(entry)

            # Commit resources for the duration of the task and advance clock
            self._commit_resources(task, remaining)
            clock += task.duration
            # Release resources after completion so they are available to
            # subsequent tasks.
            self._release_resources(task, remaining)

        self._current_schedule = schedule
        self._total_utility = sum(e.utility for e in schedule)

        logger.info(
            "Schedule built: %d entries  total_utility=%.2f",
            len(schedule),
            self._total_utility,
        )
        return schedule

    def reschedule(
        self,
        emergency_task: SchedulableTask,
        current_time: float = 0.0,
    ) -> List[ScheduleEntry]:
        """
        Insert an emergency task into the current schedule, preempting
        lower-utility tasks if necessary.

        Args:
            emergency_task: The emergency task to insert.
            current_time:   Absolute time now.

        Returns:
            Updated schedule with the emergency task inserted.
        """
        logger.warning(
            "RESCHEDULE: inserting emergency task '%s' at t=%.2f",
            emergency_task.name,
            current_time,
        )

        emergency_task.emergency = True

        # Collect all tasks from the existing schedule that haven't started
        remaining_tasks: List[SchedulableTask] = []
        preserved: List[ScheduleEntry] = []

        for entry in self._current_schedule:
            if entry.start_time < current_time:
                # Already started or completed -- keep as-is
                preserved.append(entry)
            else:
                remaining_tasks.append(entry.task)

        # If resources are tight, preempt lowest-utility preemptible tasks
        # until there is room for the emergency.
        remaining_tasks = self._preempt_for_emergency(
            emergency_task, remaining_tasks,
        )

        # Insert emergency task at front of remaining candidates
        all_candidates = [emergency_task] + remaining_tasks

        # Schedule from current_time
        new_schedule_tail = self.schedule(
            all_candidates,
            available_resources=self._default_resources,
            current_time=current_time,
        )

        # Merge preserved entries with new tail
        full_schedule = preserved + new_schedule_tail
        self._current_schedule = full_schedule
        self._total_utility = sum(e.utility for e in full_schedule)

        logger.info(
            "Reschedule complete: %d entries  total_utility=%.2f",
            len(full_schedule),
            self._total_utility,
        )
        return full_schedule

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self,
        tasks: List[SchedulableTask],
        resources: Dict[str, float],
        current_time: float,
    ) -> List[SchedulableTask]:
        """
        Filter out tasks that are past their deadline or have zero
        utility at *current_time*.
        """
        viable: List[SchedulableTask] = []
        for task in tasks:
            utility = task.utility_function.evaluate(current_time)
            if utility <= 0:
                logger.debug(
                    "Filtering '%s': zero utility at t=%.2f",
                    task.name,
                    current_time,
                )
                continue
            if current_time > task.deadline:
                logger.debug(
                    "Filtering '%s': past deadline (dl=%.2f  now=%.2f)",
                    task.name,
                    task.deadline,
                    current_time,
                )
                continue
            viable.append(task)
        return viable

    def _compute_utility_density_at(
        self,
        task: SchedulableTask,
        current_time: float,
    ) -> float:
        """
        Compute time-aware utility density: current utility divided by
        total resource demand.
        """
        utility = task.utility_function.evaluate(current_time)
        total_resource = sum(task.resource_requirements.values()) or 1.0
        return utility / total_resource

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    @staticmethod
    def _resources_available(
        task: SchedulableTask,
        remaining: Dict[str, float],
    ) -> bool:
        """Check whether *remaining* can satisfy *task*'s requirements."""
        for resource, required in task.resource_requirements.items():
            if remaining.get(resource, 0.0) < required:
                return False
        return True

    @staticmethod
    def _commit_resources(
        task: SchedulableTask,
        remaining: Dict[str, float],
    ) -> None:
        """Deduct *task*'s resource needs from *remaining*."""
        for resource, required in task.resource_requirements.items():
            remaining[resource] = remaining.get(resource, 0.0) - required

    @staticmethod
    def _release_resources(
        task: SchedulableTask,
        remaining: Dict[str, float],
    ) -> None:
        """Return *task*'s resources to the *remaining* pool."""
        for resource, required in task.resource_requirements.items():
            remaining[resource] = remaining.get(resource, 0.0) + required

    # ------------------------------------------------------------------
    # Preemption logic
    # ------------------------------------------------------------------

    def _preempt_for_emergency(
        self,
        emergency: SchedulableTask,
        candidates: List[SchedulableTask],
    ) -> List[SchedulableTask]:
        """
        Remove the lowest-utility preemptible tasks from *candidates*
        until there is enough room (resource-wise) for *emergency*.

        Preempted tasks are stored in ``self._preempted_tasks``.

        Returns:
            The filtered candidate list.
        """
        # Check if emergency can fit without preemption
        total_available = dict(self._default_resources)
        if self._resources_available(emergency, total_available):
            return candidates

        # Sort candidates by utility (ascending) so we preempt least
        # valuable first
        sorted_candidates = sorted(
            candidates,
            key=lambda t: t.utility_function.base_value,
        )

        freed: Dict[str, float] = {}
        kept: List[SchedulableTask] = []
        preempted: List[SchedulableTask] = []
        enough = False

        for task in sorted_candidates:
            if enough or not task.preemptible:
                kept.append(task)
                continue

            # Preempt this task
            preempted.append(task)
            for res, val in task.resource_requirements.items():
                freed[res] = freed.get(res, 0.0) + val

            # Check if we've freed enough
            enough = all(
                freed.get(res, 0.0) >= req
                for res, req in emergency.resource_requirements.items()
            )

        self._preempted_tasks.extend(preempted)
        if preempted:
            logger.info(
                "Preempted %d tasks for emergency '%s': %s",
                len(preempted),
                emergency.name,
                [t.name for t in preempted],
            )

        return kept

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_schedule_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the current schedule."""
        return {
            "num_entries": len(self._current_schedule),
            "total_utility": self._total_utility,
            "preempted_count": len(self._preempted_tasks),
            "entries": [
                {
                    "task": e.task.name,
                    "start": e.start_time,
                    "end": e.end_time,
                    "utility": e.utility,
                }
                for e in self._current_schedule
            ],
        }

    def __repr__(self) -> str:
        return (
            f"<UASScheduler entries={len(self._current_schedule)} "
            f"utility={self._total_utility:.2f}>"
        )
