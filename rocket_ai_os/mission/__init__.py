"""
Mission Management Subsystem for Autonomous Rocket AI OS.

Implements a complete goal-plan-execute-monitor loop for autonomous
launch vehicle operations:

- **HTN Planner** -- Hierarchical Task Network decomposition of
  high-level goals into primitive actions.
- **Reactive Executive** -- T-REX-style task dispatcher with continuous
  monitoring and failure-triggered replanning.
- **UA Scheduler** -- Utility Accrual scheduling that maximises mission
  value under resource and deadline constraints.
- **GOAC** -- Goal-Oriented Autonomous Controller that integrates the
  planner, executive, and scheduler into a closed-loop architecture.

References:
    ESA GOAC Specification
    T-REX (Teleo-Reactive Executive) -- McGann et al.
    HTN Planning -- Erol, Hendler & Nau
"""

from .planner import (
    TaskStatus,
    PrimitiveTask,
    CompoundTask,
    HTNPlanner,
    PlanningError,
)

from .executive import (
    ExecutionStatus,
    TaskOutcome,
    ExecutionResult,
    FaultInfo,
    Executive,
)

from .scheduler import (
    CurveType,
    UtilityFunction,
    SchedulableTask,
    ScheduleEntry,
    UASScheduler,
)

from .goac import (
    GOACState,
    GoalStatus,
    Goal,
    WorldState,
    GOAC,
)

__all__ = [
    # Planner
    "TaskStatus",
    "PrimitiveTask",
    "CompoundTask",
    "HTNPlanner",
    "PlanningError",
    # Executive
    "ExecutionStatus",
    "TaskOutcome",
    "ExecutionResult",
    "FaultInfo",
    "Executive",
    # Scheduler
    "CurveType",
    "UtilityFunction",
    "SchedulableTask",
    "ScheduleEntry",
    "UASScheduler",
    # GOAC
    "GOACState",
    "GoalStatus",
    "Goal",
    "WorldState",
    "GOAC",
]
