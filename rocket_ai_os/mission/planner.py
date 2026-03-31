"""
Hierarchical Task Network (HTN) Planner for autonomous rocket mission management.

Decomposes high-level mission goals into ordered sequences of primitive
actions by recursively expanding compound tasks until only executable
primitives remain.  The planner maintains a task library of standard
launch-vehicle operations and supports re-planning when a primitive
task fails during execution.

References:
    Erol, Hendler & Nau -- "HTN Planning: Complexity and Expressivity" (1996)
    ESA GOAC Architecture -- "Goal Oriented Autonomous Controller" (2010)
    NASA Europa Clipper Autonomy -- JPL D-80307 (2019)
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from rocket_ai_os.config import MissionPhase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Lifecycle status of a planned task."""
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveTask:
    """
    An atomic, directly-executable action.

    Attributes:
        name:             Human-readable task identifier.
        preconditions:    World-state predicates that must hold before execution.
                          Keys are state variable names; values are the required
                          values (exact match or callable predicate).
        effects:          State changes applied after successful execution.
                          Keys are state variable names; values are new values.
        duration_estimate: Expected wall-clock duration in seconds.
        criticality:      Integer 1-5 where 1 is highest criticality.
        status:           Current lifecycle status.
        parameters:       Arbitrary execution parameters for downstream consumers.
    """
    name: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    duration_estimate: float = 1.0
    criticality: int = 3
    status: TaskStatus = TaskStatus.PENDING
    parameters: Dict[str, Any] = field(default_factory=dict)

    def check_preconditions(self, world_state: Dict[str, Any]) -> bool:
        """
        Return True if all preconditions are satisfied by *world_state*.

        A precondition value may be a plain value (compared with ``==``) or
        a callable that receives the current state value and returns bool.
        """
        for key, required in self.preconditions.items():
            current = world_state.get(key)
            if callable(required):
                if not required(current):
                    return False
            elif current != required:
                return False
        return True

    def apply_effects(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a new world-state dict with this task's effects applied.

        Original *world_state* is not mutated.
        """
        new_state = dict(world_state)
        for key, value in self.effects.items():
            if callable(value):
                new_state[key] = value(new_state.get(key))
            else:
                new_state[key] = value
        return new_state


@dataclass
class CompoundTask:
    """
    A non-primitive task that must be decomposed into subtasks.

    Attributes:
        name:                  Human-readable task identifier.
        subtasks:              Ordered list of child task names (primitive or
                               compound).
        decomposition_method:  Name of the decomposition strategy used.
        preconditions:         Conditions that must hold for this decomposition
                               to be applicable.
    """
    name: str
    subtasks: List[str] = field(default_factory=list)
    decomposition_method: str = "ordered"
    preconditions: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTN Planner
# ---------------------------------------------------------------------------

class HTNPlanner:
    """
    Hierarchical Task Network planner for launch-vehicle operations.

    The planner holds two registries:

    * **Primitive library** -- directly executable tasks such as
      *ignite_engine*, *open_valve*, *monitor_thrust*, etc.
    * **Compound library** -- higher-level tasks that decompose into
      ordered sequences of primitives or further compounds.

    Planning proceeds by recursively expanding compound tasks until the
    plan consists entirely of primitive tasks whose preconditions can be
    satisfied in sequence.

    Parameters:
        max_depth:      Maximum recursion depth during decomposition (safety
                        bound to prevent infinite loops).
        max_replan:     Maximum number of replanning attempts per failure.
    """

    def __init__(
        self,
        max_depth: int = 20,
        max_replan: int = 3,
    ) -> None:
        self._primitive_library: Dict[str, PrimitiveTask] = {}
        self._compound_library: Dict[str, CompoundTask] = {}
        self._max_depth = max_depth
        self._max_replan = max_replan
        self._replan_count: int = 0

        # Populate the standard task library for rocket operations
        self._build_default_library()

        logger.info(
            "HTNPlanner initialised  primitives=%d  compounds=%d  "
            "max_depth=%d  max_replan=%d",
            len(self._primitive_library),
            len(self._compound_library),
            max_depth,
            max_replan,
        )

    # ------------------------------------------------------------------
    # Library management
    # ------------------------------------------------------------------

    def register_primitive(self, task: PrimitiveTask) -> None:
        """Add or replace a primitive task in the library."""
        self._primitive_library[task.name] = task
        logger.debug("Registered primitive task: %s", task.name)

    def register_compound(self, task: CompoundTask) -> None:
        """Add or replace a compound task in the library."""
        self._compound_library[task.name] = task
        logger.debug("Registered compound task: %s", task.name)

    def get_primitive(self, name: str) -> Optional[PrimitiveTask]:
        """Look up a primitive task by name."""
        return self._primitive_library.get(name)

    def get_compound(self, name: str) -> Optional[CompoundTask]:
        """Look up a compound task by name."""
        return self._compound_library.get(name)

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        world_state: Dict[str, Any],
    ) -> List[PrimitiveTask]:
        """
        Produce an ordered list of primitive tasks that achieve *goal*
        starting from *world_state*.

        The planner first checks whether *goal* names a primitive task; if
        so it returns that single task (assuming preconditions are met).
        Otherwise it looks up the compound task, verifies its
        preconditions, and recursively decomposes each subtask.

        Args:
            goal:        Name of the task (primitive or compound) to plan.
            world_state: Current state of the world as a flat dict.

        Returns:
            Ordered list of :class:`PrimitiveTask` instances.

        Raises:
            PlanningError: If the goal cannot be decomposed or preconditions
                           fail at any level.
        """
        self._replan_count = 0
        logger.info("Planning for goal '%s'", goal)
        plan = self._decompose(goal, dict(world_state), depth=0)
        logger.info(
            "Plan for '%s' contains %d primitive tasks", goal, len(plan),
        )
        return plan

    def replan(
        self,
        failed_task: PrimitiveTask,
        world_state: Dict[str, Any],
    ) -> List[PrimitiveTask]:
        """
        Generate an alternative plan after *failed_task* has failed.

        The replanner attempts to find a different decomposition path that
        avoids the failed task or substitutes it with a safe alternative.
        If no alternative exists, an abort sequence is returned.

        Args:
            failed_task: The primitive task that failed during execution.
            world_state: World state at the time of failure.

        Returns:
            A replacement plan (list of primitive tasks).
        """
        self._replan_count += 1
        logger.warning(
            "Replanning after failure of '%s'  (attempt %d/%d)",
            failed_task.name,
            self._replan_count,
            self._max_replan,
        )

        if self._replan_count > self._max_replan:
            logger.error(
                "Maximum replan attempts (%d) exceeded -- generating abort",
                self._max_replan,
            )
            return self._generate_abort_sequence(world_state)

        # Strategy 1: look for a registered alternative for the failed task
        alternative_name = f"{failed_task.name}_alt"
        if alternative_name in self._primitive_library:
            alt = copy.deepcopy(self._primitive_library[alternative_name])
            if alt.check_preconditions(world_state):
                logger.info(
                    "Found alternative task '%s' for failed '%s'",
                    alternative_name,
                    failed_task.name,
                )
                alt.status = TaskStatus.PENDING
                return [alt]

        # Strategy 2: look for a compound recovery task
        recovery_name = f"recover_{failed_task.name}"
        if recovery_name in self._compound_library:
            logger.info(
                "Decomposing recovery compound '%s'", recovery_name,
            )
            return self._decompose(recovery_name, dict(world_state), depth=0)

        # Strategy 3: attempt a safe-mode sequence
        safe_mode = self._generate_safe_mode_sequence(world_state)
        if safe_mode:
            logger.info("Falling back to safe-mode sequence")
            return safe_mode

        # Last resort: abort
        logger.error("No recovery found for '%s' -- aborting", failed_task.name)
        return self._generate_abort_sequence(world_state)

    # ------------------------------------------------------------------
    # Decomposition engine
    # ------------------------------------------------------------------

    def _decompose(
        self,
        task_name: str,
        world_state: Dict[str, Any],
        depth: int,
    ) -> List[PrimitiveTask]:
        """
        Recursively decompose *task_name* into primitive tasks.

        Raises PlanningError on failure.
        """
        if depth > self._max_depth:
            raise PlanningError(
                f"Maximum decomposition depth ({self._max_depth}) exceeded "
                f"while expanding '{task_name}'"
            )

        # --- Primitive task? -------------------------------------------
        if task_name in self._primitive_library:
            task = copy.deepcopy(self._primitive_library[task_name])
            if not task.check_preconditions(world_state):
                raise PlanningError(
                    f"Preconditions for primitive '{task_name}' not met in "
                    f"current world state"
                )
            task.status = TaskStatus.PENDING
            return [task]

        # --- Compound task? --------------------------------------------
        if task_name in self._compound_library:
            compound = self._compound_library[task_name]

            if not self._check_compound_preconditions(compound, world_state):
                raise PlanningError(
                    f"Preconditions for compound '{task_name}' not met"
                )

            plan: List[PrimitiveTask] = []
            sim_state = dict(world_state)

            for subtask_name in compound.subtasks:
                sub_plan = self._decompose(subtask_name, sim_state, depth + 1)
                for task in sub_plan:
                    plan.append(task)
                    # Forward-simulate effects for precondition checking
                    sim_state = task.apply_effects(sim_state)

            return plan

        raise PlanningError(
            f"Task '{task_name}' not found in primitive or compound libraries"
        )

    @staticmethod
    def _check_compound_preconditions(
        compound: CompoundTask,
        world_state: Dict[str, Any],
    ) -> bool:
        """Check preconditions of a compound task against world state."""
        for key, required in compound.preconditions.items():
            current = world_state.get(key)
            if callable(required):
                if not required(current):
                    return False
            elif current != required:
                return False
        return True

    # ------------------------------------------------------------------
    # Recovery & abort sequences
    # ------------------------------------------------------------------

    def _generate_abort_sequence(
        self,
        world_state: Dict[str, Any],
    ) -> List[PrimitiveTask]:
        """
        Generate an emergency abort sequence appropriate for the current
        mission phase.
        """
        phase = world_state.get("phase")
        sequence: List[PrimitiveTask] = []

        # Engine shutdown is always first
        sequence.append(PrimitiveTask(
            name="emergency_engine_shutdown",
            preconditions={},
            effects={"engines_running": False, "thrust": 0.0},
            duration_estimate=0.5,
            criticality=1,
        ))

        if phase in (
            MissionPhase.LIFTOFF,
            MissionPhase.MAX_Q,
            MissionPhase.MECO,
        ):
            sequence.append(PrimitiveTask(
                name="activate_flight_termination",
                preconditions={},
                effects={"flight_terminated": True},
                duration_estimate=0.1,
                criticality=1,
            ))
        elif phase in (
            MissionPhase.LANDING_BURN,
            MissionPhase.TERMINAL_LANDING,
        ):
            sequence.append(PrimitiveTask(
                name="abort_landing_divert",
                preconditions={},
                effects={"landing_aborted": True},
                duration_estimate=2.0,
                criticality=1,
            ))

        # Mark abort phase
        sequence.append(PrimitiveTask(
            name="enter_abort_mode",
            preconditions={},
            effects={"phase": MissionPhase.ABORT},
            duration_estimate=0.0,
            criticality=1,
        ))

        return sequence

    def _generate_safe_mode_sequence(
        self,
        world_state: Dict[str, Any],
    ) -> List[PrimitiveTask]:
        """
        Generate a safe-mode sequence that puts the vehicle into a
        stable, low-risk configuration.
        """
        tasks: List[PrimitiveTask] = []

        if world_state.get("engines_running", False):
            tasks.append(PrimitiveTask(
                name="throttle_to_minimum",
                preconditions={"engines_running": True},
                effects={"thrust_level": 0.4},
                duration_estimate=0.5,
                criticality=1,
            ))

        tasks.append(PrimitiveTask(
            name="stabilise_attitude",
            preconditions={},
            effects={"attitude_stable": True},
            duration_estimate=2.0,
            criticality=1,
        ))

        tasks.append(PrimitiveTask(
            name="enable_telemetry_burst",
            preconditions={},
            effects={"telemetry_mode": "burst"},
            duration_estimate=0.2,
            criticality=2,
        ))

        return tasks

    # ------------------------------------------------------------------
    # Default task library
    # ------------------------------------------------------------------

    def _build_default_library(self) -> None:
        """Populate primitive and compound libraries with standard rocket ops."""

        # ---- Primitive tasks ------------------------------------------

        self.register_primitive(PrimitiveTask(
            name="open_fuel_valve",
            preconditions={"fuel_valve": "closed", "fuel_remaining": lambda f: f > 0},
            effects={"fuel_valve": "open"},
            duration_estimate=0.3,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="close_fuel_valve",
            preconditions={"fuel_valve": "open"},
            effects={"fuel_valve": "closed"},
            duration_estimate=0.3,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="open_oxidiser_valve",
            preconditions={
                "oxidiser_valve": "closed",
                "fuel_remaining": lambda f: f > 0,
            },
            effects={"oxidiser_valve": "open"},
            duration_estimate=0.3,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="close_oxidiser_valve",
            preconditions={"oxidiser_valve": "open"},
            effects={"oxidiser_valve": "closed"},
            duration_estimate=0.3,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="activate_igniter",
            preconditions={
                "fuel_valve": "open",
                "oxidiser_valve": "open",
                "igniter_armed": True,
            },
            effects={"igniter_fired": True},
            duration_estimate=0.2,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="arm_igniter",
            preconditions={"igniter_armed": False},
            effects={"igniter_armed": True},
            duration_estimate=0.5,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="confirm_ignition",
            preconditions={"igniter_fired": True},
            effects={"engines_running": True, "thrust_level": 1.0},
            duration_estimate=1.0,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="monitor_thrust",
            preconditions={"engines_running": True},
            effects={"thrust_nominal": True},
            duration_estimate=2.0,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="throttle_up",
            preconditions={"engines_running": True},
            effects={"thrust_level": 1.0},
            duration_estimate=1.0,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="throttle_down",
            preconditions={"engines_running": True},
            effects={"thrust_level": 0.4},
            duration_estimate=1.0,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="shutdown_engines",
            preconditions={"engines_running": True},
            effects={
                "engines_running": False,
                "thrust_level": 0.0,
                "thrust": 0.0,
            },
            duration_estimate=0.5,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="release_clamps",
            preconditions={"engines_running": True, "clamps_locked": True},
            effects={"clamps_locked": False},
            duration_estimate=0.1,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="initiate_stage_separation",
            preconditions={"engines_running": False, "stage_connected": True},
            effects={"stage_connected": False, "stage_separated": True},
            duration_estimate=1.0,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="deploy_landing_legs",
            preconditions={"landing_legs_deployed": False},
            effects={"landing_legs_deployed": True},
            duration_estimate=2.0,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="deploy_grid_fins",
            preconditions={"grid_fins_deployed": False},
            effects={"grid_fins_deployed": True},
            duration_estimate=1.5,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="activate_rcs",
            preconditions={"rcs_active": False},
            effects={"rcs_active": True},
            duration_estimate=0.5,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="start_entry_burn",
            preconditions={
                "engines_running": False,
                "fuel_remaining": lambda f: f > 0.05,
            },
            effects={"engines_running": True, "thrust_level": 0.7},
            duration_estimate=1.0,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="start_landing_burn",
            preconditions={
                "engines_running": False,
                "fuel_remaining": lambda f: f > 0.02,
                "landing_legs_deployed": True,
            },
            effects={"engines_running": True, "thrust_level": 0.6},
            duration_estimate=1.0,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="monitor_descent",
            preconditions={"engines_running": True},
            effects={"descent_monitored": True},
            duration_estimate=5.0,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="final_touchdown",
            preconditions={
                "engines_running": True,
                "landing_legs_deployed": True,
            },
            effects={
                "engines_running": False,
                "thrust_level": 0.0,
                "landed": True,
                "phase": MissionPhase.LANDED,
            },
            duration_estimate=2.0,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="safing_sequence",
            preconditions={"landed": True},
            effects={
                "fuel_valve": "closed",
                "oxidiser_valve": "closed",
                "vehicle_safed": True,
            },
            duration_estimate=5.0,
            criticality=2,
        ))

        # Powered descent primitive tasks
        self.register_primitive(PrimitiveTask(
            name="powered_descent_init",
            preconditions={
                "phase": MissionPhase.ENTRY_BURN,
                "engines_running": False,
                "fuel_remaining": lambda f: f > 0.1,
            },
            effects={"phase": MissionPhase.LANDING_BURN},
            duration_estimate=1.0,
            criticality=1,
        ))

        self.register_primitive(PrimitiveTask(
            name="powered_descent_guide",
            preconditions={
                "phase": MissionPhase.LANDING_BURN,
                "engines_running": True,
                "guidance_active": False,
            },
            effects={"guidance_active": True},
            duration_estimate=0.5,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="powered_descent_execute",
            preconditions={
                "phase": MissionPhase.LANDING_BURN,
                "engines_running": True,
                "guidance_active": True,
                "altitude_above_landing": lambda h: h > 5.0,
            },
            effects={"descent_progress": lambda p: min(p + 0.1, 1.0)},
            duration_estimate=2.0,
            criticality=2,
        ))

        self.register_primitive(PrimitiveTask(
            name="powered_descent_complete",
            preconditions={
                "phase": MissionPhase.LANDING_BURN,
                "engines_running": True,
                "altitude_above_landing": lambda h: h <= 2.0,
                "velocity_magnitude": lambda v: v < 2.0,
            },
            effects={
                "phase": MissionPhase.TERMINAL_LANDING,
                "guidance_active": False,
            },
            duration_estimate=1.0,
            criticality=1,
        ))

        # ---- Compound tasks -------------------------------------------

        self.register_compound(CompoundTask(
            name="ignite_engine",
            subtasks=[
                "arm_igniter",
                "open_fuel_valve",
                "open_oxidiser_valve",
                "activate_igniter",
                "confirm_ignition",
            ],
            decomposition_method="ordered",
            preconditions={"fuel_remaining": lambda f: f > 0},
        ))

        self.register_compound(CompoundTask(
            name="liftoff_sequence",
            subtasks=[
                "ignite_engine",
                "monitor_thrust",
                "release_clamps",
            ],
            decomposition_method="ordered",
            preconditions={
                "phase": MissionPhase.PRE_LAUNCH,
                "clamps_locked": True,
            },
        ))

        self.register_compound(CompoundTask(
            name="stage_separation_sequence",
            subtasks=[
                "shutdown_engines",
                "initiate_stage_separation",
            ],
            decomposition_method="ordered",
            preconditions={"stage_connected": True},
        ))

        self.register_compound(CompoundTask(
            name="boostback_and_entry",
            subtasks=[
                "activate_rcs",
                "deploy_grid_fins",
                "start_entry_burn",
                "monitor_thrust",
                "shutdown_engines",
            ],
            decomposition_method="ordered",
            preconditions={
                "stage_separated": True,
                "fuel_remaining": lambda f: f > 0.05,
            },
        ))

        self.register_compound(CompoundTask(
            name="landing_sequence",
            subtasks=[
                "deploy_landing_legs",
                "start_landing_burn",
                "monitor_descent",
                "final_touchdown",
                "safing_sequence",
            ],
            decomposition_method="ordered",
            preconditions={
                "fuel_remaining": lambda f: f > 0.02,
            },
        ))

        self.register_compound(CompoundTask(
            name="full_first_stage_recovery",
            subtasks=[
                "boostback_and_entry",
                "landing_sequence",
            ],
            decomposition_method="ordered",
            preconditions={
                "stage_separated": True,
                "fuel_remaining": lambda f: f > 0.05,
            },
        ))

        # Powered descent compound task
        self.register_compound(CompoundTask(
            name="powered_descent",
            subtasks=[
                "powered_descent_init",
                "powered_descent_guide",
                "powered_descent_execute",
                "powered_descent_complete",
            ],
            decomposition_method="ordered",
            preconditions={
                "fuel_remaining": lambda f: f > 0.1,
            },
        ))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_primitives(self) -> List[str]:
        """Return names of all registered primitive tasks."""
        return sorted(self._primitive_library.keys())

    def list_compounds(self) -> List[str]:
        """Return names of all registered compound tasks."""
        return sorted(self._compound_library.keys())

    def estimate_plan_duration(self, plan: List[PrimitiveTask]) -> float:
        """Sum the duration estimates of every task in *plan*."""
        return sum(t.duration_estimate for t in plan)

    def __repr__(self) -> str:
        return (
            f"<HTNPlanner primitives={len(self._primitive_library)} "
            f"compounds={len(self._compound_library)}>"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PlanningError(Exception):
    """Raised when the planner cannot produce a valid plan."""
