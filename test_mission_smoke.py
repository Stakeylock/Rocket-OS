"""Quick smoke test for the mission management subsystem."""

def test_planner():
    from rocket_ai_os.mission.planner import (
        HTNPlanner, PrimitiveTask, CompoundTask, TaskStatus, PlanningError,
    )
    from rocket_ai_os.config import MissionPhase

    p = HTNPlanner()
    print(repr(p))
    print("Primitives:", p.list_primitives())
    print("Compounds:", p.list_compounds())

    # Test planning: ignite_engine compound task
    world = {
        "fuel_remaining": 0.9,
        "fuel_valve": "closed",
        "oxidiser_valve": "closed",
        "igniter_armed": False,
        "igniter_fired": False,
        "engines_running": False,
    }
    plan = p.plan("ignite_engine", world)
    print(f"\nPlan for 'ignite_engine': {[t.name for t in plan]}")
    print(f"Estimated duration: {p.estimate_plan_duration(plan):.1f}s")
    print("planner: PASS")

def test_executive():
    from rocket_ai_os.mission.executive import (
        Executive, ExecutionStatus, TaskOutcome, ExecutionResult, FaultInfo,
    )
    from rocket_ai_os.mission.planner import HTNPlanner, PrimitiveTask

    planner = HTNPlanner()
    ex = Executive(planner=planner)
    print(repr(ex))

    task = PrimitiveTask(
        name="test_task",
        preconditions={"ready": True},
        effects={"done": True},
        duration_estimate=1.0,
    )
    result = ex.dispatch(task, {"ready": True})
    print(f"Dispatch result: {result.outcome.name}")
    print(f"Summary: {ex.get_summary()}")
    print("executive: PASS")

def test_scheduler():
    from rocket_ai_os.mission.scheduler import (
        UASScheduler, SchedulableTask, UtilityFunction, CurveType,
    )

    sched = UASScheduler()
    tasks = [
        SchedulableTask(
            name="comms_window",
            utility_function=UtilityFunction(CurveType.LINEAR, 100, 50.0),
            resource_requirements={"cpu": 0.3},
            deadline=50.0,
            duration=5.0,
            priority=3,
        ),
        SchedulableTask(
            name="sensor_cal",
            utility_function=UtilityFunction(CurveType.STEP, 50),
            resource_requirements={"cpu": 0.2},
            deadline=100.0,
            duration=3.0,
            priority=5,
        ),
        SchedulableTask(
            name="solar_flare_response",
            utility_function=UtilityFunction(CurveType.STEP, 200),
            resource_requirements={"cpu": 0.5},
            deadline=10.0,
            duration=2.0,
            priority=1,
            emergency=True,
        ),
    ]
    schedule = sched.schedule(tasks, current_time=0.0)
    print(f"\nSchedule ({len(schedule)} entries):")
    for e in schedule:
        print(f"  {e.task.name}: t=[{e.start_time:.1f}, {e.end_time:.1f}] u={e.utility:.1f}")
    print(f"Total utility: {sched.total_utility:.1f}")
    print("scheduler: PASS")

def test_goac():
    from rocket_ai_os.mission.goac import (
        GOAC, GOACState, GoalStatus, Goal, WorldState,
    )
    from rocket_ai_os.config import MissionPhase

    goac = GOAC()
    print(repr(goac))

    ws = WorldState(
        fuel_remaining=0.9,
        phase=MissionPhase.PRE_LAUNCH,
        extra={
            "fuel_valve": "closed",
            "oxidiser_valve": "closed",
            "igniter_armed": False,
            "igniter_fired": False,
            "engines_running": False,
            "clamps_locked": True,
            "thrust_nominal": False,
        },
    )

    goac.set_goal(Goal(
        name="ignite_engine",
        priority=1,
        target_state={"engines_running": True},
        utility=100.0,
    ))

    # Run a few steps
    for i in range(6):
        actions = goac.step(ws)
        state = goac.state
        print(f"Step {i}: state={state.name}  actions={[a.name for a in actions]}")
        # Update world state from effects
        if actions:
            for a in actions:
                for k, v in a.effects.items():
                    if not callable(v):
                        ws.extra[k] = v

    status = goac.get_mission_status()
    print(f"Mission status: {status['goac_state']}")
    print("goac: PASS")

def test_init_imports():
    from rocket_ai_os.mission import (
        TaskStatus, PrimitiveTask, CompoundTask, HTNPlanner, PlanningError,
        ExecutionStatus, TaskOutcome, ExecutionResult, FaultInfo, Executive,
        CurveType, UtilityFunction, SchedulableTask, ScheduleEntry, UASScheduler,
        GOACState, GoalStatus, Goal, WorldState, GOAC,
    )
    print("All __init__ imports: PASS")

if __name__ == "__main__":
    test_planner()
    print()
    test_executive()
    print()
    test_scheduler()
    print()
    test_goac()
    print()
    test_init_imports()
    print("\n=== ALL TESTS PASSED ===")
