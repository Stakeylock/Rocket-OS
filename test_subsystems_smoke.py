"""
Comprehensive smoke tests for Autonomous Rocket AI OS subsystems.

This test suite provides basic coverage for all major subsystem modules,
ensuring core functionality works and dependencies are correctly installed.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Core/RTOS Tests
# ============================================================================

def test_core_rtos_partition():
    """Test ARINC 653 RTOS partition initialization."""
    from rocket_ai_os.core.rtos import Partition

    p = Partition(name="test", duration_ms=100, critical=True)
    assert p.name == "test"
    assert p.duration_ms == 100
    assert p.critical is True


def test_core_software_bus():
    """Test cFS Software Bus messaging."""
    from rocket_ai_os.core.software_bus import SoftwareBus, Message

    bus = SoftwareBus()

    # Publish a message
    msg = Message(msg_id=0x001, payload=b"test_data")
    bus.publish(msg)

    # Verify message was published
    assert len(bus._messages) > 0


# ============================================================================
# GNC Tests
# ============================================================================

def test_gnc_navigation_ekf():
    """Test Extended Kalman Filter navigation."""
    from rocket_ai_os.gnc.navigation import NavigationSystem
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    nav = NavigationSystem(
        vehicle_config=cfg.vehicle,
        sim_config=cfg.sim,
        seed=42
    )

    # Simulate a navigation update
    accel = np.array([0.0, 0.0, 9.81])
    omega = np.array([0.0, 0.0, 0.0])
    pos = np.array([0.0, 0.0, 1000.0])
    vel = np.array([0.0, 0.0, -100.0])
    mass = 1000.0

    nav_state = nav.step(accel, omega, pos, vel, mass, t=0.0)
    assert nav_state is not None


def test_gnc_guidance_gfold():
    """Test G-FOLD convex trajectory optimization."""
    from rocket_ai_os.gnc.guidance import GuidanceSystem
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    guidance = GuidanceSystem(
        vehicle_config=cfg.vehicle,
        guidance_config=cfg.guidance
    )

    # Simulate a guidance update
    from rocket_ai_os.gnc.navigation import NavigationState
    nav_state = NavigationState(
        position=np.array([0.0, 0.0, 1000.0]),
        velocity=np.array([0.0, 0.0, -100.0]),
        attitude=np.array([0.0, 0.0, 0.0]),
        angular_rates=np.array([0.0, 0.0, 0.0]),
        imu_biases=np.zeros(6),
        timestamp=0.0
    )

    traj = guidance.update(nav_state, t=0.0)
    assert traj is not None


def test_gnc_flight_controller():
    """Test flight controller with RL adaptive control."""
    from rocket_ai_os.gnc.control import FlightController
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    controller = FlightController(vehicle_config=cfg.vehicle, rl_seed=42)

    # Set desired state
    controller.set_desired_state(
        attitude=np.array([1.0, 0.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 500.0]),
        velocity=np.array([0.0, 0.0, -10.0]),
        throttle=0.6
    )

    # Generate control command
    from rocket_ai_os.gnc.navigation import NavigationState
    nav_state = NavigationState(
        position=np.array([0.0, 0.0, 1000.0]),
        velocity=np.array([0.0, 0.0, -50.0]),
        attitude=np.array([0.0, 0.0, 0.0]),
        angular_rates=np.array([0.0, 0.0, 0.0]),
        imu_biases=np.zeros(6),
        timestamp=0.0
    )

    cmd = controller.step(nav_state)
    assert cmd is not None
    assert hasattr(cmd, 'throttle')
    assert hasattr(cmd, 'torque_command')


# ============================================================================
# Propulsion Tests
# ============================================================================

def test_propulsion_engine_cluster():
    """Test 9-engine cluster simulation."""
    from rocket_ai_os.propulsion.engines import EngineCluster
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    cluster = EngineCluster(config=cfg.vehicle)

    # Set throttle levels
    thrust_cmds = np.full(9, 0.5)
    cluster.set_throttle(thrust_cmds)

    # Get thrust output
    for _ in range(10):
        thrust = cluster.compute_thrust(dt=0.01)
        assert isinstance(thrust, np.ndarray)
        assert thrust.shape == (3,)


def test_propulsion_ftca():
    """Test Fault-Tolerant Control Allocation."""
    from rocket_ai_os.propulsion.ftca import FTCA
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    ftca = FTCA(config=cfg.vehicle)

    # Desired command
    desired_thrust = np.array([0.0, 0.0, 500_000.0])
    desired_torque = np.array([0.0, 0.0, 0.0])

    # Compute allocation
    allocation = ftca.allocate(desired_thrust, desired_torque)
    assert allocation is not None
    assert len(allocation) == 9


# ============================================================================
# Fault Tolerance Tests
# ============================================================================

def test_fault_tolerance_simplex():
    """Test Simplex safety architecture."""
    from rocket_ai_os.fault_tolerance.simplex import (
        SimplexArchitecture, DecisionModule, SafetyEnvelope, ControlAction
    )
    import numpy as np

    # Create Simplex architecture
    env = SafetyEnvelope(max_angle_of_attack=np.radians(15.0))
    dm = DecisionModule(envelope=env)
    simplex = SimplexArchitecture(decision_module=dm)

    # Propose an action
    ai_action = ControlAction(
        forces=np.array([0.0, 0.0, 500_000.0]),
        torques=np.array([0.0, 0.0, 0.0]),
        timestamp=0.0,
        source="ai_controller"
    )

    vehicle_state = {
        'position': np.array([0.0, 0.0, 1000.0]),
        'velocity': np.array([0.0, 0.0, -50.0]),
        'attitude': np.array([0.0, 0.1, 0.0]),
        'angular_velocity': np.array([0.0, 0.0, 0.0]),
        'mass': 5000.0,
        'timestamp': 0.0
    }

    # Evaluate and select
    approved = simplex.evaluate_and_select(ai_action, vehicle_state)
    assert approved is not None
    assert hasattr(approved, 'source')


def test_fault_tolerance_tmr():
    """Test Triple Modular Redundancy."""
    from rocket_ai_os.fault_tolerance.tmr import TripleModularRedundancy

    tmr = TripleModularRedundancy(
        num_voters=3,
        voting_threshold=2
    )

    # Test voting
    outputs = [1.0, 1.0, 0.5]
    result = tmr.vote(outputs)
    assert result == 1.0


# ============================================================================
# Communications Tests
# ============================================================================

def test_comms_cognitive_radio():
    """Test cognitive radio with adaptive modulation."""
    from rocket_ai_os.comms.cognitive_radio import CognitiveRadio

    radio = CognitiveRadio(carrier_freq=2.4e9)

    # Check initial state
    assert radio.carrier_freq == 2.4e9

    # Simulate link quality update
    radio.update_link_quality(snr_db=20.0)
    modulation = radio.get_adaptive_modulation()
    assert modulation is not None


def test_comms_dtn():
    """Test Delay Tolerant Networking."""
    from rocket_ai_os.comms.dtn import Bundle, BundleRouter

    router = BundleRouter()

    # Create and route a bundle
    bundle = Bundle(src="rocket", dst="groundstation", payload=b"test_data")
    router.forward(bundle)

    assert router is not None


# ============================================================================
# Mission Management Tests
# ============================================================================

def test_mission_htn_planner():
    """Test Hierarchical Task Network planner."""
    from rocket_ai_os.mission.htn_planner import HTNPlanner, Goal

    planner = HTNPlanner()

    # Create a goal
    goal = Goal(name="land", target_position=np.array([0.0, 0.0, 0.0]))

    # Plan
    plan = planner.plan_from_state({}, goal)
    assert plan is not None


def test_mission_goac():
    """Test Goal-Oriented Autonomous Controller."""
    from rocket_ai_os.mission.goac import GOAC

    goac = GOAC()

    # Initialize
    state = {}
    current_goal = "land"

    # Get control
    ctrl = goac.step(state, current_goal)
    assert ctrl is not None


# ============================================================================
# Simulation Tests
# ============================================================================

def test_sim_vehicle_dynamics():
    """Test 6-DOF vehicle dynamics."""
    from rocket_ai_os.sim.vehicle import Vehicle
    from rocket_ai_os.config import SystemConfig, MissionPhase

    cfg = SystemConfig()
    vehicle = Vehicle(config=cfg.vehicle, initial_phase=MissionPhase.LANDING_BURN)

    # Set initial state
    vehicle.set_state(
        position=np.array([0.0, 0.0, 1000.0]),
        velocity=np.array([0.0, 0.0, -100.0]),
        mass=5000.0,
        fuel_mass=1000.0
    )

    # Simulate one step
    force = np.array([0.0, 0.0, 50_000.0])  # upward thrust
    torque = np.array([0.0, 0.0, 0.0])

    vehicle.apply_forces(force, torque, dt=0.01)

    # Check state updates
    assert vehicle.state.position is not None
    assert vehicle.state.velocity is not None


def test_sim_atmosphere():
    """Test atmospheric model."""
    from rocket_ai_os.sim.atmosphere import AtmosphereModel

    atm = AtmosphereModel()

    # Get density at different altitudes
    density_0 = atm.get_density(altitude=0.0)
    density_1000 = atm.get_density(altitude=1000.0)

    assert density_0 > density_1000


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_config_loading():
    """Test system configuration loading."""
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()

    # Verify key configs are present
    assert hasattr(cfg, 'vehicle')
    assert hasattr(cfg, 'sim')
    assert hasattr(cfg, 'guidance')
    assert hasattr(cfg, 'control')


def test_integration_package_imports():
    """Test all major package imports work."""
    import rocket_ai_os
    from rocket_ai_os import core, gnc, propulsion, fault_tolerance, comms, mission, sim

    assert rocket_ai_os is not None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
