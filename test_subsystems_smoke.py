"""
Smoke tests for Autonomous Rocket AI OS subsystems.

Verifies all major subsystem modules can be imported and core functionality works.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Core/RTOS Tests
# ============================================================================

def test_core_rtos_imports():
    """Test ARINC 653 RTOS imports."""
    from rocket_ai_os.core.rtos import (
        ARINC653Scheduler, PartitionedRTOS, TemporalPartition,
    )
    assert ARINC653Scheduler is not None


def test_core_software_bus():
    """Test cFS Software Bus messaging."""
    from rocket_ai_os.core.software_bus import SoftwareBus
    bus = SoftwareBus()
    assert bus is not None


# ============================================================================
# GNC Stack Tests
# ============================================================================

def test_gnc_navigation():
    """Test EKF navigation system."""
    from rocket_ai_os.gnc.navigation import NavigationSystem
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    nav = NavigationSystem(vehicle_config=cfg.vehicle, sim_config=cfg.sim, seed=42)
    assert nav is not None


def test_gnc_guidance():
    """Test G-FOLD guidance system."""
    from rocket_ai_os.gnc.guidance import GuidanceSystem
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    guidance = GuidanceSystem(vehicle_config=cfg.vehicle, guidance_config=cfg.guidance)
    assert guidance is not None


def test_gnc_control():
    """Test flight controller."""
    from rocket_ai_os.gnc.control import FlightController
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    controller = FlightController(vehicle_config=cfg.vehicle, rl_seed=42)
    assert controller is not None


# ============================================================================
# Propulsion Stack Tests
# ============================================================================

def test_propulsion_engine_cluster():
    """Test 9-engine cluster."""
    from rocket_ai_os.propulsion import EngineCluster
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    cluster = EngineCluster(config=cfg.vehicle)
    assert cluster is not None
    assert len(cluster.engines) == 9


def test_propulsion_anomaly_detector():
    """Test Transformer-based anomaly detection."""
    from rocket_ai_os.propulsion import TransformerAnomalyDetector

    detector = TransformerAnomalyDetector()
    assert detector is not None


# ============================================================================
# Fault Tolerance Tests
# ============================================================================

def test_fault_tolerance_simplex():
    """Test Simplex safety architecture."""
    from rocket_ai_os.fault_tolerance.simplex import (
        SimplexArchitecture, DecisionModule, SafetyEnvelope, ControlAction
    )

    env = SafetyEnvelope(max_angle_of_attack=np.radians(15.0))
    dm = DecisionModule(envelope=env)
    simplex = SimplexArchitecture(decision_module=dm)

    assert simplex is not None
    assert simplex.veto_count == 0


# ============================================================================
# Communications Tests
# ============================================================================

def test_comms_subsystem():
    """Test communications subsystem."""
    from rocket_ai_os import comms
    assert comms is not None


# ============================================================================
# Mission Management Tests
# ============================================================================

def test_mission_executive():
    """Test mission executive."""
    from rocket_ai_os.mission.executive import Executive

    executive = Executive()
    assert executive is not None


# ============================================================================
# Simulation Tests
# ============================================================================

def test_sim_vehicle():
    """Test 6-DOF vehicle dynamics."""
    from rocket_ai_os.sim.vehicle import Vehicle
    from rocket_ai_os.config import SystemConfig, MissionPhase

    cfg = SystemConfig()
    vehicle = Vehicle(config=cfg.vehicle, initial_phase=MissionPhase.LANDING_BURN)
    assert vehicle is not None


def test_sim_scenarios():
    """Test scenarios module."""
    from rocket_ai_os.sim import scenarios
    assert scenarios is not None


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_config():
    """Test system configuration."""
    from rocket_ai_os.config import SystemConfig

    cfg = SystemConfig()
    assert hasattr(cfg, 'vehicle')
    assert hasattr(cfg, 'sim')
    assert hasattr(cfg, 'guidance')


def test_integration_all_subsystems():
    """Test all major subsystem modules can be imported."""
    from rocket_ai_os import (
        core, gnc, propulsion, fault_tolerance,
        comms, mission, environment, sim
    )
    assert all([core, gnc, propulsion, fault_tolerance, comms, mission, environment, sim])


def test_integration_dependencies():
    """Test required dependencies are installed."""
    import numpy
    import scipy
    import sklearn
    assert all([numpy, scipy, sklearn])
    try:
        import cvxpy
        assert cvxpy is not None
    except ImportError:
        pytest.skip("cvxpy not installed (optional for G-FOLD guidance)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
