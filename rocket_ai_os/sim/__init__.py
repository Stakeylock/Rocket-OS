"""
Simulation engine for the Autonomous Rocket AI OS.

Provides the vehicle dynamics model, physics environment (atmosphere,
aerodynamics, gravity, wind), and self-contained test scenarios for
verifying GNC, FDIR, and autonomy subsystems.
"""

from rocket_ai_os.sim.vehicle import VehicleState, Vehicle
from rocket_ai_os.sim.physics import (
    Atmosphere,
    AerodynamicModel,
    GravityModel,
    WindModel,
)
from rocket_ai_os.sim.scenarios import (
    ScenarioResult,
    Scenario,
    NominalLandingScenario,
    EngineOutScenario,
    SensorDegradationScenario,
    FullMissionScenario,
)

__all__ = [
    # Vehicle
    "VehicleState",
    "Vehicle",
    # Physics
    "Atmosphere",
    "AerodynamicModel",
    "GravityModel",
    "WindModel",
    # Scenarios
    "ScenarioResult",
    "Scenario",
    "NominalLandingScenario",
    "EngineOutScenario",
    "SensorDegradationScenario",
    "FullMissionScenario",
]
