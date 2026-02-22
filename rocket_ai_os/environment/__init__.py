"""Environmental Analysis subsystem: terrain hazards, space weather, debris avoidance."""

from .alhat import (
    HazardType,
    TerrainCell,
    LandingSite,
    DigitalElevationModel,
    HazardDetector,
    LandingSiteSelector,
    ALHATSystem,
)
from .space_weather import (
    SpaceWeatherCondition,
    ShieldingMode,
    SolarParticleEvent,
    SpaceWeatherMonitor,
    RadiationShieldManager,
)
from .debris import (
    TrackedObject,
    CollisionPrediction,
    BurnPlan,
    DebrisTracker,
    CollisionAssessment,
    CollisionAvoidanceManeuver,
)

__all__ = [
    # ALHAT
    "HazardType",
    "TerrainCell",
    "LandingSite",
    "DigitalElevationModel",
    "HazardDetector",
    "LandingSiteSelector",
    "ALHATSystem",
    # Space Weather
    "SpaceWeatherCondition",
    "ShieldingMode",
    "SolarParticleEvent",
    "SpaceWeatherMonitor",
    "RadiationShieldManager",
    # Debris
    "TrackedObject",
    "CollisionPrediction",
    "BurnPlan",
    "DebrisTracker",
    "CollisionAssessment",
    "CollisionAvoidanceManeuver",
]
