"""
Fault Tolerance Subsystem for Autonomous Rocket AI OS.

Implements multi-layered fault tolerance:
- TTEthernet deterministic networking (ARINC 664p7)
- Simplex Architecture for AI safety assurance
- Triple Modular Redundancy with SEU mitigation
- Fault Detection, Isolation, and Recovery (FDIR)
"""

from .ttethernet import (
    TrafficClass,
    TTFrame,
    TTEthernetSwitch,
    TTEthernetNetwork,
)

from .simplex import (
    ControlAction,
    SafetyEnvelope,
    DecisionModule,
    SafetyController,
    SimplexArchitecture,
)

from .tmr import (
    TMRVoter,
    TMRProcess,
)

from .fdir import (
    FaultType,
    FaultRecord,
    FDIRSystem,
)

__all__ = [
    # TTEthernet
    "TrafficClass",
    "TTFrame",
    "TTEthernetSwitch",
    "TTEthernetNetwork",
    # Simplex
    "ControlAction",
    "SafetyEnvelope",
    "DecisionModule",
    "SafetyController",
    "SimplexArchitecture",
    # TMR
    "TMRVoter",
    "TMRProcess",
    # FDIR
    "FaultType",
    "FaultRecord",
    "FDIRSystem",
]
