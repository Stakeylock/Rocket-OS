"""
Communications subsystem for the Autonomous Rocket AI OS.

Provides three integrated communication layers:

- **Cognitive Radio** -- adaptive SDR with spectrum sensing, policy-based
  reconfiguration, automatic modulation degradation, and hierarchical
  lost-link recovery.
- **Disruption-Tolerant Networking (DTN)** -- store-and-forward Bundle
  Protocol agent with custody transfer and link-opportunity flushing.
- **Self-Healing Mesh** -- multi-node mesh topology with an SDN controller,
  Dijkstra routing, flow-rule management, and automatic reroute around
  failed nodes.
"""

# -- Cognitive Radio ---------------------------------------------------------
from rocket_ai_os.comms.cognitive_radio import (
    Band,
    ModulationScheme,
    RecoveryPhase,
    LinkState,
    CognitiveRadioEngine,
    LinkRecoverySystem,
)

# -- Disruption-Tolerant Networking ------------------------------------------
from rocket_ai_os.comms.dtn import (
    BundlePriority,
    BundleStatus,
    Bundle,
    CustodySignal,
    CustodySignalType,
    CustodyTransfer,
    BundleProtocolAgent,
)

# -- Self-Healing Mesh -------------------------------------------------------
from rocket_ai_os.comms.mesh import (
    MeshNode,
    SDNFlowRule,
    Packet,
    SDNController,
    MeshNetwork,
)

__all__ = [
    # cognitive_radio
    "Band",
    "ModulationScheme",
    "RecoveryPhase",
    "LinkState",
    "CognitiveRadioEngine",
    "LinkRecoverySystem",
    # dtn
    "BundlePriority",
    "BundleStatus",
    "Bundle",
    "CustodySignal",
    "CustodySignalType",
    "CustodyTransfer",
    "BundleProtocolAgent",
    # mesh
    "MeshNode",
    "SDNFlowRule",
    "Packet",
    "SDNController",
    "MeshNetwork",
]
