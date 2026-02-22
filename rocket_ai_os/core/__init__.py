"""
Core module -- ARINC 653 RTOS with Time/Space Partitioning,
cFS-style Software Bus, and DDS integration.

Exports the principal classes needed to build and run the
partitioned real-time operating system simulator.
"""

from rocket_ai_os.core.rtos import (
    # Enumerations
    PartitionMode,
    PartitionHealth,
    PortDirection,
    PortDiscipline,
    # Exceptions
    MemoryAccessViolation,
    PartitionFault,
    SchedulingError,
    # Data structures
    MemoryRegion,
    PortMessage,
    ExecutionStatistics,
    # Spatial partitioning
    MemoryPartition,
    # Temporal partitioning
    TemporalPartition,
    # Scheduler
    ARINC653Scheduler,
    # Inter-partition communication
    SamplingPort,
    QueuingPort,
    Channel,
    # Top-level OS
    PartitionedRTOS,
)

__all__ = [
    # Enumerations
    "PartitionMode",
    "PartitionHealth",
    "PortDirection",
    "PortDiscipline",
    # Exceptions
    "MemoryAccessViolation",
    "PartitionFault",
    "SchedulingError",
    # Data structures
    "MemoryRegion",
    "PortMessage",
    "ExecutionStatistics",
    # Spatial partitioning
    "MemoryPartition",
    # Temporal partitioning
    "TemporalPartition",
    # Scheduler
    "ARINC653Scheduler",
    # Inter-partition communication
    "SamplingPort",
    "QueuingPort",
    "Channel",
    # Top-level OS
    "PartitionedRTOS",
]
