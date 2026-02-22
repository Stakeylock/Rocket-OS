"""
ARINC 653-Style Real-Time Operating System Simulator
with Time and Space Partitioning.

Implements the core RTOS concepts from ARINC 653 Part 1:
- Spatial partitioning via memory isolation (MMU simulation)
- Temporal partitioning via cyclic executive scheduling
- Health monitoring and fault containment
- Inter-partition communication via sampling/queuing ports

Reference: ARINC 653 Part 1, Supplement 4 -- Avionics Application
Standard Software Interface (March 2015).
"""

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from rocket_ai_os.config import CriticalityLevel, RTOSConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PartitionMode(Enum):
    """ARINC 653 partition operating modes."""
    IDLE = auto()
    COLD_START = auto()
    WARM_START = auto()
    NORMAL = auto()
    FAULTED = auto()
    SHUTDOWN = auto()


class PartitionHealth(Enum):
    """Health status of a partition."""
    HEALTHY = auto()
    DEGRADED = auto()
    FAULTED = auto()
    FAILED = auto()


class PortDirection(Enum):
    """Direction for inter-partition communication ports."""
    SOURCE = auto()
    DESTINATION = auto()


class PortDiscipline(Enum):
    """ARINC 653 port disciplines."""
    SAMPLING = auto()   # Latest-value semantics (overwrite)
    QUEUING = auto()    # FIFO semantics


class MemoryAccessViolation(Exception):
    """Raised when a partition attempts out-of-bounds memory access."""
    pass


class PartitionFault(Exception):
    """Raised when a partition experiences a containable fault."""
    pass


class SchedulingError(Exception):
    """Raised for scheduling configuration or runtime errors."""
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryRegion:
    """A contiguous block of simulated memory assigned to a partition."""
    base_address: int
    size_bytes: int
    owner_partition: str
    readable: bool = True
    writable: bool = True
    executable: bool = True

    @property
    def end_address(self) -> int:
        return self.base_address + self.size_bytes

    def contains(self, address: int, length: int = 1) -> bool:
        """Check whether a given address range falls within this region."""
        return (self.base_address <= address
                and address + length <= self.end_address)


@dataclass
class PortMessage:
    """A single message transmitted through an inter-partition port."""
    payload: Any
    timestamp: float
    source_partition: str


@dataclass
class ExecutionStatistics:
    """Runtime statistics collected for each partition."""
    total_activations: int = 0
    total_execution_time_ms: float = 0.0
    worst_case_execution_time_ms: float = 0.0
    best_case_execution_time_ms: float = float("inf")
    deadline_misses: int = 0
    time_budget_overruns: int = 0
    faults_raised: int = 0
    last_execution_time_ms: float = 0.0

    @property
    def average_execution_time_ms(self) -> float:
        if self.total_activations == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_activations

    def record_execution(self, elapsed_ms: float, overran: bool) -> None:
        """Record one execution window's statistics."""
        self.total_activations += 1
        self.total_execution_time_ms += elapsed_ms
        self.last_execution_time_ms = elapsed_ms
        if elapsed_ms > self.worst_case_execution_time_ms:
            self.worst_case_execution_time_ms = elapsed_ms
        if elapsed_ms < self.best_case_execution_time_ms:
            self.best_case_execution_time_ms = elapsed_ms
        if overran:
            self.time_budget_overruns += 1


# ---------------------------------------------------------------------------
# MemoryPartition  --  Spatial partitioning / MMU simulation
# ---------------------------------------------------------------------------

class MemoryPartition:
    """Simulates spatial partitioning via an MMU-like address-space model.

    Each partition receives an isolated memory region.  Any access outside
    the assigned region raises a ``MemoryAccessViolation`` that the RTOS
    can contain without affecting other partitions.

    The simulated address space uses a simple byte-array backing store
    so that real read/write patterns can be exercised in tests.
    """

    def __init__(
        self,
        partition_name: str,
        budget_kb: int,
        base_address: int = 0,
        criticality: CriticalityLevel = CriticalityLevel.DAL_E,
    ) -> None:
        self.partition_name = partition_name
        self.budget_bytes = budget_kb * 1024
        self.criticality = criticality
        self._lock = threading.Lock()

        # Build the memory region descriptor
        self.region = MemoryRegion(
            base_address=base_address,
            size_bytes=self.budget_bytes,
            owner_partition=partition_name,
        )

        # Backing store -- lazily zeroed
        self._store: bytearray = bytearray(self.budget_bytes)

        # Track high-water mark for memory utilisation reporting
        self._hwm_offset: int = 0

        logger.info(
            "MemoryPartition '%s' created: base=0x%08X size=%d KB "
            "criticality=%s",
            partition_name,
            base_address,
            budget_kb,
            criticality.name,
        )

    # -- public API --

    def read(self, offset: int, length: int = 1) -> bytes:
        """Read *length* bytes starting at *offset* within this partition.

        Raises ``MemoryAccessViolation`` if the range falls outside the
        allocated region.
        """
        self._check_bounds(offset, length, "read")
        with self._lock:
            return bytes(self._store[offset : offset + length])

    def write(self, offset: int, data: bytes) -> None:
        """Write *data* starting at *offset* within this partition.

        Raises ``MemoryAccessViolation`` if the range falls outside the
        allocated region.
        """
        length = len(data)
        self._check_bounds(offset, length, "write")
        with self._lock:
            self._store[offset : offset + length] = data
            end = offset + length
            if end > self._hwm_offset:
                self._hwm_offset = end

    def zero(self) -> None:
        """Clear the entire backing store (cold-start reset)."""
        with self._lock:
            self._store = bytearray(self.budget_bytes)
            self._hwm_offset = 0
        logger.debug("MemoryPartition '%s' zeroed.", self.partition_name)

    @property
    def utilisation_fraction(self) -> float:
        """Fraction of the allocated budget that has been written to."""
        if self.budget_bytes == 0:
            return 0.0
        return self._hwm_offset / self.budget_bytes

    # -- internal helpers --

    def _check_bounds(
        self, offset: int, length: int, operation: str
    ) -> None:
        if offset < 0 or offset + length > self.budget_bytes:
            msg = (
                f"MemoryAccessViolation in partition '{self.partition_name}': "
                f"{operation} at offset {offset} length {length} exceeds "
                f"budget of {self.budget_bytes} bytes"
            )
            logger.error(msg)
            raise MemoryAccessViolation(msg)

    def __repr__(self) -> str:
        return (
            f"MemoryPartition(name={self.partition_name!r}, "
            f"budget={self.budget_bytes} B, "
            f"utilisation={self.utilisation_fraction:.1%})"
        )


# ---------------------------------------------------------------------------
# TemporalPartition  --  Time partitioning / WCET tracking
# ---------------------------------------------------------------------------

class TemporalPartition:
    """Represents one time-partition inside the ARINC 653 cyclic schedule.

    A temporal partition owns:
    * A guaranteed *time_slice_ms* within every major frame.
    * An optional *task_callable* that the scheduler invokes each window.
    * WCET tracking counters.
    * A health/mode state machine.
    """

    def __init__(
        self,
        name: str,
        time_slice_ms: float,
        criticality: CriticalityLevel = CriticalityLevel.DAL_E,
        task: Optional[Callable[[], None]] = None,
    ) -> None:
        self.name = name
        self.time_slice_ms = time_slice_ms
        self.criticality = criticality
        self._task: Optional[Callable[[], None]] = task

        # Mode / health
        self.mode: PartitionMode = PartitionMode.COLD_START
        self.health: PartitionHealth = PartitionHealth.HEALTHY

        # Statistics
        self.stats = ExecutionStatistics()

        # Unique identifier for this partition instance
        self.instance_id: str = uuid.uuid4().hex[:12]

        logger.info(
            "TemporalPartition '%s' created: slice=%.2f ms, "
            "criticality=%s, id=%s",
            name,
            time_slice_ms,
            criticality.name,
            self.instance_id,
        )

    # -- task management --

    def set_task(self, task: Callable[[], None]) -> None:
        """Bind (or rebind) the callable executed during this partition's
        window.  Supports the *independent build, link, load* model:
        partition images are loaded separately."""
        self._task = task
        logger.debug(
            "Partition '%s' task bound: %s",
            self.name,
            getattr(task, "__name__", repr(task)),
        )

    @property
    def has_task(self) -> bool:
        return self._task is not None

    # -- execution --

    def execute(self, wall_clock_s: float) -> float:
        """Run this partition's task with WCET enforcement.

        The task is executed inside a thread with a timeout equal to
        ``time_slice_ms``.  If the task exceeds the budget the thread
        is abandoned (simulating a forced context switch) and a budget
        overrun is recorded.

        Returns the actual elapsed time in **milliseconds**.
        """
        if self._task is None:
            logger.debug(
                "Partition '%s' has no task; skipping.", self.name
            )
            return 0.0

        if self.mode == PartitionMode.FAULTED:
            logger.warning(
                "Partition '%s' is FAULTED; skipping execution.",
                self.name,
            )
            return 0.0

        # Transition to NORMAL on first real execution
        if self.mode in (PartitionMode.COLD_START, PartitionMode.WARM_START):
            self.mode = PartitionMode.NORMAL

        overran = False
        fault_occurred = False
        elapsed_ms = 0.0
        result_container: Dict[str, Any] = {"exception": None}

        def _run_task() -> None:
            try:
                self._task()  # type: ignore[misc]
            except Exception as exc:
                result_container["exception"] = exc

        start = time.perf_counter()
        thread = threading.Thread(
            target=_run_task, name=f"partition-{self.name}", daemon=True
        )
        thread.start()
        # Wait at most the allocated time slice (convert ms -> s)
        timeout_s = self.time_slice_ms / 1000.0
        thread.join(timeout=timeout_s)
        elapsed_s = time.perf_counter() - start
        elapsed_ms = elapsed_s * 1000.0

        if thread.is_alive():
            # Task exceeded its time budget -- forced context switch
            overran = True
            elapsed_ms = self.time_slice_ms  # charge full budget
            logger.warning(
                "Partition '%s' OVERRAN time budget (%.2f ms). "
                "Forced context switch.",
                self.name,
                self.time_slice_ms,
            )

        if result_container["exception"] is not None:
            fault_occurred = True
            self.stats.faults_raised += 1
            self.health = PartitionHealth.DEGRADED
            logger.error(
                "Partition '%s' raised exception: %s",
                self.name,
                result_container["exception"],
            )

        self.stats.record_execution(elapsed_ms, overran)
        return elapsed_ms

    # -- mode transitions --

    def restart(self, warm: bool = False) -> None:
        """Restart the partition (cold or warm)."""
        target = (
            PartitionMode.WARM_START if warm else PartitionMode.COLD_START
        )
        logger.info(
            "Partition '%s' restarting (%s).", self.name, target.name
        )
        self.mode = target
        self.health = PartitionHealth.HEALTHY

    def set_faulted(self) -> None:
        self.mode = PartitionMode.FAULTED
        self.health = PartitionHealth.FAULTED
        logger.error("Partition '%s' marked FAULTED.", self.name)

    def shutdown(self) -> None:
        self.mode = PartitionMode.SHUTDOWN
        logger.info("Partition '%s' SHUTDOWN.", self.name)

    def __repr__(self) -> str:
        return (
            f"TemporalPartition(name={self.name!r}, "
            f"slice={self.time_slice_ms:.2f} ms, "
            f"mode={self.mode.name}, health={self.health.name})"
        )


# ---------------------------------------------------------------------------
# ARINC653Scheduler  --  Cyclic-executive scheduler
# ---------------------------------------------------------------------------

class ARINC653Scheduler:
    """Cyclic-executive scheduler implementing ARINC 653 major-frame timing.

    Within each major frame the scheduler iterates through the partition
    schedule in order, giving each partition its guaranteed time window.
    A forced context switch occurs at the end of each window regardless
    of whether the partition's task has completed.

    Key properties
    * **Deterministic**: the schedule repeats identically every major frame.
    * **Time-isolated**: a misbehaving partition cannot steal cycles from
      others.
    * **Observable**: full execution statistics are recorded per partition.
    """

    def __init__(
        self,
        major_frame_ms: float,
        partitions: Optional[List[TemporalPartition]] = None,
    ) -> None:
        self.major_frame_ms = major_frame_ms
        self._schedule: List[TemporalPartition] = list(
            partitions or []
        )
        self._running = False
        self._major_frame_count: int = 0
        self._wall_clock_ms: float = 0.0

        # Validate on construction
        self._validate_schedule()

        logger.info(
            "ARINC653Scheduler created: major_frame=%.2f ms, "
            "%d partition(s).",
            major_frame_ms,
            len(self._schedule),
        )

    # -- schedule management --

    def add_partition(self, partition: TemporalPartition) -> None:
        self._schedule.append(partition)
        self._validate_schedule()

    def remove_partition(self, name: str) -> None:
        self._schedule = [
            p for p in self._schedule if p.name != name
        ]

    @property
    def schedule_order(self) -> List[str]:
        return [p.name for p in self._schedule]

    @property
    def total_allocated_ms(self) -> float:
        return sum(p.time_slice_ms for p in self._schedule)

    @property
    def slack_ms(self) -> float:
        """Unallocated time in the major frame (spare capacity)."""
        return self.major_frame_ms - self.total_allocated_ms

    @property
    def major_frame_count(self) -> int:
        return self._major_frame_count

    # -- execution --

    def run_one_major_frame(self) -> Dict[str, float]:
        """Execute exactly one major frame, returning per-partition
        elapsed times in milliseconds."""
        self._major_frame_count += 1
        frame_start = time.perf_counter()
        elapsed_map: Dict[str, float] = {}

        logger.debug(
            "--- Major frame #%d start (wall=%.3f ms) ---",
            self._major_frame_count,
            self._wall_clock_ms,
        )

        for partition in self._schedule:
            elapsed = partition.execute(self._wall_clock_ms)
            elapsed_map[partition.name] = elapsed
            self._wall_clock_ms += partition.time_slice_ms

        frame_elapsed = (time.perf_counter() - frame_start) * 1000.0

        # If we finished early, log the slack
        if frame_elapsed < self.major_frame_ms:
            logger.debug(
                "Major frame #%d completed in %.3f ms (slack %.3f ms).",
                self._major_frame_count,
                frame_elapsed,
                self.major_frame_ms - frame_elapsed,
            )
        else:
            logger.warning(
                "Major frame #%d overran: %.3f ms > %.3f ms budget.",
                self._major_frame_count,
                frame_elapsed,
                self.major_frame_ms,
            )

        return elapsed_map

    def run(self, num_frames: int = 1) -> List[Dict[str, float]]:
        """Run *num_frames* consecutive major frames."""
        self._running = True
        results: List[Dict[str, float]] = []
        for _ in range(num_frames):
            if not self._running:
                break
            results.append(self.run_one_major_frame())
        self._running = False
        return results

    def stop(self) -> None:
        """Request the scheduler to stop after the current major frame."""
        self._running = False

    # -- health overview --

    def partition_health_summary(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Return a dict of per-partition health and statistics."""
        summary: Dict[str, Dict[str, Any]] = {}
        for p in self._schedule:
            summary[p.name] = {
                "mode": p.mode.name,
                "health": p.health.name,
                "criticality": p.criticality.name,
                "total_activations": p.stats.total_activations,
                "wcet_ms": p.stats.worst_case_execution_time_ms,
                "avg_ms": p.stats.average_execution_time_ms,
                "overruns": p.stats.time_budget_overruns,
                "faults": p.stats.faults_raised,
            }
        return summary

    # -- internal --

    def _validate_schedule(self) -> None:
        total = self.total_allocated_ms
        if total > self.major_frame_ms:
            raise SchedulingError(
                f"Schedule overcommitted: {total:.2f} ms allocated "
                f"but major frame is only {self.major_frame_ms:.2f} ms."
            )


# ---------------------------------------------------------------------------
# Inter-partition communication ports (ARINC 653 sampling / queuing)
# ---------------------------------------------------------------------------

@dataclass
class SamplingPort:
    """ARINC 653 sampling port -- latest-value semantics.

    A SOURCE port writes; a DESTINATION port reads the latest value.
    Validity is determined by a configurable *refresh_period_ms*; if the
    message age exceeds the period the port is considered INVALID.
    """

    name: str
    direction: PortDirection
    max_message_size: int = 4096
    refresh_period_ms: float = 100.0
    _message: Optional[PortMessage] = field(default=None, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def write(
        self, payload: Any, timestamp: float, source: str
    ) -> None:
        if self.direction != PortDirection.SOURCE:
            raise PartitionFault(
                f"Cannot write to DESTINATION sampling port '{self.name}'."
            )
        with self._lock:
            self._message = PortMessage(
                payload=payload, timestamp=timestamp,
                source_partition=source,
            )

    def read(self, current_time: float) -> Tuple[Optional[Any], bool]:
        """Return *(payload, valid)*.  Valid is False when no message
        has been written or the message has expired."""
        with self._lock:
            if self._message is None:
                return None, False
            age_ms = (current_time - self._message.timestamp) * 1000.0
            valid = age_ms <= self.refresh_period_ms
            return self._message.payload, valid


@dataclass
class QueuingPort:
    """ARINC 653 queuing port -- FIFO semantics with bounded depth."""

    name: str
    direction: PortDirection
    max_message_size: int = 4096
    max_depth: int = 16
    _queue: Deque[PortMessage] = field(
        default_factory=deque, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def send(
        self, payload: Any, timestamp: float, source: str
    ) -> bool:
        """Enqueue a message.  Returns False if the queue is full."""
        if self.direction != PortDirection.SOURCE:
            raise PartitionFault(
                f"Cannot send on DESTINATION queuing port '{self.name}'."
            )
        with self._lock:
            if len(self._queue) >= self.max_depth:
                logger.warning(
                    "QueuingPort '%s' full (%d messages); dropping.",
                    self.name,
                    self.max_depth,
                )
                return False
            self._queue.append(
                PortMessage(
                    payload=payload, timestamp=timestamp,
                    source_partition=source,
                )
            )
            return True

    def receive(self) -> Optional[PortMessage]:
        """Dequeue the oldest message, or return None if empty."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)


# ---------------------------------------------------------------------------
# Channel -- connects a SOURCE port to one or more DESTINATION ports
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """Unidirectional data channel connecting ports across partitions."""
    name: str
    source_port: str          # "<partition>:<port>"
    destination_ports: List[str] = field(default_factory=list)
    discipline: PortDiscipline = PortDiscipline.SAMPLING


# ---------------------------------------------------------------------------
# PartitionedRTOS  --  Top-level OS abstraction
# ---------------------------------------------------------------------------

class PartitionedRTOS:
    """Top-level ARINC 653 partitioned RTOS simulator.

    Responsibilities
    * Create and manage temporal **and** spatial partitions.
    * Drive the ``ARINC653Scheduler`` cyclic executive.
    * Enforce memory isolation across partitions.
    * Contain faults within the faulting partition.
    * Provide inter-partition communication via sampling/queuing ports.
    * Support the *independent build, link, load* model: partitions
      can be added and configured independently before the scheduler
      is started.
    * Maintain a comprehensive activity log.
    """

    def __init__(self, config: Optional[RTOSConfig] = None) -> None:
        self._config = config or RTOSConfig()

        # Temporal partitions keyed by name
        self._temporal: Dict[str, TemporalPartition] = {}

        # Spatial partitions keyed by name
        self._spatial: Dict[str, MemoryPartition] = {}

        # Inter-partition communication
        self._sampling_ports: Dict[str, SamplingPort] = {}
        self._queuing_ports: Dict[str, QueuingPort] = {}
        self._channels: Dict[str, Channel] = {}

        # Address allocator cursor (simple bump allocator)
        self._next_base_address: int = 0x1000_0000

        # Build the scheduler (partitions added lazily)
        self._scheduler = ARINC653Scheduler(
            major_frame_ms=self._config.major_frame_ms
        )

        # Activity log (bounded ring buffer)
        self._activity_log: Deque[str] = deque(maxlen=10_000)

        self._started = False

        self._log("PartitionedRTOS initialised (major_frame=%.2f ms).",
                   self._config.major_frame_ms)

    # ------------------------------------------------------------------
    # Partition creation  (independent build, link, load)
    # ------------------------------------------------------------------

    def create_partition(
        self,
        name: str,
        time_slice_ms: Optional[float] = None,
        memory_kb: Optional[int] = None,
        criticality: Optional[CriticalityLevel] = None,
        task: Optional[Callable[[], None]] = None,
    ) -> Tuple[TemporalPartition, MemoryPartition]:
        """Create a paired temporal + spatial partition.

        If *time_slice_ms*, *memory_kb*, or *criticality* are not supplied
        they are looked up from the ``RTOSConfig`` by partition *name*.
        """
        if name in self._temporal:
            raise ValueError(
                f"Partition '{name}' already exists."
            )

        # Resolve configuration defaults
        slice_ms = time_slice_ms or self._config.partition_schedule.get(
            name, 10.0
        )
        mem_kb = memory_kb or self._config.memory_budget_kb.get(
            name, 1024
        )
        crit = criticality or self._config.partition_criticality.get(
            name, CriticalityLevel.DAL_E
        )

        # Spatial partition
        base = self._next_base_address
        mem_part = MemoryPartition(
            partition_name=name,
            budget_kb=mem_kb,
            base_address=base,
            criticality=crit,
        )
        self._next_base_address += mem_kb * 1024
        self._spatial[name] = mem_part

        # Temporal partition
        temp_part = TemporalPartition(
            name=name,
            time_slice_ms=slice_ms,
            criticality=crit,
            task=task,
        )
        self._temporal[name] = temp_part

        # Register with the scheduler
        self._scheduler.add_partition(temp_part)

        self._log(
            "Created partition '%s': slice=%.2f ms, memory=%d KB, "
            "DAL=%s, base=0x%08X.",
            name, slice_ms, mem_kb, crit.name, base,
        )
        return temp_part, mem_part

    def load_partition_image(
        self, name: str, task: Callable[[], None]
    ) -> None:
        """Bind (or rebind) a partition's executable image.

        This mirrors the *independent build, link, load* concept:
        partition software is built separately and loaded into the
        RTOS at integration time.
        """
        if name not in self._temporal:
            raise KeyError(
                f"Partition '{name}' does not exist."
            )
        self._temporal[name].set_task(task)
        self._log("Loaded image for partition '%s'.", name)

    def remove_partition(self, name: str) -> None:
        """Remove a partition entirely (for reconfiguration)."""
        self._temporal.pop(name, None)
        self._spatial.pop(name, None)
        self._scheduler.remove_partition(name)
        self._log("Removed partition '%s'.", name)

    # ------------------------------------------------------------------
    # Memory isolation helpers
    # ------------------------------------------------------------------

    def partition_memory_read(
        self, partition_name: str, offset: int, length: int = 1
    ) -> bytes:
        """Read from a partition's private memory space."""
        mem = self._get_memory_partition(partition_name)
        return mem.read(offset, length)

    def partition_memory_write(
        self, partition_name: str, offset: int, data: bytes
    ) -> None:
        """Write to a partition's private memory space."""
        mem = self._get_memory_partition(partition_name)
        mem.write(offset, data)

    def cross_partition_check(
        self,
        requester: str,
        target: str,
        offset: int,
        length: int,
    ) -> None:
        """Verify that *requester* is NOT accessing *target*'s memory.

        In a real ARINC 653 system the MMU would trap this.  Here we
        raise ``MemoryAccessViolation`` so the RTOS can contain the
        fault within the requesting partition.
        """
        if requester != target and target in self._spatial:
            raise MemoryAccessViolation(
                f"Partition '{requester}' attempted to access memory "
                f"of partition '{target}' at offset {offset} "
                f"length {length}."
            )

    # ------------------------------------------------------------------
    # Inter-partition communication (IPC)
    # ------------------------------------------------------------------

    def create_sampling_port(
        self,
        port_name: str,
        direction: PortDirection,
        max_message_size: int = 4096,
        refresh_period_ms: float = 100.0,
    ) -> SamplingPort:
        key = port_name
        port = SamplingPort(
            name=port_name,
            direction=direction,
            max_message_size=max_message_size,
            refresh_period_ms=refresh_period_ms,
        )
        self._sampling_ports[key] = port
        self._log(
            "Created sampling port '%s' (%s).",
            port_name, direction.name,
        )
        return port

    def create_queuing_port(
        self,
        port_name: str,
        direction: PortDirection,
        max_message_size: int = 4096,
        max_depth: int = 16,
    ) -> QueuingPort:
        key = port_name
        port = QueuingPort(
            name=port_name,
            direction=direction,
            max_message_size=max_message_size,
            max_depth=max_depth,
        )
        self._queuing_ports[key] = port
        self._log(
            "Created queuing port '%s' (%s, depth=%d).",
            port_name, direction.name, max_depth,
        )
        return port

    def create_channel(
        self,
        channel_name: str,
        source_port: str,
        destination_ports: List[str],
        discipline: PortDiscipline = PortDiscipline.SAMPLING,
    ) -> Channel:
        ch = Channel(
            name=channel_name,
            source_port=source_port,
            destination_ports=list(destination_ports),
            discipline=discipline,
        )
        self._channels[channel_name] = ch
        self._log(
            "Created channel '%s': %s -> %s (%s).",
            channel_name,
            source_port,
            destination_ports,
            discipline.name,
        )
        return ch

    def get_sampling_port(self, name: str) -> SamplingPort:
        return self._sampling_ports[name]

    def get_queuing_port(self, name: str) -> QueuingPort:
        return self._queuing_ports[name]

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    def create_all_configured_partitions(self) -> None:
        """Bulk-create partitions from ``RTOSConfig.partition_schedule``.

        Useful for quickly standing up the full default partition set.
        """
        for name, slice_ms in self._config.partition_schedule.items():
            if name not in self._temporal:
                self.create_partition(
                    name=name,
                    time_slice_ms=slice_ms,
                )

    def run_one_frame(self) -> Dict[str, float]:
        """Run one major frame and return per-partition elapsed times."""
        self._started = True
        result = self._scheduler.run_one_major_frame()
        self._log(
            "Major frame #%d completed.",
            self._scheduler.major_frame_count,
        )
        return result

    def run(self, num_frames: int = 1) -> List[Dict[str, float]]:
        """Run *num_frames* consecutive major frames."""
        self._started = True
        results = self._scheduler.run(num_frames)
        self._log(
            "Ran %d major frame(s). Total frames: %d.",
            num_frames,
            self._scheduler.major_frame_count,
        )
        return results

    def stop(self) -> None:
        self._scheduler.stop()
        self._log("Scheduler stopped.")

    # ------------------------------------------------------------------
    # Fault handling  (containment without affecting other partitions)
    # ------------------------------------------------------------------

    def handle_partition_fault(
        self, partition_name: str, fault: Exception
    ) -> None:
        """Contain a fault within the named partition.

        * DAL-A/B partitions are restarted (warm).
        * DAL-C/D partitions are placed in FAULTED mode.
        * DAL-E partitions are shut down.
        """
        self._log(
            "FAULT in partition '%s': %s", partition_name, fault
        )
        temp = self._temporal.get(partition_name)
        if temp is None:
            logger.error(
                "Cannot handle fault: partition '%s' unknown.",
                partition_name,
            )
            return

        crit = temp.criticality
        if crit in (CriticalityLevel.DAL_A, CriticalityLevel.DAL_B):
            # Safety-critical: attempt warm restart
            self._log(
                "Warm-restarting safety-critical partition '%s' "
                "(DAL %s).",
                partition_name,
                crit.name,
            )
            temp.restart(warm=True)
            mem = self._spatial.get(partition_name)
            if mem:
                mem.zero()  # scrub memory for isolation
        elif crit in (CriticalityLevel.DAL_C, CriticalityLevel.DAL_D):
            self._log(
                "Marking partition '%s' FAULTED (DAL %s).",
                partition_name,
                crit.name,
            )
            temp.set_faulted()
        else:
            self._log(
                "Shutting down non-critical partition '%s' (DAL %s).",
                partition_name,
                crit.name,
            )
            temp.shutdown()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def scheduler(self) -> ARINC653Scheduler:
        return self._scheduler

    @property
    def partition_names(self) -> List[str]:
        return list(self._temporal.keys())

    def get_temporal_partition(self, name: str) -> TemporalPartition:
        return self._temporal[name]

    def get_memory_partition(self, name: str) -> MemoryPartition:
        return self._spatial[name]

    def health_report(self) -> Dict[str, Dict[str, Any]]:
        """Consolidated health report for all partitions."""
        report: Dict[str, Dict[str, Any]] = {}
        for name in self._temporal:
            temp = self._temporal[name]
            mem = self._spatial.get(name)
            report[name] = {
                "mode": temp.mode.name,
                "health": temp.health.name,
                "criticality": temp.criticality.name,
                "time_slice_ms": temp.time_slice_ms,
                "stats": {
                    "activations": temp.stats.total_activations,
                    "wcet_ms": temp.stats.worst_case_execution_time_ms,
                    "avg_ms": temp.stats.average_execution_time_ms,
                    "overruns": temp.stats.time_budget_overruns,
                    "faults": temp.stats.faults_raised,
                },
                "memory": {
                    "budget_bytes": (
                        mem.budget_bytes if mem else None
                    ),
                    "utilisation": (
                        mem.utilisation_fraction if mem else None
                    ),
                } if mem else None,
            }
        return report

    @property
    def activity_log(self) -> List[str]:
        """Return the activity log as a list (most recent last)."""
        return list(self._activity_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_memory_partition(self, name: str) -> MemoryPartition:
        try:
            return self._spatial[name]
        except KeyError:
            raise KeyError(
                f"No memory partition named '{name}'."
            ) from None

    def _log(self, msg: str, *args: Any) -> None:
        formatted = msg % args if args else msg
        self._activity_log.append(formatted)
        logger.info(formatted)

    def __repr__(self) -> str:
        return (
            f"PartitionedRTOS(partitions={self.partition_names}, "
            f"major_frame={self._config.major_frame_ms:.2f} ms, "
            f"frames_run={self._scheduler.major_frame_count})"
        )
