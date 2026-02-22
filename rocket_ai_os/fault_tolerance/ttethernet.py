"""
Time-Triggered Ethernet (TTEthernet / ARINC 664p7) Network Implementation.

Provides deterministic communication for safety-critical avionics:
- Time-triggered (TT) traffic with microsecond-level synchronization
- Rate-constrained (RC) traffic with guaranteed bandwidth (AFDX compliant)
- Best-effort (BE) traffic for non-critical data
- Guardian functionality to filter babbling-idiot failures
- Dual/triplex redundant topology with automatic failover
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class TrafficClass(Enum):
    """TTEthernet traffic classification per ARINC 664p7."""
    TIME_TRIGGERED = auto()    # Deterministic, clock-synchronized
    RATE_CONSTRAINED = auto()  # Bounded latency, AFDX style
    BEST_EFFORT = auto()       # Standard Ethernet, no guarantees


@dataclass
class TTFrame:
    """A TTEthernet frame with traffic classification metadata."""
    frame_id: int
    traffic_class: TrafficClass
    source: str
    destination: str
    payload: np.ndarray
    timestamp: float
    priority: int = 0

    def size_bytes(self) -> int:
        """Compute frame size including overhead (Ethernet + AFDX headers)."""
        header_bytes = 62  # Ethernet(14) + IP(20) + UDP(8) + AFDX sub-header(20)
        payload_bytes = self.payload.nbytes if isinstance(self.payload, np.ndarray) else 0
        return header_bytes + payload_bytes


# ---------------------------------------------------------------------------
# Guardian -- filters babbling-idiot failures at the switch port level
# ---------------------------------------------------------------------------

class _Guardian:
    """
    Per-port guardian that enforces the TT schedule.

    A babbling-idiot failure occurs when a node transmits outside its
    allocated time slot.  The guardian blocks any frame that arrives
    outside the sender's scheduled window.
    """

    def __init__(self, enabled: bool, tt_cycle_us: float):
        self.enabled = enabled
        self.tt_cycle_us = tt_cycle_us
        # schedule: source -> list of (window_start_us, window_end_us)
        self._schedule: Dict[str, List[Tuple[float, float]]] = {}
        self._blocked_count: int = 0

    # -- public API ---------------------------------------------------------

    def register_window(self, source: str, start_us: float, end_us: float) -> None:
        """Register an allowed transmission window for *source*."""
        self._schedule.setdefault(source, []).append((start_us, end_us))

    def check(self, frame: TTFrame, current_time_us: float) -> bool:
        """
        Return True if the frame is allowed through, False if blocked.

        Non-TT traffic is always allowed when the guardian is enabled
        (it only polices TT windows).  If the guardian is disabled every
        frame passes.
        """
        if not self.enabled:
            return True

        if frame.traffic_class != TrafficClass.TIME_TRIGGERED:
            return True

        windows = self._schedule.get(frame.source)
        if windows is None:
            # No registered window -- block by default for TT traffic
            self._blocked_count += 1
            logger.warning(
                "Guardian blocked TT frame from %s -- no registered window",
                frame.source,
            )
            return False

        phase = current_time_us % self.tt_cycle_us
        for start, end in windows:
            if start <= phase <= end:
                return True

        self._blocked_count += 1
        logger.warning(
            "Guardian blocked babbling-idiot frame from %s at phase %.1f us",
            frame.source,
            phase,
        )
        return False

    @property
    def blocked_count(self) -> int:
        return self._blocked_count


# ---------------------------------------------------------------------------
# TTEthernetSwitch
# ---------------------------------------------------------------------------

class TTEthernetSwitch:
    """
    A single TTEthernet switch with three traffic-class queues.

    Features
    --------
    * **TT queue** -- frames dispatched on a strict microsecond schedule.
    * **RC queue** -- AFDX-compliant bandwidth allocation with BAG
      (Bandwidth Allocation Gap) enforcement.
    * **BE queue** -- standard FIFO, dispatched only when TT/RC are idle.
    * **Guardian** -- per-port filtering of babbling-idiot failures.
    """

    def __init__(
        self,
        switch_id: str,
        tt_cycle_us: float = 500.0,
        rc_bandwidth_mbps: float = 100.0,
        be_bandwidth_mbps: float = 1000.0,
        guardian_enabled: bool = True,
    ):
        self.switch_id = switch_id
        self.tt_cycle_us = tt_cycle_us
        self.rc_bandwidth_mbps = rc_bandwidth_mbps
        self.be_bandwidth_mbps = be_bandwidth_mbps
        self.healthy = True

        # Per-class queues
        self._tt_queue: List[TTFrame] = []
        self._rc_queue: Deque[TTFrame] = deque()
        self._be_queue: Deque[TTFrame] = deque()

        # RC bandwidth tracking -- bytes transmitted in current window
        self._rc_bytes_this_window: float = 0.0
        self._rc_window_start: float = 0.0
        self._rc_window_us: float = 1000.0  # 1 ms measurement window

        # TT schedule: frame_id -> scheduled dispatch phase (microseconds)
        self._tt_schedule: Dict[int, float] = {}

        # Guardian
        self._guardian = _Guardian(enabled=guardian_enabled, tt_cycle_us=tt_cycle_us)

        # Delivered frames per destination
        self._output_buffers: Dict[str, Deque[TTFrame]] = {}

        # Statistics
        self.frames_forwarded: int = 0
        self.frames_dropped: int = 0

    # -- configuration ------------------------------------------------------

    def add_tt_schedule_entry(self, frame_id: int, dispatch_phase_us: float) -> None:
        """Register when a given TT frame_id should be dispatched."""
        self._tt_schedule[frame_id] = dispatch_phase_us

    def register_guardian_window(self, source: str, start_us: float, end_us: float) -> None:
        """Allow *source* to transmit TT traffic in [start_us, end_us]."""
        self._guardian.register_window(source, start_us, end_us)

    # -- ingress ------------------------------------------------------------

    def ingest(self, frame: TTFrame, current_time_us: float) -> bool:
        """
        Accept a frame from a port.  Returns False if blocked by the
        guardian or if the switch is unhealthy.
        """
        if not self.healthy:
            self.frames_dropped += 1
            return False

        if not self._guardian.check(frame, current_time_us):
            self.frames_dropped += 1
            return False

        if frame.traffic_class == TrafficClass.TIME_TRIGGERED:
            self._tt_queue.append(frame)
        elif frame.traffic_class == TrafficClass.RATE_CONSTRAINED:
            self._rc_queue.append(frame)
        else:
            self._be_queue.append(frame)
        return True

    # -- egress / dispatch --------------------------------------------------

    def dispatch(self, current_time_us: float) -> List[TTFrame]:
        """
        Run one dispatch cycle.  Returns a list of frames forwarded to
        their destinations during this cycle.

        Priority order: TT > RC > BE.
        """
        delivered: List[TTFrame] = []

        if not self.healthy:
            return delivered

        # 1. Time-triggered -- deterministic, zero-jitter delivery
        phase = current_time_us % self.tt_cycle_us
        remaining_tt: List[TTFrame] = []
        for frame in self._tt_queue:
            scheduled = self._tt_schedule.get(frame.frame_id)
            if scheduled is not None and abs(phase - scheduled) < 1.0:
                self._deliver(frame)
                delivered.append(frame)
            else:
                remaining_tt.append(frame)
        self._tt_queue = remaining_tt

        # 2. Rate-constrained -- AFDX BAG enforcement
        if current_time_us - self._rc_window_start >= self._rc_window_us:
            self._rc_bytes_this_window = 0.0
            self._rc_window_start = current_time_us

        max_bytes_per_window = (self.rc_bandwidth_mbps * 1e6 / 8) * (self._rc_window_us / 1e6)
        while self._rc_queue:
            frame = self._rc_queue[0]
            frame_bytes = frame.size_bytes()
            if self._rc_bytes_this_window + frame_bytes <= max_bytes_per_window:
                self._rc_queue.popleft()
                self._rc_bytes_this_window += frame_bytes
                self._deliver(frame)
                delivered.append(frame)
            else:
                break  # bandwidth exhausted for this window

        # 3. Best-effort -- only if TT and RC had nothing
        if not delivered and self._be_queue:
            frame = self._be_queue.popleft()
            self._deliver(frame)
            delivered.append(frame)

        return delivered

    # -- internal -----------------------------------------------------------

    def _deliver(self, frame: TTFrame) -> None:
        dest = frame.destination
        if dest not in self._output_buffers:
            self._output_buffers[dest] = deque(maxlen=256)
        self._output_buffers[dest].append(frame)
        self.frames_forwarded += 1

    def collect(self, node_id: str) -> List[TTFrame]:
        """Retrieve all frames delivered to *node_id* and clear the buffer."""
        buf = self._output_buffers.get(node_id)
        if not buf:
            return []
        frames = list(buf)
        buf.clear()
        return frames

    @property
    def guardian_blocked(self) -> int:
        return self._guardian.blocked_count

    def status(self) -> Dict:
        return {
            "switch_id": self.switch_id,
            "healthy": self.healthy,
            "tt_pending": len(self._tt_queue),
            "rc_pending": len(self._rc_queue),
            "be_pending": len(self._be_queue),
            "forwarded": self.frames_forwarded,
            "dropped": self.frames_dropped,
            "guardian_blocked": self._guardian.blocked_count,
        }


# ---------------------------------------------------------------------------
# TTEthernetNetwork -- redundant multi-switch topology
# ---------------------------------------------------------------------------

class TTEthernetNetwork:
    """
    Dual/triplex redundant TTEthernet network.

    Features
    --------
    * Multiple independent switches forming redundant lanes (A, B, C).
    * Automatic failover with zero additional latency for TT frames
      (frames are sent simultaneously on every healthy lane; the
      receiver picks the first valid copy).
    * Safety-core / AI-core traffic isolation enforced by VLAN-like
      domain tagging.
    * Fault injection for testing.

    Parameters
    ----------
    redundancy : int
        Number of parallel lanes (2 = dual, 3 = triplex).
    """

    # Traffic domains for isolation
    DOMAIN_SAFETY = "safety_core"
    DOMAIN_AI = "ai_core"

    def __init__(
        self,
        tt_cycle_us: float = 500.0,
        rc_bandwidth_mbps: float = 100.0,
        be_bandwidth_mbps: float = 1000.0,
        redundancy: int = 3,
        guardian_enabled: bool = True,
    ):
        self.redundancy = redundancy
        self.tt_cycle_us = tt_cycle_us

        # One switch per redundant lane
        self._switches: List[TTEthernetSwitch] = []
        for idx in range(redundancy):
            lane = chr(ord("A") + idx)
            sw = TTEthernetSwitch(
                switch_id=f"SW_{lane}",
                tt_cycle_us=tt_cycle_us,
                rc_bandwidth_mbps=rc_bandwidth_mbps,
                be_bandwidth_mbps=be_bandwidth_mbps,
                guardian_enabled=guardian_enabled,
            )
            self._switches.append(sw)

        # Simulated global clock (microseconds)
        self._clock_us: float = 0.0

        # Node -> domain mapping for traffic isolation
        self._node_domains: Dict[str, str] = {}

        # Faulted links: (switch_index, node_id) pairs that are severed
        self._faulted_links: set = set()

        # Sequence numbers for frame deduplication at the receiver
        self._seen_frames: Dict[str, set] = {}  # node_id -> set of frame_ids

        # Statistics
        self.total_sent: int = 0
        self.total_received: int = 0
        self.failover_events: int = 0

        logger.info(
            "TTEthernetNetwork initialised: %d-redundant, TT cycle %d us, "
            "guardian %s",
            redundancy,
            int(tt_cycle_us),
            "ON" if guardian_enabled else "OFF",
        )

    # -- configuration ------------------------------------------------------

    def register_node(self, node_id: str, domain: str) -> None:
        """
        Register a network node and assign it to a traffic domain.

        Parameters
        ----------
        domain : str
            One of ``TTEthernetNetwork.DOMAIN_SAFETY`` or
            ``TTEthernetNetwork.DOMAIN_AI``.
        """
        self._node_domains[node_id] = domain
        self._seen_frames[node_id] = set()

    def add_tt_schedule(self, frame_id: int, dispatch_phase_us: float) -> None:
        """Add a TT schedule entry to every redundant switch."""
        for sw in self._switches:
            sw.add_tt_schedule_entry(frame_id, dispatch_phase_us)

    def register_guardian_window(
        self, source: str, start_us: float, end_us: float
    ) -> None:
        """Register guardian windows on all switches."""
        for sw in self._switches:
            sw.register_guardian_window(source, start_us, end_us)

    # -- traffic isolation check --------------------------------------------

    def _domains_compatible(self, source: str, destination: str) -> bool:
        """
        Enforce traffic isolation: safety-core nodes cannot receive
        from AI-core nodes and vice-versa unless both are in the same
        domain or one of them is unregistered (permissive default for
        housekeeping traffic).
        """
        src_dom = self._node_domains.get(source)
        dst_dom = self._node_domains.get(destination)
        if src_dom is None or dst_dom is None:
            return True  # permissive for unregistered nodes
        return src_dom == dst_dom

    # -- send / receive -----------------------------------------------------

    def send(self, frame: TTFrame) -> bool:
        """
        Transmit a frame across all healthy redundant lanes simultaneously.

        Returns True if the frame was accepted by at least one lane.
        """
        if not self._domains_compatible(frame.source, frame.destination):
            logger.warning(
                "Traffic isolation violation: %s (%s) -> %s (%s) -- dropped",
                frame.source,
                self._node_domains.get(frame.source, "?"),
                frame.destination,
                self._node_domains.get(frame.destination, "?"),
            )
            return False

        accepted = False
        for idx, sw in enumerate(self._switches):
            if (idx, frame.source) in self._faulted_links:
                continue
            if (idx, frame.destination) in self._faulted_links:
                continue
            if sw.ingest(frame, self._clock_us):
                accepted = True

        if accepted:
            self.total_sent += 1
        return accepted

    def receive(self, node_id: str) -> List[TTFrame]:
        """
        Collect frames for *node_id* from all healthy lanes, deduplicate
        (keep first copy only -- zero switchover delay for TT traffic).
        """
        frames: List[TTFrame] = []
        seen_ids = self._seen_frames.get(node_id, set())

        for idx, sw in enumerate(self._switches):
            if (idx, node_id) in self._faulted_links:
                continue
            for frame in sw.collect(node_id):
                if frame.frame_id not in seen_ids:
                    seen_ids.add(frame.frame_id)
                    frames.append(frame)
                else:
                    # Duplicate from redundant lane -- evidence of working
                    # failover path.  Count the first duplicate per id.
                    pass

        self.total_received += len(frames)
        # Clear dedup set periodically to avoid unbounded growth
        if len(seen_ids) > 10_000:
            seen_ids.clear()
        return frames

    # -- clock advance & dispatch -------------------------------------------

    def tick(self, dt_us: Optional[float] = None) -> None:
        """
        Advance the network clock by *dt_us* microseconds (default =
        one TT cycle) and dispatch all switches.
        """
        if dt_us is None:
            dt_us = self.tt_cycle_us
        self._clock_us += dt_us
        for sw in self._switches:
            sw.dispatch(self._clock_us)

    @property
    def clock_us(self) -> float:
        return self._clock_us

    # -- fault injection ----------------------------------------------------

    def inject_fault(
        self,
        switch_id: Optional[str] = None,
        link_id: Optional[Tuple[int, str]] = None,
    ) -> None:
        """
        Inject a fault for testing.

        Parameters
        ----------
        switch_id : str, optional
            Mark the entire switch as unhealthy (e.g. ``"SW_A"``).
        link_id : tuple(int, str), optional
            Sever a specific link between switch *index* and a *node_id*,
            e.g. ``(0, "flight_control")``.
        """
        if switch_id is not None:
            for sw in self._switches:
                if sw.switch_id == switch_id:
                    sw.healthy = False
                    self.failover_events += 1
                    logger.warning("Fault injected: switch %s disabled", switch_id)
                    return
            logger.error("Switch %s not found for fault injection", switch_id)

        if link_id is not None:
            self._faulted_links.add(link_id)
            self.failover_events += 1
            logger.warning(
                "Fault injected: link between switch %d and node '%s' severed",
                link_id[0],
                link_id[1],
            )

    def clear_faults(self) -> None:
        """Remove all injected faults and restore healthy state."""
        for sw in self._switches:
            sw.healthy = True
        self._faulted_links.clear()
        logger.info("All network faults cleared")

    # -- health monitoring --------------------------------------------------

    def health(self) -> Dict:
        """Return overall network health summary."""
        switch_statuses = [sw.status() for sw in self._switches]
        healthy_lanes = sum(1 for sw in self._switches if sw.healthy)
        return {
            "clock_us": self._clock_us,
            "redundancy": self.redundancy,
            "healthy_lanes": healthy_lanes,
            "degraded": healthy_lanes < self.redundancy,
            "failed": healthy_lanes == 0,
            "total_sent": self.total_sent,
            "total_received": self.total_received,
            "failover_events": self.failover_events,
            "faulted_links": len(self._faulted_links),
            "switches": switch_statuses,
        }
