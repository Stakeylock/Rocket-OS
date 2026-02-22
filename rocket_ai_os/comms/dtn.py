"""
Disruption-Tolerant Networking (DTN) Bundle Protocol Agent.

Implements a store-and-forward overlay network for intermittent deep-space links:
- Bundle Protocol (RFC 5050 / BPv7 concepts simplified)
- Custody Transfer with retransmission guarantees
- Non-volatile storage simulation for in-flight buffering
- Link-opportunity detection to flush queued bundles

References:
    - RFC 5050 – Bundle Protocol Specification
    - CCSDS 734.2-B-1 – CCSDS Bundle Protocol Specification
    - NASA DTN implementation (ION / DTNME)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BundlePriority(Enum):
    """Bundle forwarding priority classes (CCSDS CoS)."""
    BULK = auto()        # Best-effort background
    NORMAL = auto()      # Standard telemetry
    EXPEDITED = auto()   # Critical commands / safety data


class BundleStatus(Enum):
    """Lifecycle status of a single bundle."""
    CREATED = auto()
    STORED = auto()
    FORWARDED = auto()
    DELIVERED = auto()
    CUSTODY_ACCEPTED = auto()
    EXPIRED = auto()
    FAILED = auto()


class CustodySignalType(Enum):
    """Custody transfer signal type."""
    ACCEPTANCE = auto()
    REFUSAL = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bundle:
    """
    Protocol data unit of the Bundle Protocol.

    Attributes
    ----------
    bundle_id : str
        Globally unique bundle identifier.
    source : str
        Source endpoint identifier (EID).
    destination : str
        Destination EID.
    payload : bytes
        Opaque application data.
    creation_time : float
        Monotonic creation timestamp (seconds).
    expiry_time : float
        Absolute expiry timestamp (seconds).  After this the bundle is dropped.
    custody_accepted : bool
        Whether custody has been accepted by the local node.
    priority : BundlePriority
        Forwarding class of service.
    """
    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    destination: str = ""
    payload: bytes = b""
    creation_time: float = field(default_factory=time.monotonic)
    expiry_time: float = 0.0
    custody_accepted: bool = False
    priority: BundlePriority = BundlePriority.NORMAL

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Return True if the bundle's TTL has elapsed."""
        if self.expiry_time <= 0:
            return False  # no expiry set
        now = now if now is not None else time.monotonic()
        return now >= self.expiry_time

    @property
    def payload_size(self) -> int:
        return len(self.payload)


@dataclass
class CustodySignal:
    """Signal acknowledging (or refusing) custody of a bundle."""
    bundle_id: str
    signal_type: CustodySignalType
    reason: str = ""
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Custody Transfer Manager
# ---------------------------------------------------------------------------

class CustodyTransfer:
    """
    Manages custody acceptance / refusal and retransmission timers.

    When a node accepts custody it becomes responsible for reliable delivery
    until the next custodian acknowledges.  If no acknowledgement arrives
    within the retransmission interval, the bundle is re-sent.

    Parameters
    ----------
    retransmission_interval : float
        Seconds to wait before retransmitting an unacknowledged bundle.
    max_retransmissions : int
        Maximum retransmission attempts before declaring failure.
    """

    def __init__(
        self,
        retransmission_interval: float = 10.0,
        max_retransmissions: int = 5,
    ) -> None:
        self._retx_interval = retransmission_interval
        self._max_retx = max_retransmissions

        # Custody records: bundle_id -> metadata
        self._custody_bundles: Dict[str, Dict] = {}
        # Signals received
        self._signals: List[CustodySignal] = []

    # -- public API ----------------------------------------------------------

    def accept_custody(self, bundle: Bundle) -> CustodySignal:
        """
        Accept custody of a bundle.

        Returns a CUSTODY_ACCEPTED signal to send upstream.
        """
        bundle.custody_accepted = True
        self._custody_bundles[bundle.bundle_id] = {
            "bundle": bundle,
            "accepted_at": time.monotonic(),
            "last_sent": time.monotonic(),
            "retx_count": 0,
            "released": False,
        }
        signal = CustodySignal(
            bundle_id=bundle.bundle_id,
            signal_type=CustodySignalType.ACCEPTANCE,
        )
        self._signals.append(signal)
        return signal

    def release_custody(self, bundle_id: str) -> Optional[CustodySignal]:
        """
        Release custody after downstream acknowledgement.

        Returns the received signal, or None if bundle_id unknown.
        """
        record = self._custody_bundles.get(bundle_id)
        if record is None:
            return None
        record["released"] = True
        signal = CustodySignal(
            bundle_id=bundle_id,
            signal_type=CustodySignalType.ACCEPTANCE,
            reason="downstream custody accepted",
        )
        self._signals.append(signal)
        return signal

    def refuse_custody(self, bundle_id: str, reason: str = "") -> CustodySignal:
        """Generate a custody refusal signal."""
        signal = CustodySignal(
            bundle_id=bundle_id,
            signal_type=CustodySignalType.REFUSAL,
            reason=reason,
        )
        self._signals.append(signal)
        return signal

    def get_bundles_needing_retransmission(self, now: Optional[float] = None) -> List[Bundle]:
        """
        Return bundles whose retransmission timer has expired.

        Each call increments the retransmission counter.  Bundles that
        exceed ``max_retransmissions`` are marked as failed and excluded.
        """
        now = now if now is not None else time.monotonic()
        retx_list: List[Bundle] = []

        for bid, record in list(self._custody_bundles.items()):
            if record["released"]:
                continue
            bundle: Bundle = record["bundle"]
            if bundle.is_expired(now):
                record["released"] = True
                continue
            if record["retx_count"] >= self._max_retx:
                record["released"] = True  # give up
                continue
            elapsed = now - record["last_sent"]
            if elapsed >= self._retx_interval:
                record["last_sent"] = now
                record["retx_count"] += 1
                retx_list.append(bundle)

        return retx_list

    def has_custody(self, bundle_id: str) -> bool:
        record = self._custody_bundles.get(bundle_id)
        if record is None:
            return False
        return not record["released"]

    @property
    def custody_count(self) -> int:
        """Number of bundles currently held in custody."""
        return sum(
            1 for r in self._custody_bundles.values() if not r["released"]
        )

    @property
    def signals(self) -> List[CustodySignal]:
        return list(self._signals)


# ---------------------------------------------------------------------------
# Bundle Protocol Agent
# ---------------------------------------------------------------------------

class BundleProtocolAgent:
    """
    Store-and-forward Bundle Protocol Agent.

    Provides:
    - Non-volatile storage simulation (in-memory with capacity limit)
    - Link-opportunity detection and queued bundle flushing
    - Custody Transfer integration
    - Priority-aware forwarding

    Parameters
    ----------
    local_eid : str
        The endpoint ID of this node (e.g., ``"dtn://rocket-1"``).
    storage_capacity_bytes : int
        Maximum aggregate payload bytes that can be stored.
    custody_transfer : CustodyTransfer or None
        If provided, custody tracking is enabled.
    rng_seed : int
        For any probabilistic simulation (link reliability, etc.).
    """

    def __init__(
        self,
        local_eid: str = "dtn://rocket-1",
        storage_capacity_bytes: int = 10 * 1024 * 1024,  # 10 MB
        custody_transfer: Optional[CustodyTransfer] = None,
        rng_seed: int = 42,
    ) -> None:
        self._local_eid = local_eid
        self._storage_capacity = storage_capacity_bytes
        self._custody = custody_transfer or CustodyTransfer()
        self._rng = np.random.default_rng(rng_seed)

        # Bundle storage: destination -> priority-sorted list
        self._store: Dict[str, List[Bundle]] = {}
        self._stored_bytes: int = 0

        # Delivered bundles (for local consumption)
        self._inbox: List[Bundle] = []

        # Known reachable destinations (link-up map)
        self._link_up: Dict[str, bool] = {}

        # Event log
        self._event_log: List[Dict] = []

    # -- properties ----------------------------------------------------------

    @property
    def local_eid(self) -> str:
        return self._local_eid

    @property
    def stored_bytes(self) -> int:
        return self._stored_bytes

    @property
    def storage_capacity(self) -> int:
        return self._storage_capacity

    @property
    def storage_utilisation(self) -> float:
        """Fraction [0,1] of storage used."""
        if self._storage_capacity <= 0:
            return 1.0
        return self._stored_bytes / self._storage_capacity

    @property
    def event_log(self) -> List[Dict]:
        return list(self._event_log)

    # -- internal helpers ----------------------------------------------------

    def _log_event(self, event: str, bundle_id: str = "", detail: str = "") -> None:
        self._event_log.append({
            "timestamp": time.monotonic(),
            "event": event,
            "bundle_id": bundle_id,
            "detail": detail,
        })

    def _priority_key(self, bundle: Bundle) -> int:
        """Sort key: lower value = higher priority (expedited first)."""
        order = {
            BundlePriority.EXPEDITED: 0,
            BundlePriority.NORMAL: 1,
            BundlePriority.BULK: 2,
        }
        return order.get(bundle.priority, 99)

    def _evict_expired(self, now: Optional[float] = None) -> int:
        """Remove expired bundles from the store. Return count evicted."""
        now = now if now is not None else time.monotonic()
        evicted = 0
        for dest in list(self._store.keys()):
            remaining: List[Bundle] = []
            for b in self._store[dest]:
                if b.is_expired(now):
                    self._stored_bytes -= b.payload_size
                    evicted += 1
                    self._log_event("expired", b.bundle_id)
                else:
                    remaining.append(b)
            if remaining:
                self._store[dest] = remaining
            else:
                del self._store[dest]
        return evicted

    def _store_bundle(self, bundle: Bundle) -> bool:
        """
        Persist a bundle to the non-volatile store.

        Returns False if storage capacity is exceeded (after eviction).
        """
        self._evict_expired()

        if self._stored_bytes + bundle.payload_size > self._storage_capacity:
            self._log_event("storage_full", bundle.bundle_id,
                            f"need {bundle.payload_size}, avail "
                            f"{self._storage_capacity - self._stored_bytes}")
            return False

        dest = bundle.destination
        if dest not in self._store:
            self._store[dest] = []
        self._store[dest].append(bundle)
        self._store[dest].sort(key=self._priority_key)
        self._stored_bytes += bundle.payload_size
        self._log_event("stored", bundle.bundle_id, f"dest={dest}")
        return True

    # -- public API ----------------------------------------------------------

    def send_bundle(self, bundle: Bundle) -> Dict[str, object]:
        """
        Submit a bundle for transmission.

        If the destination link is up, the bundle is forwarded immediately;
        otherwise it is stored for later forwarding.

        Returns
        -------
        status : dict
            ``bundle_id`` : str
            ``status``    : :class:`BundleStatus`
            ``detail``    : str
        """
        bundle.source = bundle.source or self._local_eid

        # Accept custody if applicable
        self._custody.accept_custody(bundle)

        # Check if link to destination is available
        dest = bundle.destination
        if self._link_up.get(dest, False):
            self._log_event("forwarded", bundle.bundle_id, f"dest={dest}")
            return {
                "bundle_id": bundle.bundle_id,
                "status": BundleStatus.FORWARDED,
                "detail": f"Forwarded immediately to {dest}",
            }

        # Store-and-forward
        stored = self._store_bundle(bundle)
        if stored:
            return {
                "bundle_id": bundle.bundle_id,
                "status": BundleStatus.STORED,
                "detail": f"Stored pending link to {dest}",
            }
        else:
            return {
                "bundle_id": bundle.bundle_id,
                "status": BundleStatus.FAILED,
                "detail": "Storage capacity exceeded",
            }

    def receive_bundle(self) -> List[Bundle]:
        """
        Retrieve bundles delivered to this node (local inbox).

        Drains the inbox and returns all pending bundles.
        """
        delivered = list(self._inbox)
        self._inbox.clear()
        return delivered

    def deliver_to_local(self, bundle: Bundle) -> None:
        """
        Simulate receiving a bundle from a remote node.

        If the bundle is addressed to us it goes to the inbox;
        otherwise it is stored for forwarding (transit node behaviour).
        """
        if bundle.is_expired():
            self._log_event("expired_on_receive", bundle.bundle_id)
            return

        if bundle.destination == self._local_eid:
            self._inbox.append(bundle)
            self._log_event("delivered", bundle.bundle_id)
            # Release custody upstream
            self._custody.release_custody(bundle.bundle_id)
        else:
            # Transit: store for next hop
            self._custody.accept_custody(bundle)
            self._store_bundle(bundle)

    def set_link_state(self, destination: str, is_up: bool) -> None:
        """Inform the agent that the link to *destination* is up or down."""
        was_up = self._link_up.get(destination, False)
        self._link_up[destination] = is_up

        if is_up and not was_up:
            self._log_event("link_up", detail=f"dest={destination}")
            # Flush stored bundles
            self.on_link_available(destination)
        elif not is_up and was_up:
            self._log_event("link_down", detail=f"dest={destination}")

    def on_link_available(self, destination: str) -> List[Bundle]:
        """
        Flush all stored bundles destined for *destination*.

        Called automatically when a link comes up, or can be called manually.

        Returns the list of bundles forwarded.
        """
        self._evict_expired()
        bundles_to_send = self._store.pop(destination, [])
        forwarded: List[Bundle] = []

        for bundle in bundles_to_send:
            self._stored_bytes -= bundle.payload_size
            self._log_event("forwarded", bundle.bundle_id,
                            f"flushed to {destination}")
            # Release custody once forwarded
            self._custody.release_custody(bundle.bundle_id)
            forwarded.append(bundle)

        return forwarded

    def get_pending_bundles(self) -> List[Bundle]:
        """Return a flat list of all stored (not yet forwarded) bundles."""
        self._evict_expired()
        result: List[Bundle] = []
        for dest_bundles in self._store.values():
            result.extend(dest_bundles)
        result.sort(key=self._priority_key)
        return result

    def get_pending_by_destination(self) -> Dict[str, List[Bundle]]:
        """Return stored bundles grouped by destination EID."""
        self._evict_expired()
        return {dest: list(bl) for dest, bl in self._store.items()}

    def process_custody_retransmissions(self, now: Optional[float] = None) -> List[Bundle]:
        """
        Check custody timers and retransmit overdue bundles.

        Should be called periodically from the comms scheduler.
        """
        retx = self._custody.get_bundles_needing_retransmission(now)
        for b in retx:
            self._log_event("retransmit", b.bundle_id,
                            f"custody retx to {b.destination}")
        return retx
