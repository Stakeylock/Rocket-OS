"""
NASA cFS-style Software Bus implementation.

Provides a publish-subscribe message routing system modelled after the
NASA core Flight System (cFS) Software Bus.  Key capabilities:

* Named publish-subscribe messaging with message-ID routing
* Priority-aware delivery with configurable QoS per subscription
* Automatic failover -- if a primary publisher becomes unresponsive a
  registered backup seamlessly takes over
* Per-topic and aggregate statistics (published, delivered, dropped)
* Full message history / logging with configurable depth

References:
    NASA cFS Software Bus Users Guide
    CCSDS 133.0-B-2 (Space Packet Protocol)
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict, deque
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

from rocket_ai_os.config import CriticalityLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MessagePriority(Enum):
    """Message delivery priority levels (highest first)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BEST_EFFORT = 4


class QoSReliability(Enum):
    """Delivery guarantee expected by a subscriber."""
    BEST_EFFORT = auto()
    RELIABLE = auto()


class PublisherRole(Enum):
    """Role of a publisher for a given message-ID."""
    PRIMARY = auto()
    BACKUP = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Message:
    """
    A single Software Bus message.

    Fields are ordered so that messages sort by *priority* first (lower
    numerical value == higher priority) and then by *timestamp*.
    """
    priority: MessagePriority
    timestamp: float
    msg_id: int
    payload: Any = field(compare=False)
    source: str = field(compare=False, default="unknown")
    sequence: int = field(compare=False, default=0)

    # Convenience helpers ------------------------------------------------
    @property
    def age_ms(self) -> float:
        """Wall-clock age of this message in milliseconds."""
        return (time.monotonic() - self.timestamp) * 1000.0


@dataclass
class Subscription:
    """
    A single subscription entry on the Software Bus.

    Attributes:
        app_name:   Subscribing application name (partition / cFS app).
        msg_id:     The message-ID being subscribed to.
        callback:   Callable invoked on delivery -- ``callback(message)``.
        reliability: Requested QoS reliability level.
        priority_filter: If set, only messages at this priority or higher
                         are delivered.
        active:     Can be disabled without removing the subscription.
    """
    app_name: str
    msg_id: int
    callback: Callable[[Message], None]
    reliability: QoSReliability = QoSReliability.RELIABLE
    priority_filter: Optional[MessagePriority] = None
    active: bool = True


@dataclass
class PublisherInfo:
    """Tracks a registered publisher for failover management."""
    app_name: str
    msg_id: int
    role: PublisherRole
    last_publish_time: float = 0.0
    publish_count: int = 0
    healthy: bool = True


@dataclass
class MessageStats:
    """Aggregate statistics for a single message-ID."""
    published: int = 0
    delivered: int = 0
    dropped: int = 0
    last_publish_time: float = 0.0
    last_delivery_time: float = 0.0


# ---------------------------------------------------------------------------
# Software Bus
# ---------------------------------------------------------------------------

class SoftwareBus:
    """
    cFS-style Software Bus with publish-subscribe routing.

    Thread-safe: all public methods acquire an internal lock so the bus
    can be shared across RTOS partitions / threads.

    Parameters:
        history_depth:       Number of messages to retain per msg_id.
        failover_timeout_ms: If a PRIMARY publisher has not published for
                             this many milliseconds the bus promotes its
                             BACKUP to PRIMARY.
    """

    def __init__(
        self,
        history_depth: int = 256,
        failover_timeout_ms: float = 500.0,
    ) -> None:
        # Guards all mutable state
        self._lock = threading.Lock()

        # msg_id -> list of Subscription
        self._subscriptions: Dict[int, List[Subscription]] = defaultdict(list)

        # msg_id -> deque of Message (bounded history)
        self._history: Dict[int, Deque[Message]] = defaultdict(
            lambda: deque(maxlen=history_depth)
        )

        # msg_id -> list of PublisherInfo
        self._publishers: Dict[int, List[PublisherInfo]] = defaultdict(list)

        # msg_id -> MessageStats
        self._stats: Dict[int, MessageStats] = defaultdict(MessageStats)

        # Global monotonic sequence counter
        self._sequence: int = 0

        # Configuration
        self._history_depth = history_depth
        self._failover_timeout_ms = failover_timeout_ms

        # Aggregate counters
        self._total_published: int = 0
        self._total_delivered: int = 0
        self._total_dropped: int = 0

        logger.info(
            "SoftwareBus initialised  history_depth=%d  failover_timeout_ms=%.1f",
            history_depth,
            failover_timeout_ms,
        )

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        app_name: str,
        msg_id: int,
        callback: Callable[[Message], None],
        reliability: QoSReliability = QoSReliability.RELIABLE,
        priority_filter: Optional[MessagePriority] = None,
    ) -> Subscription:
        """
        Subscribe *app_name* to messages with *msg_id*.

        Returns the :class:`Subscription` handle which can later be passed
        to :meth:`unsubscribe`.
        """
        sub = Subscription(
            app_name=app_name,
            msg_id=msg_id,
            callback=callback,
            reliability=reliability,
            priority_filter=priority_filter,
        )
        with self._lock:
            self._subscriptions[msg_id].append(sub)
        logger.debug(
            "%-20s subscribed to msg_id=0x%04X  reliability=%s",
            app_name,
            msg_id,
            reliability.name,
        )
        return sub

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription.  Returns *True* if it was found."""
        with self._lock:
            subs = self._subscriptions.get(subscription.msg_id, [])
            try:
                subs.remove(subscription)
                logger.debug(
                    "%-20s unsubscribed from msg_id=0x%04X",
                    subscription.app_name,
                    subscription.msg_id,
                )
                return True
            except ValueError:
                return False

    # ------------------------------------------------------------------
    # Publisher registration & failover
    # ------------------------------------------------------------------

    def register_publisher(
        self,
        app_name: str,
        msg_id: int,
        role: PublisherRole = PublisherRole.PRIMARY,
    ) -> PublisherInfo:
        """
        Register *app_name* as a publisher for *msg_id*.

        Only one PRIMARY publisher is allowed per msg_id.  Additional
        publishers should register as BACKUP -- the bus will automatically
        promote the first healthy backup if the primary becomes
        unresponsive (see *failover_timeout_ms*).
        """
        info = PublisherInfo(app_name=app_name, msg_id=msg_id, role=role)
        with self._lock:
            existing = self._publishers[msg_id]

            # Enforce single PRIMARY
            if role is PublisherRole.PRIMARY:
                for pub in existing:
                    if pub.role is PublisherRole.PRIMARY and pub.healthy:
                        raise ValueError(
                            f"msg_id 0x{msg_id:04X} already has a healthy "
                            f"PRIMARY publisher ({pub.app_name})"
                        )

            existing.append(info)

        logger.info(
            "%-20s registered as %s publisher for msg_id=0x%04X",
            app_name,
            role.name,
            msg_id,
        )
        return info

    def _check_failover(self, msg_id: int) -> None:
        """
        Promote a BACKUP publisher to PRIMARY if the current PRIMARY has
        exceeded the failover timeout.

        Must be called with ``self._lock`` held.
        """
        publishers = self._publishers.get(msg_id, [])
        primary: Optional[PublisherInfo] = None
        backups: List[PublisherInfo] = []

        for pub in publishers:
            if pub.role is PublisherRole.PRIMARY and pub.healthy:
                primary = pub
            elif pub.role is PublisherRole.BACKUP and pub.healthy:
                backups.append(pub)

        if primary is None:
            # No healthy primary -- try to promote
            if backups:
                promoted = backups[0]
                promoted.role = PublisherRole.PRIMARY
                logger.warning(
                    "FAILOVER: promoted %s to PRIMARY for msg_id=0x%04X "
                    "(no healthy primary found)",
                    promoted.app_name,
                    msg_id,
                )
            return

        # Check timeout
        now = time.monotonic()
        age_ms = (now - primary.last_publish_time) * 1000.0
        if primary.last_publish_time > 0 and age_ms > self._failover_timeout_ms:
            primary.healthy = False
            primary.role = PublisherRole.BACKUP  # demote
            if backups:
                promoted = backups[0]
                promoted.role = PublisherRole.PRIMARY
                logger.warning(
                    "FAILOVER: %s timed out (%.1f ms) for msg_id=0x%04X  "
                    "-- promoted %s to PRIMARY",
                    primary.app_name,
                    age_ms,
                    msg_id,
                    promoted.app_name,
                )
            else:
                logger.error(
                    "FAILOVER: %s timed out for msg_id=0x%04X and no "
                    "BACKUP is available",
                    primary.app_name,
                    msg_id,
                )

    # ------------------------------------------------------------------
    # Publishing / delivery
    # ------------------------------------------------------------------

    def publish(
        self,
        msg_id: int,
        payload: Any,
        source: str = "unknown",
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Message:
        """
        Publish a message on the bus.

        The bus constructs a :class:`Message`, records it in the history,
        and delivers it to all matching subscribers.

        Returns the constructed :class:`Message`.
        """
        with self._lock:
            # Failover check before accepting the message
            self._check_failover(msg_id)

            # Update publisher bookkeeping
            self._update_publisher_stats(msg_id, source)

            # Build message
            self._sequence += 1
            now = time.monotonic()
            msg = Message(
                msg_id=msg_id,
                timestamp=now,
                payload=payload,
                source=source,
                priority=priority,
                sequence=self._sequence,
            )

            # Record in history
            self._history[msg_id].append(msg)

            # Update stats
            stats = self._stats[msg_id]
            stats.published += 1
            stats.last_publish_time = now
            self._total_published += 1

            # Gather matching subscribers (snapshot under lock)
            subscribers = list(self._subscriptions.get(msg_id, []))

        # Deliver outside the lock to avoid holding it during callbacks
        delivered, dropped = self._deliver(msg, subscribers)

        with self._lock:
            stats = self._stats[msg_id]
            stats.delivered += delivered
            stats.dropped += dropped
            if delivered:
                stats.last_delivery_time = time.monotonic()
            self._total_delivered += delivered
            self._total_dropped += dropped

        return msg

    def _update_publisher_stats(self, msg_id: int, source: str) -> None:
        """Update the bookkeeping for the publisher.  Lock must be held."""
        for pub in self._publishers.get(msg_id, []):
            if pub.app_name == source:
                pub.last_publish_time = time.monotonic()
                pub.publish_count += 1
                pub.healthy = True
                break

    def _deliver(
        self,
        msg: Message,
        subscribers: List[Subscription],
    ) -> Tuple[int, int]:
        """
        Deliver *msg* to each matching subscriber.

        Returns ``(delivered_count, dropped_count)``.
        """
        delivered = 0
        dropped = 0

        for sub in subscribers:
            if not sub.active:
                continue

            # Priority filter
            if sub.priority_filter is not None:
                if msg.priority.value > sub.priority_filter.value:
                    dropped += 1
                    continue

            try:
                sub.callback(msg)
                delivered += 1
            except Exception:
                logger.exception(
                    "Delivery to %s for msg_id=0x%04X raised an exception",
                    sub.app_name,
                    msg.msg_id,
                )
                if sub.reliability is QoSReliability.RELIABLE:
                    # For reliable QoS we count it as dropped so the
                    # stats reflect the delivery failure.
                    dropped += 1
                else:
                    # Best-effort: silently move on
                    delivered += 1

        return delivered, dropped

    # ------------------------------------------------------------------
    # History / logging queries
    # ------------------------------------------------------------------

    def get_history(
        self,
        msg_id: int,
        max_count: Optional[int] = None,
    ) -> List[Message]:
        """Return recent messages for *msg_id* (newest last)."""
        with self._lock:
            history = list(self._history.get(msg_id, []))
        if max_count is not None:
            history = history[-max_count:]
        return history

    def get_all_history(self) -> Dict[int, List[Message]]:
        """Return a snapshot of the full message history."""
        with self._lock:
            return {
                msg_id: list(msgs)
                for msg_id, msgs in self._history.items()
            }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, msg_id: int) -> MessageStats:
        """Return a copy of the stats for *msg_id*."""
        with self._lock:
            s = self._stats[msg_id]
            return MessageStats(
                published=s.published,
                delivered=s.delivered,
                dropped=s.dropped,
                last_publish_time=s.last_publish_time,
                last_delivery_time=s.last_delivery_time,
            )

    def get_all_stats(self) -> Dict[int, MessageStats]:
        """Return a snapshot of statistics for every msg_id."""
        with self._lock:
            return {
                mid: MessageStats(
                    published=s.published,
                    delivered=s.delivered,
                    dropped=s.dropped,
                    last_publish_time=s.last_publish_time,
                    last_delivery_time=s.last_delivery_time,
                )
                for mid, s in self._stats.items()
            }

    @property
    def total_published(self) -> int:
        with self._lock:
            return self._total_published

    @property
    def total_delivered(self) -> int:
        with self._lock:
            return self._total_delivered

    @property
    def total_dropped(self) -> int:
        with self._lock:
            return self._total_dropped

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_subscriptions(
        self,
        msg_id: Optional[int] = None,
    ) -> List[Subscription]:
        """List subscriptions, optionally filtered by *msg_id*."""
        with self._lock:
            if msg_id is not None:
                return list(self._subscriptions.get(msg_id, []))
            result: List[Subscription] = []
            for subs in self._subscriptions.values():
                result.extend(subs)
            return result

    def list_publishers(
        self,
        msg_id: Optional[int] = None,
    ) -> List[PublisherInfo]:
        """List registered publishers, optionally filtered by *msg_id*."""
        with self._lock:
            if msg_id is not None:
                return list(self._publishers.get(msg_id, []))
            result: List[PublisherInfo] = []
            for pubs in self._publishers.values():
                result.extend(pubs)
            return result

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all subscriptions, publishers, history, and stats."""
        with self._lock:
            self._subscriptions.clear()
            self._history.clear()
            self._publishers.clear()
            self._stats.clear()
            self._sequence = 0
            self._total_published = 0
            self._total_delivered = 0
            self._total_dropped = 0
        logger.info("SoftwareBus reset")

    def __repr__(self) -> str:
        with self._lock:
            n_subs = sum(len(v) for v in self._subscriptions.values())
            n_pubs = sum(len(v) for v in self._publishers.values())
            n_topics = len(self._stats)
        return (
            f"<SoftwareBus topics={n_topics} subscriptions={n_subs} "
            f"publishers={n_pubs} published={self.total_published} "
            f"delivered={self.total_delivered} dropped={self.total_dropped}>"
        )
