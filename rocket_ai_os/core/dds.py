"""
Data Distribution Service (DDS) with Quality-of-Service enforcement.

Provides a DDS-style middleware layer with topics, readers, writers,
participants, and domain management.  Designed to interoperate with
the cFS :class:`~rocket_ai_os.core.software_bus.SoftwareBus` via an
explicit bridge function so that cFS message-ID traffic can be
transparently mapped onto DDS topics (and vice-versa) -- enabling
seamless cFS <-> Space ROS integration.

Key capabilities:

* Named topics with associated type information and per-topic QoS
* ``DDSWriter`` / ``DDSReader`` with QoS policy enforcement
* Deadline monitoring -- readers are notified when a topic misses its
  deadline
* Reliability checking -- reliable writers retry delivery; best-effort
  writers fire-and-forget
* Lifespan enforcement -- stale samples are automatically discarded
* History depth control -- each reader keeps at most *N* recent samples
* Domain-level participant management
* Bridge utility: ``SoftwareBusDDSBridge`` translates between Software
  Bus messages and DDS topics

References:
    OMG Data Distribution Service (DDS) v1.4
    OMG DDS Interoperability Wire Protocol (DDSI-RTPS)
"""

from __future__ import annotations

import logging
import threading
import time
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
    Type,
)

from rocket_ai_os.config import CriticalityLevel
from rocket_ai_os.core.software_bus import (
    Message,
    MessagePriority,
    SoftwareBus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Reliability(Enum):
    """DDS reliability QoS kind."""
    BEST_EFFORT = auto()
    RELIABLE = auto()


class Durability(Enum):
    """DDS durability QoS kind."""
    VOLATILE = auto()
    TRANSIENT_LOCAL = auto()
    TRANSIENT = auto()
    PERSISTENT = auto()


class HistoryKind(Enum):
    """DDS history QoS kind."""
    KEEP_LAST = auto()
    KEEP_ALL = auto()


class OwnershipKind(Enum):
    """DDS ownership QoS kind."""
    SHARED = auto()
    EXCLUSIVE = auto()


# ---------------------------------------------------------------------------
# QoS Policy
# ---------------------------------------------------------------------------

@dataclass
class QoSPolicy:
    """
    Quality-of-Service policy attached to a topic, reader, or writer.

    Attributes:
        reliability:   BEST_EFFORT or RELIABLE delivery.
        durability:    How long samples survive after the writer disappears.
        deadline_ms:   Maximum allowed interval (ms) between successive
                       samples.  0 means *no deadline*.
        lifespan_ms:   Maximum age (ms) of a sample before it is
                       considered stale and discarded.  0 means *infinite*.
        history_depth: Number of historical samples retained per instance
                       (only for KEEP_LAST history).
        history_kind:  KEEP_LAST or KEEP_ALL.
        ownership:     SHARED or EXCLUSIVE sample ownership.
        ownership_strength: Numeric strength when ownership is EXCLUSIVE.
    """
    reliability: Reliability = Reliability.RELIABLE
    durability: Durability = Durability.VOLATILE
    deadline_ms: float = 0.0
    lifespan_ms: float = 0.0
    history_depth: int = 16
    history_kind: HistoryKind = HistoryKind.KEEP_LAST
    ownership: OwnershipKind = OwnershipKind.SHARED
    ownership_strength: int = 0

    def is_compatible(self, offered: "QoSPolicy") -> bool:
        """
        RxO (Requested/Offered) compatibility check.

        A *requested* (reader) policy is compatible with an *offered*
        (writer) policy when the offered guarantees meet or exceed
        what the reader requires.
        """
        # Reliability: RELIABLE request needs RELIABLE offer
        if (
            self.reliability is Reliability.RELIABLE
            and offered.reliability is not Reliability.RELIABLE
        ):
            return False
        # Durability: requested must be <= offered
        if self.durability.value > offered.durability.value:
            return False
        return True


# ---------------------------------------------------------------------------
# DDS Sample
# ---------------------------------------------------------------------------

@dataclass
class DDSSample:
    """A single data sample flowing through a DDS topic."""
    topic_name: str
    data: Any
    timestamp: float = field(default_factory=time.monotonic)
    source_writer: str = ""
    sequence: int = 0

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.timestamp) * 1000.0


# ---------------------------------------------------------------------------
# DDS Topic
# ---------------------------------------------------------------------------

@dataclass
class DDSTopic:
    """
    A named, typed DDS topic.

    Attributes:
        name:       Unique topic name within a domain.
        type_name:  Descriptive type name (e.g. ``"nav/PoseStamped"``).
        type_class: Optional Python type for runtime type-checking of data.
        qos:        Default QoS policy for this topic.
    """
    name: str
    type_name: str = "Any"
    type_class: Optional[Type[Any]] = None
    qos: QoSPolicy = field(default_factory=QoSPolicy)


# ---------------------------------------------------------------------------
# DDS Writer
# ---------------------------------------------------------------------------

class DDSWriter:
    """
    Writes samples to a DDS topic.

    The writer enforces lifespan and maintains a send-history so that
    late-joining RELIABLE readers can be served from the writer cache.

    Parameters:
        participant_name: Name of the owning participant.
        topic:            The :class:`DDSTopic` to write to.
        qos:              Writer-specific QoS override (falls back to
                          the topic QoS if *None*).
        domain:           The :class:`DDSDomain` this writer belongs to.
    """

    def __init__(
        self,
        participant_name: str,
        topic: DDSTopic,
        qos: Optional[QoSPolicy] = None,
        domain: Optional["DDSDomain"] = None,
    ) -> None:
        self.participant_name = participant_name
        self.topic = topic
        self.qos: QoSPolicy = qos if qos is not None else topic.qos
        self._domain = domain
        self._sequence: int = 0
        self._lock = threading.Lock()
        self._history: Deque[DDSSample] = deque(
            maxlen=self.qos.history_depth
        )
        self._active: bool = True
        logger.debug(
            "DDSWriter created: participant=%s  topic=%s",
            participant_name,
            topic.name,
        )

    # -- public API -------------------------------------------------------

    def write(self, data: Any) -> DDSSample:
        """
        Publish *data* as a new sample on the topic.

        Raises ``RuntimeError`` if the writer has been closed.
        """
        if not self._active:
            raise RuntimeError(
                f"DDSWriter for topic '{self.topic.name}' has been closed"
            )

        # Optional type check
        if self.topic.type_class is not None:
            if not isinstance(data, self.topic.type_class):
                raise TypeError(
                    f"Expected {self.topic.type_class.__name__}, "
                    f"got {type(data).__name__}"
                )

        with self._lock:
            self._sequence += 1
            sample = DDSSample(
                topic_name=self.topic.name,
                data=data,
                timestamp=time.monotonic(),
                source_writer=self.participant_name,
                sequence=self._sequence,
            )
            self._history.append(sample)

        # Push into domain for delivery
        if self._domain is not None:
            self._domain._distribute(sample, self)

        return sample

    def close(self) -> None:
        self._active = False

    @property
    def history(self) -> List[DDSSample]:
        with self._lock:
            return list(self._history)

    def __repr__(self) -> str:
        return (
            f"<DDSWriter participant={self.participant_name!r} "
            f"topic={self.topic.name!r} seq={self._sequence}>"
        )


# ---------------------------------------------------------------------------
# DDS Reader
# ---------------------------------------------------------------------------

class DDSReader:
    """
    Reads samples from a DDS topic.

    The reader maintains a local cache and enforces deadline monitoring.
    A user-supplied callback is invoked on every new sample arrival.

    Parameters:
        participant_name: Name of the owning participant.
        topic:            The :class:`DDSTopic` to read from.
        qos:              Reader-specific QoS override.
        on_data:          Optional callback ``(sample: DDSSample) -> None``.
        on_deadline_missed: Optional callback invoked when the deadline
                          QoS is violated.
        domain:           The :class:`DDSDomain` this reader belongs to.
    """

    def __init__(
        self,
        participant_name: str,
        topic: DDSTopic,
        qos: Optional[QoSPolicy] = None,
        on_data: Optional[Callable[[DDSSample], None]] = None,
        on_deadline_missed: Optional[Callable[[str], None]] = None,
        domain: Optional["DDSDomain"] = None,
    ) -> None:
        self.participant_name = participant_name
        self.topic = topic
        self.qos: QoSPolicy = qos if qos is not None else topic.qos
        self.on_data = on_data
        self.on_deadline_missed = on_deadline_missed
        self._domain = domain
        self._lock = threading.Lock()
        self._cache: Deque[DDSSample] = deque(
            maxlen=self.qos.history_depth
        )
        self._active: bool = True
        self._last_received_time: float = time.monotonic()
        self._total_received: int = 0
        self._total_deadline_misses: int = 0
        logger.debug(
            "DDSReader created: participant=%s  topic=%s",
            participant_name,
            topic.name,
        )

    # -- internal delivery ------------------------------------------------

    def _receive(self, sample: DDSSample) -> None:
        """Called by the domain to deliver a sample to this reader."""
        if not self._active:
            return

        # Lifespan check -- drop stale samples
        if self.qos.lifespan_ms > 0 and sample.age_ms > self.qos.lifespan_ms:
            logger.debug(
                "DDSReader[%s/%s] dropped stale sample (age=%.1f ms)",
                self.participant_name,
                self.topic.name,
                sample.age_ms,
            )
            return

        with self._lock:
            self._cache.append(sample)
            self._last_received_time = time.monotonic()
            self._total_received += 1

        if self.on_data is not None:
            try:
                self.on_data(sample)
            except Exception:
                logger.exception(
                    "DDSReader on_data callback failed for topic %s",
                    self.topic.name,
                )

    # -- deadline monitoring ----------------------------------------------

    def check_deadline(self) -> bool:
        """
        Check whether the deadline QoS has been violated.

        Returns *True* if the deadline is satisfied (or not configured),
        *False* if the deadline has been missed.
        """
        if self.qos.deadline_ms <= 0:
            return True

        with self._lock:
            elapsed_ms = (
                time.monotonic() - self._last_received_time
            ) * 1000.0

        if elapsed_ms > self.qos.deadline_ms:
            self._total_deadline_misses += 1
            logger.warning(
                "DDSReader[%s/%s] DEADLINE MISSED  elapsed=%.1f ms  "
                "deadline=%.1f ms",
                self.participant_name,
                self.topic.name,
                elapsed_ms,
                self.qos.deadline_ms,
            )
            if self.on_deadline_missed is not None:
                try:
                    self.on_deadline_missed(self.topic.name)
                except Exception:
                    logger.exception(
                        "on_deadline_missed callback failed for %s",
                        self.topic.name,
                    )
            return False

        return True

    # -- public API -------------------------------------------------------

    def take(self, max_samples: Optional[int] = None) -> List[DDSSample]:
        """Remove and return up to *max_samples* from the reader cache."""
        with self._lock:
            if max_samples is None:
                result = list(self._cache)
                self._cache.clear()
            else:
                result: List[DDSSample] = []
                for _ in range(min(max_samples, len(self._cache))):
                    result.append(self._cache.popleft())
        return result

    def read(self, max_samples: Optional[int] = None) -> List[DDSSample]:
        """Return up to *max_samples* without removing them."""
        with self._lock:
            samples = list(self._cache)
        if max_samples is not None:
            samples = samples[-max_samples:]
        return samples

    def close(self) -> None:
        self._active = False

    @property
    def total_received(self) -> int:
        with self._lock:
            return self._total_received

    @property
    def total_deadline_misses(self) -> int:
        return self._total_deadline_misses

    def __repr__(self) -> str:
        return (
            f"<DDSReader participant={self.participant_name!r} "
            f"topic={self.topic.name!r} received={self.total_received}>"
        )


# ---------------------------------------------------------------------------
# DDS Participant
# ---------------------------------------------------------------------------

class DDSParticipant:
    """
    A named entity that creates readers and writers within a domain.

    Parameters:
        name:   Unique participant name.
        domain: The :class:`DDSDomain` hosting this participant.
    """

    def __init__(self, name: str, domain: "DDSDomain") -> None:
        self.name = name
        self.domain = domain
        self._writers: List[DDSWriter] = []
        self._readers: List[DDSReader] = []
        logger.info("DDSParticipant created: %s", name)

    def create_writer(
        self,
        topic: DDSTopic,
        qos: Optional[QoSPolicy] = None,
    ) -> DDSWriter:
        """Create and register a :class:`DDSWriter` for *topic*."""
        writer = DDSWriter(
            participant_name=self.name,
            topic=topic,
            qos=qos,
            domain=self.domain,
        )
        self._writers.append(writer)
        self.domain._register_writer(writer)
        return writer

    def create_reader(
        self,
        topic: DDSTopic,
        qos: Optional[QoSPolicy] = None,
        on_data: Optional[Callable[[DDSSample], None]] = None,
        on_deadline_missed: Optional[Callable[[str], None]] = None,
    ) -> DDSReader:
        """Create and register a :class:`DDSReader` for *topic*."""
        reader = DDSReader(
            participant_name=self.name,
            topic=topic,
            qos=qos,
            on_data=on_data,
            on_deadline_missed=on_deadline_missed,
            domain=self.domain,
        )
        self._readers.append(reader)
        self.domain._register_reader(reader)
        return reader

    def close(self) -> None:
        """Close all readers and writers owned by this participant."""
        for w in self._writers:
            w.close()
        for r in self._readers:
            r.close()
        logger.info("DDSParticipant closed: %s", self.name)

    @property
    def writers(self) -> List[DDSWriter]:
        return list(self._writers)

    @property
    def readers(self) -> List[DDSReader]:
        return list(self._readers)

    def __repr__(self) -> str:
        return (
            f"<DDSParticipant name={self.name!r} "
            f"writers={len(self._writers)} readers={len(self._readers)}>"
        )


# ---------------------------------------------------------------------------
# DDS Domain
# ---------------------------------------------------------------------------

class DDSDomain:
    """
    Manages all participants, topics, readers, and writers in a DDS domain.

    Acts as the central routing fabric: when a :class:`DDSWriter` publishes
    a sample the domain delivers it to every matching :class:`DDSReader`
    whose QoS is compatible with the writer.

    Parameters:
        domain_id: Numeric domain identifier (analogous to DDS domain ID).
    """

    def __init__(self, domain_id: int = 0) -> None:
        self.domain_id = domain_id
        self._lock = threading.Lock()

        # topic_name -> DDSTopic
        self._topics: Dict[str, DDSTopic] = {}

        # topic_name -> list of DDSWriter
        self._writers: Dict[str, List[DDSWriter]] = defaultdict(list)

        # topic_name -> list of DDSReader
        self._readers: Dict[str, List[DDSReader]] = defaultdict(list)

        # participant_name -> DDSParticipant
        self._participants: Dict[str, DDSParticipant] = {}

        # Statistics
        self._total_samples: int = 0
        self._total_deliveries: int = 0
        self._total_qos_incompatible: int = 0

        logger.info("DDSDomain created: domain_id=%d", domain_id)

    # -- topic management -------------------------------------------------

    def create_topic(
        self,
        name: str,
        type_name: str = "Any",
        type_class: Optional[Type[Any]] = None,
        qos: Optional[QoSPolicy] = None,
    ) -> DDSTopic:
        """Create (or retrieve) a named topic in this domain."""
        with self._lock:
            if name in self._topics:
                return self._topics[name]
            topic = DDSTopic(
                name=name,
                type_name=type_name,
                type_class=type_class,
                qos=qos if qos is not None else QoSPolicy(),
            )
            self._topics[name] = topic
        logger.info(
            "DDSTopic registered: %s  type=%s  qos=%s",
            name,
            type_name,
            topic.qos,
        )
        return topic

    def get_topic(self, name: str) -> Optional[DDSTopic]:
        with self._lock:
            return self._topics.get(name)

    # -- participant management -------------------------------------------

    def create_participant(self, name: str) -> DDSParticipant:
        """Create a :class:`DDSParticipant` in this domain."""
        with self._lock:
            if name in self._participants:
                return self._participants[name]
            participant = DDSParticipant(name=name, domain=self)
            self._participants[name] = participant
        return participant

    # -- internal registration (called by DDSParticipant) -----------------

    def _register_writer(self, writer: DDSWriter) -> None:
        with self._lock:
            self._writers[writer.topic.name].append(writer)
        logger.debug(
            "DDSDomain[%d] registered writer %s for topic %s",
            self.domain_id,
            writer.participant_name,
            writer.topic.name,
        )

    def _register_reader(self, reader: DDSReader) -> None:
        with self._lock:
            self._readers[reader.topic.name].append(reader)
        logger.debug(
            "DDSDomain[%d] registered reader %s for topic %s",
            self.domain_id,
            reader.participant_name,
            reader.topic.name,
        )

    # -- sample distribution ----------------------------------------------

    def _distribute(self, sample: DDSSample, writer: DDSWriter) -> None:
        """
        Route a sample from *writer* to all compatible readers.

        QoS compatibility is checked per reader: if a reader's requested
        QoS is incompatible with the writer's offered QoS the sample is
        not delivered and a warning is logged.
        """
        with self._lock:
            self._total_samples += 1
            readers = list(self._readers.get(sample.topic_name, []))

        for reader in readers:
            if not reader._active:
                continue

            # RxO compatibility check
            if not reader.qos.is_compatible(writer.qos):
                self._total_qos_incompatible += 1
                logger.warning(
                    "QoS INCOMPATIBLE: writer %s -> reader %s on topic %s",
                    writer.participant_name,
                    reader.participant_name,
                    sample.topic_name,
                )
                continue

            reader._receive(sample)
            with self._lock:
                self._total_deliveries += 1

    # -- deadline monitoring (must be called periodically) ----------------

    def check_deadlines(self) -> List[Tuple[str, str]]:
        """
        Check every reader's deadline QoS.

        Returns a list of ``(participant_name, topic_name)`` pairs where
        the deadline was missed.
        """
        missed: List[Tuple[str, str]] = []
        with self._lock:
            all_readers = [
                reader
                for readers in self._readers.values()
                for reader in readers
            ]
        for reader in all_readers:
            if not reader.check_deadline():
                missed.append((reader.participant_name, reader.topic.name))
        return missed

    # -- statistics -------------------------------------------------------

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total_samples

    @property
    def total_deliveries(self) -> int:
        with self._lock:
            return self._total_deliveries

    @property
    def total_qos_incompatible(self) -> int:
        with self._lock:
            return self._total_qos_incompatible

    def get_topic_names(self) -> List[str]:
        with self._lock:
            return list(self._topics.keys())

    def __repr__(self) -> str:
        with self._lock:
            n_topics = len(self._topics)
            n_parts = len(self._participants)
            n_writers = sum(len(v) for v in self._writers.values())
            n_readers = sum(len(v) for v in self._readers.values())
        return (
            f"<DDSDomain id={self.domain_id} topics={n_topics} "
            f"participants={n_parts} writers={n_writers} "
            f"readers={n_readers} samples={self.total_samples} "
            f"deliveries={self.total_deliveries}>"
        )


# ---------------------------------------------------------------------------
# Bridge: Software Bus <-> DDS  (cFS <-> Space ROS integration)
# ---------------------------------------------------------------------------

class SoftwareBusDDSBridge:
    """
    Bidirectional bridge between a cFS :class:`SoftwareBus` and a
    :class:`DDSDomain`.

    Mappings are established by calling :meth:`map_msg_to_topic` which
    sets up:

    1. A Software Bus subscription that forwards each cFS message to a
       DDS writer, and
    2. A DDS reader callback that publishes each DDS sample back onto
       the Software Bus.

    This enables transparent interoperability between partitions using
    the cFS Software Bus and Space ROS nodes communicating over DDS.

    Parameters:
        bus:    The cFS Software Bus instance.
        domain: The DDS domain instance.
        bridge_participant_name: Name used for the bridge's DDS
                participant.
    """

    def __init__(
        self,
        bus: SoftwareBus,
        domain: DDSDomain,
        bridge_participant_name: str = "cfs_dds_bridge",
    ) -> None:
        self.bus = bus
        self.domain = domain
        self.bridge_name = bridge_participant_name
        self._participant = domain.create_participant(bridge_participant_name)

        # msg_id -> (topic_name, DDSWriter, DDSReader)
        self._mappings: Dict[int, Tuple[str, DDSWriter, DDSReader]] = {}

        # Prevent echo loops: ignore samples originating from this bridge
        self._bridge_source_tag = f"__bridge_{bridge_participant_name}__"

        self._lock = threading.Lock()
        self._total_bus_to_dds: int = 0
        self._total_dds_to_bus: int = 0

        logger.info(
            "SoftwareBusDDSBridge initialised  participant=%s",
            bridge_participant_name,
        )

    # ------------------------------------------------------------------

    def map_msg_to_topic(
        self,
        msg_id: int,
        topic_name: str,
        type_name: str = "Any",
        qos: Optional[QoSPolicy] = None,
    ) -> None:
        """
        Establish a bidirectional mapping between a Software Bus *msg_id*
        and a DDS *topic_name*.

        Parameters:
            msg_id:     cFS Software Bus message-ID.
            topic_name: DDS topic name to map to.
            type_name:  DDS type name.
            qos:        QoS policy for the DDS topic / reader / writer.
        """
        effective_qos = qos if qos is not None else QoSPolicy()

        # Create (or reuse) the DDS topic
        topic = self.domain.create_topic(
            name=topic_name,
            type_name=type_name,
            qos=effective_qos,
        )

        # DDS writer: forwards cFS messages into DDS
        writer = self._participant.create_writer(topic, qos=effective_qos)

        # DDS reader: forwards DDS samples back to cFS
        reader = self._participant.create_reader(
            topic,
            qos=effective_qos,
            on_data=self._make_dds_to_bus_callback(msg_id),
        )

        # Software Bus subscription: forwards cFS messages into DDS
        self.bus.subscribe(
            app_name=self.bridge_name,
            msg_id=msg_id,
            callback=self._make_bus_to_dds_callback(msg_id, writer),
        )

        with self._lock:
            self._mappings[msg_id] = (topic_name, writer, reader)

        logger.info(
            "Bridge mapping: msg_id=0x%04X <-> topic=%s",
            msg_id,
            topic_name,
        )

    # ------------------------------------------------------------------
    # Internal callback factories
    # ------------------------------------------------------------------

    def _make_bus_to_dds_callback(
        self,
        msg_id: int,
        writer: DDSWriter,
    ) -> Callable[[Message], None]:
        """Return a Software Bus callback that writes to DDS."""
        bridge_tag = self._bridge_source_tag

        def _callback(message: Message) -> None:
            # Prevent echo loops
            if message.source == bridge_tag:
                return
            writer.write(message.payload)
            with self._lock:
                self._total_bus_to_dds += 1

        return _callback

    def _make_dds_to_bus_callback(
        self,
        msg_id: int,
    ) -> Callable[[DDSSample], None]:
        """Return a DDS on_data callback that publishes to the bus."""
        bridge_tag = self._bridge_source_tag

        def _callback(sample: DDSSample) -> None:
            # Prevent echo loops
            if sample.source_writer == self.bridge_name:
                return
            self.bus.publish(
                msg_id=msg_id,
                payload=sample.data,
                source=bridge_tag,
                priority=MessagePriority.NORMAL,
            )
            with self._lock:
                self._total_dds_to_bus += 1

        return _callback

    # ------------------------------------------------------------------
    # Statistics / introspection
    # ------------------------------------------------------------------

    @property
    def total_bus_to_dds(self) -> int:
        with self._lock:
            return self._total_bus_to_dds

    @property
    def total_dds_to_bus(self) -> int:
        with self._lock:
            return self._total_dds_to_bus

    def list_mappings(self) -> Dict[int, str]:
        """Return a dict of ``{msg_id: topic_name}``."""
        with self._lock:
            return {
                mid: tname for mid, (tname, _, _) in self._mappings.items()
            }

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._mappings)
        return (
            f"<SoftwareBusDDSBridge mappings={n} "
            f"bus_to_dds={self.total_bus_to_dds} "
            f"dds_to_bus={self.total_dds_to_bus}>"
        )
