"""
Self-Healing Mesh Network with Software-Defined Networking (SDN) Controller.

Implements a multi-node mesh topology for intra-vehicle and vehicle-to-vehicle
communications with automatic rerouting around failed nodes.

Features:
- Distance-based link quality simulation
- Dijkstra shortest-path routing with link-quality weights
- SDN flow-rule management with priorities and expiry
- Automatic failure detection and self-healing reroute
- Dynamic node addition / removal

References:
    - OpenFlow SDN concepts (simplified for space application)
    - Wireless mesh networking for CubeSat swarms (NASA SCaN)
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeshNode:
    """
    A node in the mesh network.

    Attributes
    ----------
    node_id : str
        Unique node identifier.
    position : np.ndarray
        3-D position vector (metres).
    capabilities : dict
        Arbitrary capability map (e.g. ``{"relay": True, "gateway": False}``).
    active : bool
        Whether the node is currently operational.
    """
    node_id: str = ""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    capabilities: Dict[str, object] = field(default_factory=dict)
    active: bool = True

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeshNode):
            return NotImplemented
        return self.node_id == other.node_id


@dataclass
class SDNFlowRule:
    """
    A forwarding rule installed by the SDN controller.

    Attributes
    ----------
    source : str
        Source node ID.
    destination : str
        Destination node ID.
    next_hop : str
        Next-hop node ID from *source* towards *destination*.
    priority : int
        Higher priority rules take precedence.
    expiry : float
        Monotonic timestamp after which the rule is invalid (0 = never expires).
    """
    source: str = ""
    destination: str = ""
    next_hop: str = ""
    priority: int = 0
    expiry: float = 0.0

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expiry <= 0:
            return False
        now = now if now is not None else time.monotonic()
        return now >= self.expiry


@dataclass
class Packet:
    """A simple packet structure for mesh forwarding."""
    packet_id: str = ""
    source: str = ""
    destination: str = ""
    payload: bytes = b""
    ttl: int = 16
    route: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Link quality model
# ---------------------------------------------------------------------------

def _link_quality(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    max_range_m: float = 1000.0,
    path_loss_exp: float = 2.0,
) -> float:
    """
    Compute a link-quality metric in [0, 1] between two positions.

    Uses a simple inverse-power path-loss model.  Returns 0 when the
    distance exceeds *max_range_m*.

    Parameters
    ----------
    pos_a, pos_b : np.ndarray
        3-D positions (metres).
    max_range_m : float
        Maximum communications range.
    path_loss_exp : float
        Path-loss exponent (2 = free-space).

    Returns
    -------
    quality : float
        0.0 = unreachable, 1.0 = co-located / perfect link.
    """
    dist = float(np.linalg.norm(pos_a - pos_b))
    if dist <= 0:
        return 1.0
    if dist >= max_range_m:
        return 0.0
    # Normalised inverse-power model
    quality = 1.0 - (dist / max_range_m) ** path_loss_exp
    return float(np.clip(quality, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SDN Controller
# ---------------------------------------------------------------------------

class SDNController:
    """
    Centralised Software-Defined Networking controller for the mesh.

    Responsibilities:
    - Maintain a global topology view (adjacency with link quality)
    - Compute shortest paths using Dijkstra (weight = 1/quality)
    - Install / expire flow rules
    - Detect node failures and trigger reroute

    Parameters
    ----------
    max_range_m : float
        Maximum radio range for link feasibility.
    min_link_quality : float
        Minimum quality threshold for a link to be considered usable.
    """

    def __init__(
        self,
        max_range_m: float = 1000.0,
        min_link_quality: float = 0.1,
    ) -> None:
        self._max_range = max_range_m
        self._min_quality = min_link_quality

        # Node registry: node_id -> MeshNode
        self._nodes: Dict[str, MeshNode] = {}
        # Adjacency: node_id -> {neighbour_id: quality}
        self._adjacency: Dict[str, Dict[str, float]] = {}
        # Flow rules: (source, destination) -> list of rules (priority-sorted)
        self._flow_rules: Dict[Tuple[str, str], List[SDNFlowRule]] = {}

        # Failure tracking
        self._failed_nodes: Set[str] = set()

    # -- topology management -------------------------------------------------

    def register_node(self, node: MeshNode) -> None:
        """Add or update a node in the topology."""
        self._nodes[node.node_id] = node
        if node.active:
            self._failed_nodes.discard(node.node_id)
        else:
            self._failed_nodes.add(node.node_id)
        self._rebuild_adjacency()

    def remove_node(self, node_id: str) -> Optional[MeshNode]:
        """
        Remove a node entirely from the topology.

        Invalidates all flow rules involving this node.
        """
        node = self._nodes.pop(node_id, None)
        self._adjacency.pop(node_id, None)
        self._failed_nodes.discard(node_id)
        # Remove from neighbour lists
        for adj in self._adjacency.values():
            adj.pop(node_id, None)
        # Invalidate flow rules
        self._invalidate_rules_for_node(node_id)
        return node

    def mark_node_failed(self, node_id: str) -> None:
        """Mark a node as failed and trigger reroute for affected flows."""
        if node_id in self._nodes:
            self._nodes[node_id].active = False
        self._failed_nodes.add(node_id)
        self._rebuild_adjacency()
        self._invalidate_rules_for_node(node_id)

    def mark_node_recovered(self, node_id: str) -> None:
        """Re-enable a previously failed node."""
        if node_id in self._nodes:
            self._nodes[node_id].active = True
        self._failed_nodes.discard(node_id)
        self._rebuild_adjacency()

    def detect_failures(self, active_node_ids: Set[str]) -> List[str]:
        """
        Compare reported active nodes against the registry.

        Nodes present in the registry but absent from *active_node_ids*
        are marked as failed.

        Returns list of newly failed node IDs.
        """
        newly_failed: List[str] = []
        for nid, node in self._nodes.items():
            if nid not in active_node_ids and node.active:
                self.mark_node_failed(nid)
                newly_failed.append(nid)
        # Nodes that reappear
        for nid in active_node_ids:
            if nid in self._failed_nodes:
                self.mark_node_recovered(nid)
        return newly_failed

    # -- adjacency / graph ---------------------------------------------------

    def _rebuild_adjacency(self) -> None:
        """Recompute the adjacency matrix from current node positions."""
        self._adjacency.clear()
        active_nodes = {
            nid: n for nid, n in self._nodes.items()
            if n.active and nid not in self._failed_nodes
        }
        node_ids = list(active_nodes.keys())

        for nid in node_ids:
            self._adjacency[nid] = {}

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                nid_a, nid_b = node_ids[i], node_ids[j]
                q = _link_quality(
                    active_nodes[nid_a].position,
                    active_nodes[nid_b].position,
                    max_range_m=self._max_range,
                )
                if q >= self._min_quality:
                    self._adjacency[nid_a][nid_b] = q
                    self._adjacency[nid_b][nid_a] = q

    def get_adjacency(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of the current adjacency."""
        return {k: dict(v) for k, v in self._adjacency.items()}

    def get_link_quality(self, node_a: str, node_b: str) -> float:
        """Return link quality between two nodes, or 0.0 if no link."""
        return self._adjacency.get(node_a, {}).get(node_b, 0.0)

    # -- Dijkstra routing ----------------------------------------------------

    def compute_shortest_path(
        self, source: str, destination: str
    ) -> Tuple[List[str], float]:
        """
        Compute the shortest path using Dijkstra with quality-based weights.

        Weight of each edge = 1 / quality  (lower quality => higher cost).

        Returns
        -------
        path : list of str
            Ordered node IDs from source to destination (inclusive).
            Empty list if no path exists.
        cost : float
            Total path cost (sum of weights).  ``float('inf')`` if unreachable.
        """
        if source not in self._adjacency or destination not in self._adjacency:
            return [], float("inf")

        # Standard Dijkstra
        dist: Dict[str, float] = {nid: float("inf") for nid in self._adjacency}
        prev: Dict[str, Optional[str]] = {nid: None for nid in self._adjacency}
        dist[source] = 0.0
        # Priority queue: (cost, node_id)
        heap: List[Tuple[float, str]] = [(0.0, source)]
        visited: Set[str] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == destination:
                break
            for v, quality in self._adjacency.get(u, {}).items():
                if v in visited:
                    continue
                weight = 1.0 / quality if quality > 0 else float("inf")
                alt = d + weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

        # Reconstruct path
        if dist[destination] == float("inf"):
            return [], float("inf")

        path: List[str] = []
        node: Optional[str] = destination
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        return path, dist[destination]

    def compute_all_routes(self) -> Dict[Tuple[str, str], List[str]]:
        """Compute shortest paths between all active node pairs."""
        routes: Dict[Tuple[str, str], List[str]] = {}
        node_ids = list(self._adjacency.keys())
        for i, src in enumerate(node_ids):
            for dst in node_ids[i + 1:]:
                path, _ = self.compute_shortest_path(src, dst)
                if path:
                    routes[(src, dst)] = path
                    routes[(dst, src)] = list(reversed(path))
        return routes

    # -- flow rule management ------------------------------------------------

    def install_flow_rule(self, rule: SDNFlowRule) -> None:
        """Install a flow rule, sorted by priority (highest first)."""
        key = (rule.source, rule.destination)
        if key not in self._flow_rules:
            self._flow_rules[key] = []
        self._flow_rules[key].append(rule)
        self._flow_rules[key].sort(key=lambda r: r.priority, reverse=True)

    def get_flow_rule(
        self, source: str, destination: str, now: Optional[float] = None
    ) -> Optional[SDNFlowRule]:
        """Retrieve the highest-priority non-expired flow rule."""
        now = now if now is not None else time.monotonic()
        key = (source, destination)
        rules = self._flow_rules.get(key, [])
        for rule in rules:
            if not rule.is_expired(now):
                return rule
        return None

    def expire_flow_rules(self, now: Optional[float] = None) -> int:
        """Remove all expired flow rules. Return count removed."""
        now = now if now is not None else time.monotonic()
        removed = 0
        for key in list(self._flow_rules.keys()):
            remaining = [r for r in self._flow_rules[key] if not r.is_expired(now)]
            removed += len(self._flow_rules[key]) - len(remaining)
            if remaining:
                self._flow_rules[key] = remaining
            else:
                del self._flow_rules[key]
        return removed

    def _invalidate_rules_for_node(self, node_id: str) -> None:
        """Remove all flow rules that involve *node_id* as a hop."""
        to_delete: List[Tuple[str, str]] = []
        for key, rules in self._flow_rules.items():
            src, dst = key
            if src == node_id or dst == node_id:
                to_delete.append(key)
                continue
            # Also remove rules whose next_hop is the failed node
            remaining = [r for r in rules if r.next_hop != node_id]
            if remaining:
                self._flow_rules[key] = remaining
            else:
                to_delete.append(key)
        for key in to_delete:
            self._flow_rules.pop(key, None)

    def install_routes_as_flow_rules(
        self, priority: int = 10, ttl: float = 60.0
    ) -> int:
        """
        Compute all shortest paths and install as flow rules.

        Returns the number of rules installed.
        """
        routes = self.compute_all_routes()
        now = time.monotonic()
        expiry = now + ttl if ttl > 0 else 0.0
        count = 0

        for (src, dst), path in routes.items():
            if len(path) < 2:
                continue
            # Install a rule at every hop
            for i in range(len(path) - 1):
                rule = SDNFlowRule(
                    source=path[i],
                    destination=dst,
                    next_hop=path[i + 1],
                    priority=priority,
                    expiry=expiry,
                )
                self.install_flow_rule(rule)
                count += 1

        return count

    @property
    def nodes(self) -> Dict[str, MeshNode]:
        return dict(self._nodes)

    @property
    def failed_nodes(self) -> Set[str]:
        return set(self._failed_nodes)


# ---------------------------------------------------------------------------
# Mesh Network
# ---------------------------------------------------------------------------

class MeshNetwork:
    """
    Multi-node self-healing mesh network.

    Wraps an :class:`SDNController` with higher-level operations: packet
    forwarding, automatic route computation, and self-healing on node failure.

    Parameters
    ----------
    max_range_m : float
        Maximum radio range for link feasibility (metres).
    min_link_quality : float
        Minimum link quality for usability.
    auto_install_routes : bool
        If True, routes are recomputed and flow rules installed after every
        topology change.
    """

    def __init__(
        self,
        max_range_m: float = 1000.0,
        min_link_quality: float = 0.1,
        auto_install_routes: bool = True,
    ) -> None:
        self._controller = SDNController(
            max_range_m=max_range_m,
            min_link_quality=min_link_quality,
        )
        self._auto_routes = auto_install_routes
        self._event_log: List[Dict] = []
        self._packet_counter: int = 0

    # -- properties ----------------------------------------------------------

    @property
    def controller(self) -> SDNController:
        """Direct access to the SDN controller for advanced operations."""
        return self._controller

    @property
    def event_log(self) -> List[Dict]:
        return list(self._event_log)

    @property
    def nodes(self) -> Dict[str, MeshNode]:
        return self._controller.nodes

    @property
    def active_nodes(self) -> Dict[str, MeshNode]:
        return {
            nid: n for nid, n in self._controller.nodes.items()
            if n.active and nid not in self._controller.failed_nodes
        }

    # -- logging -------------------------------------------------------------

    def _log(self, event: str, detail: str = "") -> None:
        self._event_log.append({
            "timestamp": time.monotonic(),
            "event": event,
            "detail": detail,
        })

    # -- topology management -------------------------------------------------

    def add_node(self, node: MeshNode) -> None:
        """
        Add a node to the mesh.

        Triggers adjacency rebuild and (optionally) route recomputation.
        """
        self._controller.register_node(node)
        self._log("node_added", f"id={node.node_id} pos={node.position.tolist()}")
        if self._auto_routes:
            self._controller.install_routes_as_flow_rules()

    def remove_node(self, node_id: str) -> Optional[MeshNode]:
        """
        Remove a node and self-heal affected routes.
        """
        node = self._controller.remove_node(node_id)
        if node is not None:
            self._log("node_removed", f"id={node_id}")
            if self._auto_routes:
                self._controller.install_routes_as_flow_rules()
        return node

    def fail_node(self, node_id: str) -> None:
        """Simulate a node failure and trigger self-healing reroute."""
        self._controller.mark_node_failed(node_id)
        self._log("node_failed", f"id={node_id}")
        if self._auto_routes:
            self._controller.install_routes_as_flow_rules()

    def recover_node(self, node_id: str) -> None:
        """Restore a failed node and recompute routes."""
        self._controller.mark_node_recovered(node_id)
        self._log("node_recovered", f"id={node_id}")
        if self._auto_routes:
            self._controller.install_routes_as_flow_rules()

    # -- routing -------------------------------------------------------------

    def get_route(self, source: str, destination: str) -> List[str]:
        """
        Get the current route from *source* to *destination*.

        Uses the SDN flow rules.  Falls back to a live Dijkstra computation
        if no rule is installed.

        Returns
        -------
        route : list of str
            Ordered list of node IDs.  Empty if no route exists.
        """
        # Try flow rule chain first
        route = self._trace_flow_rules(source, destination)
        if route:
            return route
        # Fallback to Dijkstra
        path, cost = self._controller.compute_shortest_path(source, destination)
        return path

    def _trace_flow_rules(self, source: str, destination: str) -> List[str]:
        """Trace the forwarding path via installed flow rules."""
        path = [source]
        current = source
        visited: Set[str] = {source}
        max_hops = len(self._controller.nodes) + 1

        for _ in range(max_hops):
            if current == destination:
                return path
            rule = self._controller.get_flow_rule(current, destination)
            if rule is None:
                return []  # no rule for this hop
            next_hop = rule.next_hop
            if next_hop in visited:
                return []  # loop detected
            visited.add(next_hop)
            path.append(next_hop)
            current = next_hop

        return []  # exceeded max hops

    # -- packet forwarding ---------------------------------------------------

    def send_packet(
        self,
        source: str,
        destination: str,
        payload: bytes = b"",
    ) -> Dict[str, object]:
        """
        Send a packet through the mesh from *source* to *destination*.

        Returns
        -------
        result : dict
            ``packet_id``  : str
            ``delivered``  : bool
            ``route``      : list of node IDs traversed
            ``hops``       : int
            ``detail``     : str
        """
        self._packet_counter += 1
        pkt = Packet(
            packet_id=f"pkt-{self._packet_counter}",
            source=source,
            destination=destination,
            payload=payload,
        )

        route = self.get_route(source, destination)
        if not route or route[-1] != destination:
            self._log("packet_dropped", f"{pkt.packet_id}: no route "
                       f"{source} -> {destination}")
            return {
                "packet_id": pkt.packet_id,
                "delivered": False,
                "route": [],
                "hops": 0,
                "detail": f"No route from {source} to {destination}",
            }

        pkt.route = route
        hops = len(route) - 1

        # Check that every node along the route is active
        for nid in route:
            node = self._controller.nodes.get(nid)
            if node is None or not node.active or nid in self._controller.failed_nodes:
                self._log("packet_dropped", f"{pkt.packet_id}: node {nid} "
                           "down along route")
                return {
                    "packet_id": pkt.packet_id,
                    "delivered": False,
                    "route": route,
                    "hops": hops,
                    "detail": f"Node {nid} is down along the route",
                }

        self._log("packet_delivered", f"{pkt.packet_id}: "
                   f"{' -> '.join(route)} ({hops} hops)")
        return {
            "packet_id": pkt.packet_id,
            "delivered": True,
            "route": route,
            "hops": hops,
            "detail": f"Delivered via {hops} hop(s)",
        }

    # -- utility -------------------------------------------------------------

    def get_topology_summary(self) -> Dict[str, object]:
        """Return a human-readable summary of the mesh topology."""
        adj = self._controller.get_adjacency()
        active = self.active_nodes
        total = self._controller.nodes

        link_count = sum(len(neighbours) for neighbours in adj.values()) // 2

        return {
            "total_nodes": len(total),
            "active_nodes": len(active),
            "failed_nodes": sorted(self._controller.failed_nodes),
            "links": link_count,
            "adjacency": adj,
        }
