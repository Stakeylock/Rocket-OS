"""
Dynamic fuel management with multi-tank balancing and CoM tracking.

Tracks propellant mass in multiple tanks (e.g., LOX and RP-1), computes
the centre-of-mass shift as fuel depletes, detects asymmetric drain, and
manages cross-feed valves to maintain CoM within acceptable limits.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TankState:
    """Observable state of a single propellant tank."""
    tank_id: str
    fuel_mass: float            # kg -- current propellant remaining
    capacity: float             # kg -- maximum propellant the tank can hold
    flow_rate: float = 0.0      # kg/s -- current outflow (positive = draining)
    temperature: float = 90.0   # K   -- cryogenic propellant temperature
    pressure: float = 3.0e5     # Pa  -- ullage / head pressure
    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )                           # (3,) body-frame position of tank centroid

    @property
    def fill_fraction(self) -> float:
        """Fraction of tank that still contains propellant."""
        if self.capacity <= 0:
            return 0.0
        return self.fuel_mass / self.capacity

    @property
    def is_empty(self) -> bool:
        return self.fuel_mass <= 0.0

    def copy(self) -> "TankState":
        return TankState(
            tank_id=self.tank_id,
            fuel_mass=self.fuel_mass,
            capacity=self.capacity,
            flow_rate=self.flow_rate,
            temperature=self.temperature,
            pressure=self.pressure,
            position=self.position.copy(),
        )


@dataclass
class CrossFeedValve:
    """A valve connecting two tanks for propellant cross-feed."""
    valve_id: str
    tank_a_id: str
    tank_b_id: str
    max_flow_rate: float = 50.0     # kg/s
    is_open: bool = False
    position: float = 0.0           # 0=closed, 1=fully open


# ---------------------------------------------------------------------------
# Fuel Manager
# ---------------------------------------------------------------------------

class FuelManager:
    """Manages propellant state across all tanks.

    Capabilities
    -------------
    * Multi-tank management (LOX + RP-1 or arbitrary propellant pairs).
    * Centre-of-Mass (CoM) tracking as propellant depletes.
    * CoM balancing via cross-feed valve actuation.
    * Asymmetric drain detection.
    * Fuel budget / remaining burn-time prediction.
    * Provides updated mass properties to the GNC plant model.
    """

    # Oxidizer-to-fuel mass ratio (LOX/RP-1 ~ 2.36:1 for Merlin-class)
    DEFAULT_OF_RATIO: float = 2.36

    def __init__(
        self,
        tanks: List[TankState],
        dry_mass: float = 22_200.0,
        dry_com: np.ndarray = None,
        of_ratio: float = 2.36,
        cross_feed_valves: Optional[List[CrossFeedValve]] = None,
    ):
        """
        Parameters
        ----------
        tanks : list of TankState
            Initial tank configuration.
        dry_mass : float
            Vehicle dry mass in kg.
        dry_com : (3,) ndarray
            CoM of the dry vehicle structure in the body frame.
        of_ratio : float
            Oxidizer-to-fuel mass ratio.
        cross_feed_valves : list of CrossFeedValve, optional
            Valves for propellant cross-feed between tanks.
        """
        self.tanks: Dict[str, TankState] = {t.tank_id: t.copy() for t in tanks}
        self.dry_mass = dry_mass
        self.dry_com = dry_com if dry_com is not None else np.array([0.0, 0.0, 20.0])
        self.of_ratio = of_ratio
        self.cross_feed_valves: Dict[str, CrossFeedValve] = {}
        if cross_feed_valves:
            for v in cross_feed_valves:
                self.cross_feed_valves[v.valve_id] = v

        # CoM balancing controller state
        self._com_target: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._com_deadband: float = 0.05  # m -- don't actuate within this band

        # Asymmetry detection
        self._asymmetry_threshold: float = 0.05  # 5 % fill-fraction difference

        # History (for budget prediction)
        self._total_consumed: float = 0.0
        self._consumption_history: List[Tuple[float, float]] = []  # (time, mass)
        self._elapsed_time: float = 0.0

    # ------------------------------------------------------------------
    # Tank queries
    # ------------------------------------------------------------------

    @property
    def total_fuel_mass(self) -> float:
        return sum(t.fuel_mass for t in self.tanks.values())

    @property
    def total_vehicle_mass(self) -> float:
        return self.dry_mass + self.total_fuel_mass

    def get_tank(self, tank_id: str) -> TankState:
        return self.tanks[tank_id]

    def get_all_tank_states(self) -> List[TankState]:
        return [t.copy() for t in self.tanks.values()]

    # ------------------------------------------------------------------
    # Centre of Mass
    # ------------------------------------------------------------------

    def compute_com(self) -> np.ndarray:
        """Compute the current vehicle centre-of-mass in the body frame.

        Uses mass-weighted average of dry structure CoM and each tank's
        centroid (approximate -- assumes propellant CoM at tank centroid
        scaled by fill level along body Z).
        """
        total_mass = self.total_vehicle_mass
        if total_mass < 1e-6:
            return self.dry_com.copy()

        weighted = self.dry_mass * self.dry_com.copy()
        for t in self.tanks.values():
            # As propellant drains the liquid level drops; approximate
            # the propellant CoM as shifting downward proportional to fill.
            prop_com = t.position.copy()
            # Shift Z component: full tank -> centroid; empty -> bottom
            prop_com[2] -= (1.0 - t.fill_fraction) * 2.0  # ~2 m half-height
            weighted += t.fuel_mass * prop_com

        return weighted / total_mass

    def compute_inertia_adjustment(self) -> np.ndarray:
        """Return a diagonal inertia adjustment tensor (3,) to add to the
        base moment-of-inertia, accounting for propellant distribution.

        This is a simplified parallel-axis contribution from each tank.
        """
        com = self.compute_com()
        I_adj = np.zeros(3)
        for t in self.tanks.values():
            r = t.position - com
            # Parallel-axis theorem (diagonal approximation)
            r_sq = np.dot(r, r)
            I_adj[0] += t.fuel_mass * (r_sq - r[0] ** 2)
            I_adj[1] += t.fuel_mass * (r_sq - r[1] ** 2)
            I_adj[2] += t.fuel_mass * (r_sq - r[2] ** 2)
        return I_adj

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def consume_fuel(
        self,
        flow_rates: Dict[str, float],
        dt: float,
    ) -> List[TankState]:
        """Drain propellant from tanks over a time-step.

        Parameters
        ----------
        flow_rates : dict  {tank_id: flow_rate_kg_per_s}
            Positive flow = propellant leaving the tank.
        dt : float
            Time-step (s).

        Returns
        -------
        list of updated TankState copies.
        """
        total_consumed = 0.0
        for tid, rate in flow_rates.items():
            if tid not in self.tanks:
                continue
            t = self.tanks[tid]
            consumed = min(rate * dt, t.fuel_mass)  # cannot go negative
            t.fuel_mass -= consumed
            t.flow_rate = rate
            total_consumed += consumed

            # Simple temperature model: slight warming as ullage volume grows
            if t.capacity > 0:
                t.temperature += 0.002 * (1.0 - t.fill_fraction) * dt

            # Pressure drop model (ideal gas expansion in ullage)
            if t.fill_fraction < 1.0:
                ullage_frac = max(1.0 - t.fill_fraction, 0.01)
                t.pressure = 3.0e5 / ullage_frac  # isothermal expansion approx

        self._total_consumed += total_consumed
        self._elapsed_time += dt
        self._consumption_history.append((self._elapsed_time, self.total_fuel_mass))

        # Cross-feed balancing (if valves are open)
        self._apply_cross_feed(dt)

        return self.get_all_tank_states()

    # ------------------------------------------------------------------
    # Cross-feed balancing
    # ------------------------------------------------------------------

    def _apply_cross_feed(self, dt: float) -> None:
        """Transfer propellant through open cross-feed valves."""
        for v in self.cross_feed_valves.values():
            if not v.is_open or v.position < 1e-3:
                continue
            ta = self.tanks.get(v.tank_a_id)
            tb = self.tanks.get(v.tank_b_id)
            if ta is None or tb is None:
                continue

            # Flow from higher-pressure to lower-pressure tank
            dp = ta.pressure - tb.pressure
            flow = v.position * v.max_flow_rate * np.sign(dp) * min(abs(dp) / 1e5, 1.0)
            transfer = flow * dt

            if transfer > 0:
                transfer = min(transfer, ta.fuel_mass)
                ta.fuel_mass -= transfer
                tb.fuel_mass += transfer
            else:
                transfer = min(-transfer, tb.fuel_mass)
                tb.fuel_mass -= transfer
                ta.fuel_mass += transfer

    def open_valve(self, valve_id: str, position: float = 1.0) -> None:
        """Open a cross-feed valve to a given position [0, 1]."""
        if valve_id in self.cross_feed_valves:
            v = self.cross_feed_valves[valve_id]
            v.is_open = position > 0.01
            v.position = float(np.clip(position, 0.0, 1.0))

    def close_valve(self, valve_id: str) -> None:
        """Close a cross-feed valve."""
        self.open_valve(valve_id, 0.0)

    def balance_com(self) -> Dict[str, float]:
        """Compute cross-feed valve commands to move CoM toward target.

        Returns
        -------
        dict  {valve_id: position}  recommended valve openings.
        """
        com = self.compute_com()
        error = com[:2] - self._com_target[:2]  # Only balance lateral (X, Y)
        error_mag = np.linalg.norm(error)

        commands: Dict[str, float] = {}
        if error_mag < self._com_deadband:
            # Within deadband, close all valves
            for vid in self.cross_feed_valves:
                commands[vid] = 0.0
            return commands

        # For each valve, determine if opening it reduces the error
        for vid, v in self.cross_feed_valves.items():
            ta = self.tanks.get(v.tank_a_id)
            tb = self.tanks.get(v.tank_b_id)
            if ta is None or tb is None:
                commands[vid] = 0.0
                continue

            # Direction of mass transfer: from A to B shifts CoM toward B
            direction = tb.position[:2] - ta.position[:2]
            if np.linalg.norm(direction) < 1e-6:
                commands[vid] = 0.0
                continue

            # Dot product: positive means transfer helps
            benefit = np.dot(direction / np.linalg.norm(direction),
                             -error / (error_mag + 1e-9))
            if benefit > 0.1:
                commands[vid] = min(benefit, 1.0)
            else:
                commands[vid] = 0.0

        return commands

    # ------------------------------------------------------------------
    # Asymmetric drain detection
    # ------------------------------------------------------------------

    def detect_asymmetric_drain(self) -> List[Tuple[str, str, float]]:
        """Detect pairs of tanks with significantly different fill levels.

        Returns list of (tank_a_id, tank_b_id, fill_difference) where
        the difference exceeds the threshold.
        """
        alerts: List[Tuple[str, str, float]] = []
        tank_list = list(self.tanks.values())
        for i in range(len(tank_list)):
            for j in range(i + 1, len(tank_list)):
                ta, tb = tank_list[i], tank_list[j]
                # Only compare tanks of similar capacity (same propellant type)
                if abs(ta.capacity - tb.capacity) > 0.1 * max(ta.capacity, tb.capacity):
                    continue
                diff = abs(ta.fill_fraction - tb.fill_fraction)
                if diff > self._asymmetry_threshold:
                    alerts.append((ta.tank_id, tb.tank_id, diff))
        return alerts

    # ------------------------------------------------------------------
    # Fuel budget prediction
    # ------------------------------------------------------------------

    def predict_remaining_burn_time(self) -> float:
        """Estimate remaining burn time based on recent consumption rate.

        Uses a sliding window over the last 5 seconds of history.

        Returns
        -------
        float  Estimated seconds of burn remaining (inf if no consumption).
        """
        if len(self._consumption_history) < 2:
            return float("inf")

        # Use last 5 s of data
        t_now = self._consumption_history[-1][0]
        window = 5.0
        recent = [
            (t, m) for t, m in self._consumption_history
            if t >= t_now - window
        ]
        if len(recent) < 2:
            return float("inf")

        dt = recent[-1][0] - recent[0][0]
        dm = recent[0][1] - recent[-1][1]  # mass decrease
        if dt < 1e-9 or dm < 1e-9:
            return float("inf")

        avg_rate = dm / dt  # kg/s
        return self.total_fuel_mass / avg_rate

    # ------------------------------------------------------------------
    # GNC interface
    # ------------------------------------------------------------------

    def get_mass_properties(self) -> Dict:
        """Return current mass properties for the GNC plant model.

        Returns dict with keys: total_mass, com, inertia_adjustment,
        fuel_remaining_fraction.
        """
        total = self.total_vehicle_mass
        com = self.compute_com()
        I_adj = self.compute_inertia_adjustment()
        initial_fuel = sum(t.capacity for t in self.tanks.values())
        frac = self.total_fuel_mass / initial_fuel if initial_fuel > 0 else 0.0

        return {
            "total_mass": total,
            "com": com,
            "inertia_adjustment": I_adj,
            "fuel_remaining_fraction": frac,
        }


# ---------------------------------------------------------------------------
# Factory for a typical two-propellant (LOX + RP-1) configuration
# ---------------------------------------------------------------------------

def create_default_tanks() -> List[TankState]:
    """Create a default tank layout modelled after a Falcon-9-class first stage.

    * 1 LOX tank (upper) -- ~287 t
    * 1 RP-1 tank (lower) -- ~109 t
    """
    lox = TankState(
        tank_id="LOX",
        fuel_mass=287_000.0,
        capacity=287_000.0,
        temperature=90.0,       # LOX boiling point ~90 K
        pressure=3.0e5,
        position=np.array([0.0, 0.0, 25.0]),  # Upper tank centroid
    )
    rp1 = TankState(
        tank_id="RP1",
        fuel_mass=108_700.0,
        capacity=108_700.0,
        temperature=293.0,      # Room temperature
        pressure=3.0e5,
        position=np.array([0.0, 0.0, 10.0]),  # Lower tank centroid
    )
    return [lox, rp1]
