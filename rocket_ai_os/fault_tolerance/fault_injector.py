"""Fault Injection Infrastructure

Provides deterministic, schedule-based fault injection for evaluating FDIR and Simplex systems.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from rocket_ai_os.gnc.navigation import NavigationState

class FaultInjector:
    """Schedule-based fault injector for system evaluation.
    
    Can inject faults into sensor readings (NavigationState), actuator commands
    (torque/throttle), or internal states, based on simulation time.
    """
    
    def __init__(self, schedule: list[dict[str, Any]]):
        """Initialise with a timeline of faults.
        
        Args:
            schedule: List of fault definitions. Expected keys:
                - 't': Time to trigger the fault [s].
                - 'type': Fault type (e.g. 'rl_spike', 'sensor_corrupt', 'gyro_drift', 'adversarial', 'stuck', 'noise').
                - 'magnitude': Severity or scaling factor of the fault.
        """
        self.schedule = sorted(schedule, key=lambda x: x["t"])
        self.active_faults: list[dict[str, Any]] = []
        
    def step_faults(self, t: float, dt: float):
        """Update the list of active faults based on current time."""
        for fault in self.schedule:
            # Check if within activation window (e.g., fault lasts indefinitely unless specified)
            # For simplicity, if timestamp is reached, it becomes active.
            if t >= fault["t"] and fault not in self.active_faults:
                self.active_faults.append(fault)

    def apply(self, t: float, state: NavigationState, control_torque: np.ndarray, control_throttle: float) -> tuple[np.ndarray, float]:
        """Apply active faults to the navigation state and control outputs.
        
        Modifies `state` in place for sensor faults.
        Returns corrupted `(control_torque, control_throttle)` for actuator faults.
        """
        corrupted_torque = control_torque.copy()
        corrupted_throttle = control_throttle
        
        for fault in self.schedule:
            # For momentary or persistent faults, trigger condition:
            if abs(t - fault["t"]) < 0.01 or (t >= fault["t"]):
                ftype = fault["type"]
                mag = fault["magnitude"]
                
                # Actuator / Controller Faults
                if ftype == "rl_spike":
                    corrupted_torque *= mag
                    corrupted_throttle *= mag
                elif ftype == "adversarial":
                    corrupted_torque = -control_torque * mag
                    corrupted_throttle = -control_throttle * mag
                elif ftype == "stuck":
                    corrupted_torque = np.ones_like(control_torque) * mag * 50.0
                    corrupted_throttle = max(min(mag, 1.0), 0.0)
                elif ftype == "noise":
                    # Noise grows with magnitude
                    corrupted_torque += np.random.normal(0, mag * 30.0, control_torque.shape)
                    corrupted_throttle += np.random.normal(0, mag * 0.3)
                    
                # Sensor Faults (persist once triggered if t >= fault["t"])
                elif ftype == "sensor_corrupt" and abs(t - fault["t"]) < 0.01:
                    # One-off spike
                    state.attitude += np.random.randn(4) * mag
                    state.attitude /= np.linalg.norm(state.attitude)
                elif ftype == "gyro_drift":
                    # Ongoing drift added to angular rates
                    state.angular_rates += mag * (t - fault["t"])

        return corrupted_torque, corrupted_throttle
