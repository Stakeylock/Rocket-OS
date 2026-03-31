"""
Gym-compatible environment wrapper for Autonomous Rocket AI OS.

This wrapper allows the rocket AI OS to be used as a reinforcement learning
environment compatible with OpenAI Gym/gymnasium standards.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces

from rocket_ai_os.main import main
from rocket_ai_os.config import SystemConfig, MissionPhase
from rocket_ai_os.sim.vehicle import Vehicle, VehicleState
from rocket_ai_os.gnc.navigation import NavigationSystem
from rocket_ai_os.gnc.guidance import GuidanceSystem
from rocket_ai_os.gnc.control import FlightController
from rocket_ai_os.propulsion.engine import EngineCluster
from rocket_ai_os.propulsion.ftca import FTCAAllocator
from rocket_ai_os.fault_tolerance.fdir import FDIRSystem


class RocketAviaryEnv(gym.Env):
    """
    Gym-compatible environment for autonomous rocket landing.

    State Space:
        - Position (3): [x, y, z] in meters
        - Velocity (3): [vx, vy, vz] in m/s
        - Attitude (4): [qw, qx, qy, qz] quaternion
        - Angular rates (3): [wx, wy, wz] in rad/s
        - Mass (1): current vehicle mass in kg
        - Fuel (1): remaining fuel mass in kg

    Action Space:
        - Throttle (1): [0, 1] normalized throttle command
        - Gimbal pitch (1): [-max_angle, max_angle] in radians
        - Gimbal yaw (1): [-max_angle, max_angle] in radians

    Reward Function:
        - Negative distance to target landing zone
        - Negative velocity magnitude (encourage soft landing)
        - Negative fuel consumption (encourage efficiency)
        - Large negative reward for crashing or going out of bounds
        - Large positive reward for successful landing
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__()

        self.config = config or SystemConfig()
        self.vehicle_config = self.config.vehicle

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.vehicle_config.max_gimbal_angle, -self.vehicle_config.max_gimbal_angle]),
            high=np.array([1.0, self.vehicle_config.max_gimbal_angle, self.vehicle_config.max_gimbal_angle]),
            dtype=np.float32
        )

        # Observation: [pos(3), vel(3), quat(4), ang_vel(3), mass(1), fuel(1)]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*15),
            high=np.array([np.inf]*15),
            dtype=np.float32
        )

        # Initialize subsystem components
        self._reset_subsystems()

        # Episode tracking
        self.current_step = 0
        self.max_steps = int(self.config.sim.max_time / self.config.sim.dt)
        self.landed = False
        self.crashed = False
        self.out_of_bounds = False

    def _reset_subsystems(self):
        """Reset all subsystem components to initial state."""
        # Vehicle dynamics
        self.vehicle = Vehicle(config=self.vehicle_config, initial_phase=MissionPhase.LANDING_BURN)
        self.vehicle.set_state(
            position=np.array([200.0, 50.0, 1500.0]),
            velocity=np.array([-50.0, 0.0, -80.0]),
            mass=self.vehicle_config.total_mass,
            fuel_mass=self.vehicle_config.fuel_mass,
        )

        # Guidance, navigation, control
        self.navigation = NavigationSystem(
            vehicle_config=self.vehicle_config,
            sim_config=self.config.sim,
            seed=42
        )
        self.guidance = GuidanceSystem(
            vehicle_config=self.vehicle_config,
            guidance_config=self.config.guidance
        )
        self.controller = FlightController(
            vehicle_config=self.vehicle_config,
            sim_config=self.config.sim,
            rl_seed=42
        )

        # Propulsion
        self.engine_cluster = EngineCluster(config=self.vehicle_config)
        self.ftca = FTCAAllocator(self.engine_cluster)

        # Fault tolerance
        self.fdir = FDIRSystem()

        # Set initial conditions
        self.navigation.initialise(
            position=np.array([200.0, 50.0, 1500.0]),
            velocity=np.array([-50.0, 0.0, -80.0]),
            mass=self.vehicle_config.total_mass
        )

        # Set guidance target (landing pad at origin)
        self.guidance.set_phase(MissionPhase.LANDING_BURN)

        # Set initial desired state for controller
        self.controller.set_desired_state(
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),  # level attitude
            position=np.array([0.0, 0.0, 0.0]),       # target at origin
            velocity=np.array([0.0, 0.0, 0.0]),       # zero velocity
            throttle=0.0
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self._reset_subsystems()
        self.current_step = 0
        self.landed = False
        self.crashed = False
        self.out_of_bounds = False

        observation = self._get_observation()
        info = self._get_info()

        return observation.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep of the environment dynamics."""
        if self.landed or self.crashed or self.out_of_bounds:
            # Episode already terminated
            observation = self._get_observation()
            reward = self._compute_reward()
            info = self._get_info()
            return observation.astype(np.float32), reward, True, False, info

        # Parse action: [throttle, gimbal_pitch, gimbal_yaw]
        throttle_cmd = np.clip(action[0], 0.0, 1.0)
        gimbal_pitch = np.clip(action[1], -self.vehicle_config.max_gimbal_angle, self.vehicle_config.max_gimbal_angle)
        gimbal_yaw = np.clip(action[2], -self.vehicle_config.max_gimbal_angle, self.vehicle_config.max_gimbal_angle)

        # Apply action to engine cluster
        gimbal_commands = np.tile([gimbal_pitch, gimbal_yaw], (self.engine_cluster.num_engines, 1))
        throttle_commands = np.full(self.engine_cluster.num_engines, throttle_cmd)
        self.engine_cluster.command_all(throttle_commands, gimbal_commands)

        # Physics integration step
        dt = self.config.sim.dt

        # Get current state
        vehicle_state = self.vehicle.state
        nav_state = self.navigation.get_latest_state()

        # Update navigation with IMU data (simplified)
        accel_meas = np.array([0.0, 0.0, 9.81])  # Simplified IMU reading
        gyro_meas = vehicle_state.angular_velocity

        updated_nav_state = self.navigation.step(
            true_accel_body=accel_meas,
            true_omega_body=gyro_meas,
            true_position=vehicle_state.position,
            true_velocity=vehicle_state.velocity,
            mass=vehicle_state.mass,
            time=self.current_step * dt
        )

        # Debug: Check the attitude quaternion
        # print(f"Navigation attitude: {updated_nav_state.attitude}")  # Debug line

        # Update guidance
        self.guidance.update(updated_nav_state, time=self.current_step * dt)

        # Get guidance command
        traj_point = self.guidance.update(updated_nav_state, time=self.current_step * dt)
        if traj_point is not None:
            # Debug: Check the thrust direction
            # print(f"Guidance thrust direction: {traj_point.thrust_direction}")
            # print(f"Guidance thrust direction shape: {traj_point.thrust_direction.shape}")
            pass  # Continue to controller update

        # Update controller
        control_cmd = self.controller.step(updated_nav_state)

        # Override with our RL action for the engine commands
        # (In a more sophisticated version, we'd blend RL and guidance)
        final_throttle = throttle_cmd
        final_gimbal = np.array([gimbal_pitch, gimbal_yaw])

        # Apply final commands to engines
        final_gimbal_commands = np.tile(final_gimbal, (self.engine_cluster.num_engines, 1))
        final_throttle_commands = np.full(self.engine_cluster.num_engines, final_throttle)
        self.engine_cluster.command_all(final_throttle_commands, final_gimbal_commands)

        # Step physics
        engine_states = self.engine_cluster.step(dt, altitude=vehicle_state.position[2])
        total_force, total_torque = self.engine_cluster.get_total_force_and_torque()

        # Simple physics integration (would be more sophisticated in practice)
        # For now, we'll use the vehicle's built-in dynamics
        self.vehicle.apply_forces(total_force, total_torque, dt)

        # Consume fuel based on throttle
        total_thrust = np.sum([engine.thrust_actual for engine in engine_states])
        if total_thrust > 0:
            isp = 282.0  # sea-level specific impulse (simplified)
            mass_flow = total_thrust / (isp * 9.81)
            self.vehicle.consume_fuel(mass_flow, dt)

        # Update FDIR with engine telemetry
        engine_telemetry = {}
        for engine_state in engine_states:
            engine_telemetry[f"engine_{engine_state.engine_id}_chamber_pressure"] = engine_state.chamber_pressure
            engine_telemetry[f"engine_{engine_state.engine_id}_turbopump_rpm"] = engine_state.turbopump_rpm
        self.fdir.detect(engine_telemetry, timestamp=self.current_step * dt)

        # Increment step counter
        self.current_step += 1

        # Check termination conditions
        observation = self._get_observation()
        reward = self._compute_reward()
        info = self._get_info()

        # Check if episode is done
        done = self.landed or self.crashed or self.out_of_bounds or (self.current_step >= self.max_steps)

        return observation.astype(np.float32), reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        vehicle_state = self.vehicle.state

        # [pos(3), vel(3), quat(4), ang_vel(3), mass(1), fuel(1)]
        obs = np.concatenate([
            vehicle_state.position,
            vehicle_state.velocity,
            vehicle_state.attitude,
            vehicle_state.angular_velocity,
            [vehicle_state.mass],
            [vehicle_state.fuel_mass]
        ])

        return obs

    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        vehicle_state = self.vehicle.state

        # Distance to target (origin)
        distance_to_target = np.linalg.norm(vehicle_state.position[:2])  # xy distance only
        altitude = vehicle_state.position[2]

        # Velocity magnitude
        velocity_magnitude = np.linalg.norm(vehicle_state.velocity)
        vertical_velocity = abs(vehicle_state.velocity[2])  # z-component

        # Fuel consumption (negative reward for using fuel)
        fuel_used = self.vehicle_config.fuel_mass - vehicle_state.fuel_mass

        # Base reward components
        reward = 0.0

        # Penalty for distance from target
        reward -= 0.1 * distance_to_target

        # Penalty for high velocity (especially vertical)
        reward -= 0.05 * velocity_magnitude
        reward -= 0.1 * vertical_velocity

        # Small penalty for fuel consumption (encourage efficiency)
        reward -= 0.001 * fuel_used

        # Check for landing success
        if altitude < 1.0 and vertical_velocity < 2.0 and distance_to_target < 5.0:
            if not self.landed:
                self.landed = True
                # Large reward for successful landing
                reward += 100.0

        # Check for crash
        if altitude < 0.0:  # Hit ground
            if not self.crashed:
                self.crashed = True
                # Large penalty for crashing
                reward -= 100.0

        # Check for out of bounds
        bounds_limit = 500.0  # meters
        if (abs(vehicle_state.position[0]) > bounds_limit or
            abs(vehicle_state.position[1]) > bounds_limit or
            vehicle_state.position[2] < -10.0):  # Below ground with margin
            if not self.out_of_bounds:
                self.out_of_bounds = True
                # Penalty for going out of bounds
                reward -= 50.0

        # Small survival bonus
        if not (self.landed or self.crashed or self.out_of_bounds):
            reward += 0.01

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        vehicle_state = self.vehicle.state

        return {
            "position": vehicle_state.position.copy(),
            "velocity": vehicle_state.velocity.copy(),
            "altitude": vehicle_state.position[2],
            "fuel_remaining": vehicle_state.fuel_mass,
            "mass": vehicle_state.mass,
            "landed": self.landed,
            "crashed": self.crashed,
            "out_of_bounds": self.out_of_bounds,
            "step": self.current_step,
            "fdir_faults": self.fdir.stats()["total_faults_detected"],
        }


# Register the environment with gymnasium
def register_rocket_env():
    """Register the RocketAviary environment with gymnasium."""
    try:
        import gymnasium as gym
        gym.register(
            id='RocketAviary-v0',
            entry_point='rocket_ai_os.gym_wrapper:RocketAviaryEnv',
            max_episode_steps=1000,
        )
    except Exception:
        pass  # Gymnasium might not be available


if __name__ == "__main__":
    # Simple test run
    env = RocketAviaryEnv()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        if done:
            break

    env.close()