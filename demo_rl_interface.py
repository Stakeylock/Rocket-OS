#!/usr/bin/env python3
"""
Demonstration of the Gymnasium RL Interface for Autonomous Rocket AI OS.

This script shows how to use the RocketAviaryEnv environment with
standard RL algorithms for policy learning.

Author: Jinitangsu Das (23011p0521@jntuhceh.ac.in) | GitHub: Stakeylock | Institution: JNTU Hyderabad
"""

import numpy as np
import gymnasium as gym
from rocket_ai_os.gym_wrapper import RocketAviaryEnv, register_rocket_env

def demo_helicopter_policy():
    """Demonstrate the environment with a simple helicopter-like policy."""
    print("=== Autonomous Rocket AI OS Gymnasium Interface Demo ===\n")

    # Register the environment (if not already registered)
    register_rocket_env()

    # Create the environment
    env = RocketAviaryEnv()

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print(f"Action Shape: {env.action_space.shape}\n")

    # Run multiple episodes with a simple stabilizing policy
    num_episodes = 3
    max_steps_per_episode = 500

    for episode in range(num_episodes):
        print(f"--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        obs, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0

        print(f"Initial altitude: {obs[2]:.1f}m")
        print(f"Initial position: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}]m")
        print(f"Initial velocity: [{obs[3]:.1f}, {obs[4]:.1f}, {obs[5]:.1f}]m/s")
        print(f"Initial fuel: {obs[13]:.1f}kg\n")

        # Run episode with a simple PD-like policy
        for step in range(max_steps_per_episode):
            # Extract relevant state variables
            position = obs[0:3]      # [x, y, z]
            velocity = obs[3:6]      # [vx, vy, vz]
            # attitude = obs[6:10]     # [qw, qx, qy, qz] - not used in simple policy
            # angular_velocity = obs[10:13]  # [wx, wy, wz] - not used in simple policy
            mass = obs[13]
            fuel = obs[14]

            # Simple PD controller for position and velocity control
            target_position = np.array([0.0, 0.0, 0.0])  # Target landing pad
            target_velocity = np.array([0.0, 0.0, 0.0])  # Target zero velocity

            # Position and velocity errors
            pos_error = target_position - position
            vel_error = target_velocity - velocity

            # Simple gains (these would be tuned for proper performance)
            kp_pos = 0.02
            kd_vel = 0.1
            kp_vel = 0.05

            # Compute desired acceleration
            desired_accel = kp_pos * pos_error + kp_vel * vel_error + kd_vel * (-velocity)
            desired_accel[2] += 9.81  # Add gravity compensation

            # Convert to thrust command (normalize to [0,1] range)
            max_accel = 20.0  # m/s^2, approximate max acceleration
            thrust_magnitude = np.clip(np.linalg.norm(desired_accel) / max_accel, 0.0, 1.0)

            # Compute gimbal angles to point thrust in desired direction
            if np.linalg.norm(desired_accel) > 0.1:
                thrust_direction = desired_accel / np.linalg.norm(desired_accel)
                # Simple conversion from thrust direction to gimbal angles
                # (this is a simplification - real implementation would be more complex)
                gimbal_pitch = np.arcsin(np.clip(thrust_direction[0], -0.5, 0.5))
                gimbal_yaw = np.arcsin(np.clip(thrust_direction[1], -0.5, 0.5))
            else:
                gimbal_pitch = 0.0
                gimbal_yaw = 0.0

            # Construct action: [throttle, gimbal_pitch, gimbal_yaw]
            action = np.array([thrust_magnitude, gimbal_pitch, gimbal_yaw])

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Print progress every 100 steps
            if step % 100 == 0 and step > 0:
                altitude = obs[2]
                speed = np.linalg.norm(obs[3:6])
                fuel = obs[14]
                print(f"  Step {step:3d}: Alt={altitude:6.1f}m, Speed={speed:6.1f}m/s, "
                      f"Fuel={fuel:6.1f}kg, Reward={reward:7.3f}")

            # Check if episode is done
            if terminated or truncated:
                print(f"  Episode terminated at step {step}")
                print(f"  Final altitude: {obs[2]:.1f}m")
                print(f"  Final position: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}]m")
                print(f"  Final speed: {np.linalg.norm(obs[3:6]):.1f}m/s")
                print(f"  Fuel remaining: {obs[14]:.1f}kg")
                print(f"  Total reward: {total_reward:.2f}")
                outcome = "Landed" if info['landed'] else "Crashed" if info['crashed'] else "Out of Bounds" if info['out_of_bounds'] else "Timeout"
                print(f"  Outcome: {outcome}")
                break

        if not (terminated or truncated):
            print(f"  Episode ended after {max_steps_per_episode} steps (timeout)")
            print(f"  Total reward: {total_reward:.2f}")

        print()

    env.close()
    print("Demo completed!")

def demo_observation_components():
    """Show what each component of the observation represents."""
    print("\n=== Observation Space Components ===")
    env = RocketAviaryEnv()
    obs, _ = env.reset()

    component_names = [
        "Position X (m)", "Position Y (m)", "Position Z (m)",  # pos[0:3]
        "Velocity X (m/s)", "Velocity Y (m/s)", "Velocity Z (m/s)",  # vel[0:3]
        "Attitude W", "Attitude X", "Attitude Y", "Attitude Z",  # quat[0:3]
        "Angular Rate X (rad/s)", "Angular Rate Y (rad/s)", "Angular Rate Z (rad/s)",  # ang_vel[0:3]
        "Mass (kg)", "Fuel Mass (kg)"
    ]

    print("Observation Vector:")
    for i, (name, value) in enumerate(zip(component_names, obs)):
        print(f"  [{i:2d}] {name:<20}: {value:8.3f}")

    env.close()

if __name__ == "__main__":
    demo_observation_components()
    demo_helicopter_policy()