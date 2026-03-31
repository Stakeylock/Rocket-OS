#!/usr/bin/env python3
"""
Example of training an RL agent on the Autonomous Rocket AI OS Gymnasium environment.

This example shows how to use Stable-Baselines3 to train a policy for rocket landing.
Note: Stable-Baselines3 needs to be installed separately: pip install stable-baselines3

Author: Jinitangsu Das (23011p0521@jntuhceh.ac.in) | GitHub: Stakeylock | Institution: JNTU Hyderabad
"""

def show_training_example():
    """Show example code for RL training (requires stable-baselines3)."""
    print("=== RL Training Example for Autonomous Rocket AI OS ===\n")
    print("This example shows how to train an RL agent using Stable-Baselines3.")
    print("To run this example, first install:\n")
    print("  pip install stable-baselines3[extra]")
    print()

    example_code = '''
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rocket_ai_os.gym_wrapper import RocketAviaryEnv, register_rocket_env

# Register the environment
register_rocket_env()

# Create vectorized environment for training
def make_env():
    return RocketAviaryEnv()

vec_env = make_vec_env(make_env, n_envs=4)

# Create and train the model
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./rocket_ppo_tensorboard/")
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("rocket_landing_ppo")

# Test the trained model
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    # Optionally render or log results

print("Training complete! Model saved as 'rocket_landing_ppo.zip")
'''

    print("Example code:")
    print("=" * 50)
    print(example_code)
    print("=" * 50)

    # Show what the actual environment looks like
    print("\nEnvironment Specification:")
    print("-" * 30)

    # Import and show environment details without actually training
    try:
        from rocket_ai_os.gym_wrapper import register_rocket_env, RocketAviaryEnv
        register_rocket_env()
        env = RocketAviaryEnv()

        print(f"Observation Space: {env.observation_space}")
        print(f"  Shape: {env.observation_space.shape}")
        print(f"  Type: {env.observation_space.dtype}")
        print()
        print(f"Action Space: {env.action_space}")
        print(f"  Shape: {env.action_space.shape}")
        print(f"  Bounds: low={env.action_space.low}, high={env.action_space.high}")
        print()
        print("Observation Components:")
        components = [
            "Pos X (m)", "Pos Y (m)", "Pos Z (m)",
            "Vel X (m/s)", "Vel Y (m/s)", "Vel Z (m/s)",
            "Att W", "Att X", "Att Y", "Att Z",
            "Ang Rate X (rad/s)", "Ang Rate Y (rad/s)", "Ang Rate Z (rad/s)",
            "Mass (kg)", "Fuel (kg)"
        ]
        obs, _ = env.reset()
        for i, (name, val) in enumerate(zip(components, obs)):
            print(f"  {name:<18}: {val:>8.3f}")

        env.close()

    except Exception as e:
        print(f"Error initializing environment: {e}")

if __name__ == "__main__":
    show_training_example()