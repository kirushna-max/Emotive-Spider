"""
============================================================================
Trained Model Visualization for Emotive-Spider
============================================================================
This script loads a trained policy checkpoint and runs it in the MuJoCo
viewer for interactive visualization of the learned walking behavior.

Usage:
    python vis.py                                    # Use latest checkpoint
    python vis.py --checkpoint checkpoints/final.pkl  # Use specific checkpoint

Controls in viewer:
    - ESC: Close viewer
    - SPACE: Pause/unpause simulation
    - R: Reset episode
    - Mouse: Rotate/zoom camera
============================================================================
"""

import os
import pickle
import argparse
import time

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import mujoco
import mujoco.viewer

from env import SpiderEnv
from train import ActorCritic, sample_action


def visualize_policy(
    checkpoint_path: str = None,
    max_episodes: int = 10,
    episode_length: int = 1000,
    render_speed: float = 1.0,
):
    """
    Visualize a trained policy in MuJoCo viewer.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses latest.
        max_episodes: Maximum number of episodes to run
        episode_length: Maximum steps per episode
        render_speed: Playback speed multiplier (1.0 = real-time)
    """
    print("=" * 60)
    print("EMOTIVE-SPIDER TRAINED MODEL VISUALIZATION")
    print("=" * 60)
    
    # ========================================================================
    # LOAD CHECKPOINT
    # ========================================================================
    
    if checkpoint_path is None:
        # Look for latest checkpoint
        checkpoint_dir = "checkpoints"
        if os.path.exists(os.path.join(checkpoint_dir, "latest.pkl")):
            checkpoint_path = os.path.join(checkpoint_dir, "latest.pkl")
        elif os.path.exists(os.path.join(checkpoint_dir, "final.pkl")):
            checkpoint_path = os.path.join(checkpoint_dir, "final.pkl")
        elif os.path.exists(os.path.join(checkpoint_dir, "vis_trained.pkl")):
            checkpoint_path = os.path.join(checkpoint_dir, "vis_trained.pkl")
        else:
            print("ERROR: No checkpoint found!")
            print("Please train a model first with: python train.py")
            print("Or specify a checkpoint with: python vis.py --checkpoint path/to/checkpoint.pkl")
            return
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    
    if 'update' in checkpoint:
        print(f"Checkpoint from update: {checkpoint['update']}")
    if 'rewards' in checkpoint:
        final_rewards = checkpoint['rewards'][-10:] if len(checkpoint['rewards']) > 10 else checkpoint['rewards']
        print(f"Final mean reward: {np.mean(final_rewards):.4f}")
    
    # ========================================================================
    # SETUP ENVIRONMENT AND NETWORK
    # ========================================================================
    
    print("\nInitializing environment...")
    env = SpiderEnv()
    mj_model = env.get_mj_model()
    mj_data = mujoco.MjData(mj_model)
    
    # Create network
    network = ActorCritic(action_size=env.action_size)
    
    # Initialize random key
    rng = random.PRNGKey(42)
    
    # ========================================================================
    # RUN VISUALIZATION
    # ========================================================================
    
    print("\nLaunching viewer...")
    print("Controls:")
    print("  - ESC: Close viewer")
    print("  - SPACE: Pause/unpause")
    print("  - R: Reset episode")
    print("  - Mouse: Rotate/zoom camera")
    
    @jax.jit
    def get_action(rng, obs):
        action, _, _ = sample_action(rng, params, network, obs[None, :])
        return action[0]
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        episode = 0
        step = 0
        episode_reward = 0.0
        
        # Reset to initial state
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.2  # Start height
        
        print(f"\nStarting Episode {episode + 1}")
        
        while viewer.is_running() and episode < max_episodes:
            # Get observation
            obs = _get_obs(mj_data)
            
            # Get action from policy
            rng, action_key = random.split(rng)
            action = get_action(action_key, obs)
            
            # Apply action
            mj_data.ctrl[:] = np.array(action) * 10.0
            
            # Step simulation
            for _ in range(4):  # action_repeat
                mujoco.mj_step(mj_model, mj_data)
            
            step += 1
            
            # Compute approximate reward (for display)
            forward_vel = mj_data.qvel[0]
            episode_reward += forward_vel * 0.1  # Simplified reward
            
            # Check termination
            fallen = mj_data.qpos[2] < 0.05
            timeout = step >= episode_length
            
            if fallen or timeout:
                x_traveled = mj_data.qpos[0]
                print(f"Episode {episode + 1} finished | "
                      f"Steps: {step} | "
                      f"Distance: {x_traveled:.3f}m | "
                      f"{'FALLEN' if fallen else 'TIMEOUT'}")
                
                # Reset
                episode += 1
                step = 0
                episode_reward = 0.0
                
                if episode < max_episodes:
                    mujoco.mj_resetData(mj_model, mj_data)
                    mj_data.qpos[2] = 0.2
                    print(f"\nStarting Episode {episode + 1}")
                    time.sleep(0.5)  # Brief pause between episodes
            
            # Update viewer
            viewer.sync()
            
            # Control playback speed
            time.sleep(0.002 * 4 / render_speed)  # dt * action_repeat
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


def _get_obs(mj_data: mujoco.MjData) -> jnp.ndarray:
    """Extract observation from MuJoCo data."""
    joint_pos = mj_data.qpos[7:]
    joint_vel = mj_data.qvel[6:]
    body_quat = mj_data.qpos[3:7]
    body_angvel = mj_data.qvel[3:6]
    
    return jnp.concatenate([
        jnp.array(joint_pos),
        jnp.array(joint_vel),
        jnp.array(body_quat),
        jnp.array(body_angvel),
    ])


# ============================================================================
# DEMO MODE (No checkpoint required)
# ============================================================================

def run_random_demo():
    """
    Run demo with random actions (for testing before training).
    """
    print("=" * 60)
    print("EMOTIVE-SPIDER RANDOM POLICY DEMO")
    print("=" * 60)
    print("\nNo trained model - showing random actions")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "urdf", "robot_no_ground.urdf")
    
    print(f"Loading model from: {urdf_path}")
    mj_model = mujoco.MjModel.from_xml_path(urdf_path)
    mj_data = mujoco.MjData(mj_model)
    
    rng = random.PRNGKey(0)
    
    print("\nLaunching viewer with random policy...")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Reset
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.2
        
        step = 0
        
        while viewer.is_running():
            # Random action
            rng, key = random.split(rng)
            action = random.uniform(key, (mj_model.nu,), minval=-1.0, maxval=1.0)
            mj_data.ctrl[:] = np.array(action) * 10.0
            
            # Step
            for _ in range(4):
                mujoco.mj_step(mj_model, mj_data)
            
            step += 1
            
            # Reset if fallen
            if mj_data.qpos[2] < 0.05 or step > 500:
                mujoco.mj_resetData(mj_model, mj_data)
                mj_data.qpos[2] = 0.2
                step = 0
            
            viewer.sync()
            time.sleep(0.01)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained Emotive-Spider policy")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time)")
    parser.add_argument("--demo", action="store_true",
                        help="Run random policy demo (no checkpoint needed)")
    
    args = parser.parse_args()
    
    if args.demo:
        run_random_demo()
    else:
        visualize_policy(
            checkpoint_path=args.checkpoint,
            max_episodes=args.episodes,
            render_speed=args.speed,
        )
