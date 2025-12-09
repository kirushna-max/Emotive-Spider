"""
============================================================================
Real-Time Training Visualization for Emotive-Spider
============================================================================
This script watches the checkpoints directory and visualizes the latest
trained policy in real-time. Run this alongside train.py to watch progress.

Usage:
    Terminal 1: python train.py
    Terminal 2: python training_vis.py

Press ESC in the viewer window to stop.
============================================================================
"""

import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import mujoco
import mujoco.viewer

from train import ActorCritic, sample_action


def get_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> tuple:
    """
    Find and load the latest checkpoint.
    
    Returns:
        (params, update_num, mean_reward) or (None, 0, 0) if no checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Try latest.pkl first
    latest_file = checkpoint_path / "latest.pkl"
    if latest_file.exists():
        try:
            with open(latest_file, 'rb') as f:
                checkpoint = pickle.load(f)
            update_num = checkpoint.get('update', 0)
            rewards = checkpoint.get('rewards', [])
            mean_reward = np.mean(rewards[-10:]) if rewards else 0
            return checkpoint['params'], update_num, mean_reward
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
    
    # Fallback: find highest numbered checkpoint
    checkpoints = list(checkpoint_path.glob("checkpoint_*.pkl"))
    if checkpoints:
        # Sort by number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
        latest = checkpoints[-1]
        try:
            with open(latest, 'rb') as f:
                checkpoint = pickle.load(f)
            update_num = checkpoint.get('update', 0)
            rewards = checkpoint.get('rewards', [])
            mean_reward = np.mean(rewards[-10:]) if rewards else 0
            return checkpoint['params'], update_num, mean_reward
        except Exception as e:
            print(f"Error loading {latest}: {e}")
    
    return None, 0, 0


def detect_obs_size(params) -> int:
    """
    Detect observation size from checkpoint parameters.
    
    Returns:
        obs_size: Either 31 (old format) or 46 (new format)
    """
    # The first Dense layer kernel shape is (obs_size, 256)
    try:
        kernel_shape = params['params']['Dense_0']['kernel'].shape
        return kernel_shape[0]
    except:
        return 31  # Default to old format


def visualize_training(
    checkpoint_dir: str = "checkpoints",
    reload_interval: float = 10.0,  # Seconds between checkpoint reloads
):
    """
    Visualize the current trained policy, auto-reloading checkpoints.
    
    Args:
        checkpoint_dir: Directory containing training checkpoints
        reload_interval: How often to check for new checkpoints (seconds)
    """
    print("=" * 60)
    print("EMOTIVE-SPIDER TRAINING VISUALIZER")
    print("=" * 60)
    print(f"\nWatching checkpoint directory: {checkpoint_dir}")
    print(f"Reload interval: {reload_interval}s")
    print("\nWaiting for first checkpoint...")
    
    # Load model directly (not through env to avoid obs_size mismatch)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "urdf", "robot_converted.xml")
    mj_model = mujoco.MjModel.from_xml_path(urdf_path)
    mj_data = mujoco.MjData(mj_model)
    action_size = mj_model.nu
    
    # Track current checkpoint
    current_update = 0
    last_reload_time = 0
    obs_size = 31  # Will be updated based on checkpoint
    
    # Try to load initial checkpoint
    loaded_params, loaded_update, mean_reward = get_latest_checkpoint(checkpoint_dir)
    if loaded_params is not None:
        params = loaded_params
        current_update = loaded_update
        obs_size = detect_obs_size(params)
        print(f"\n✓ Loaded checkpoint {current_update} (reward: {mean_reward:.4f})")
        print(f"  Detected obs_size: {obs_size}")
    else:
        # Initialize with dummy params
        rng = random.PRNGKey(42)
        network = ActorCritic(action_size=action_size)
        dummy_obs = jnp.zeros((1, obs_size))
        params = network.init(rng, dummy_obs)
    
    # Create network with correct obs size
    network = ActorCritic(action_size=action_size)
    
    # Store current params in a mutable container for closure
    params_container = {'params': params, 'obs_size': obs_size}
    
    # JIT compile action function
    def get_action(rng, obs, params):
        action, _, _ = sample_action(rng, params, network, obs[None, :])
        return action[0]
    
    get_action_jit = jax.jit(get_action)
    
    rng = random.PRNGKey(42)
    
    print("\nLaunching viewer...")
    print("Controls: ESC to quit | Automatically reloads latest checkpoint")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Reset
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.2
        
        episode = 0
        step = 0
        episode_distance = 0
        start_x = 0
        prev_action = np.zeros(action_size)
        
        while viewer.is_running():
            current_time = time.time()
            
            # Check for new checkpoint periodically
            if current_time - last_reload_time > reload_interval:
                last_reload_time = current_time
                loaded_params, loaded_update, mean_reward = get_latest_checkpoint(checkpoint_dir)
                
                if loaded_params is not None and loaded_update > current_update:
                    params_container['params'] = loaded_params
                    current_update = loaded_update
                    new_obs_size = detect_obs_size(loaded_params)
                    
                    if new_obs_size != params_container['obs_size']:
                        params_container['obs_size'] = new_obs_size
                        print(f"  Obs size changed to: {new_obs_size}")
                    
                    print(f"✓ Loaded checkpoint {current_update} | "
                          f"Avg reward: {mean_reward:.4f} | "
                          f"Episode: {episode}")
            
            # Get observation based on detected format
            joint_pos = mj_data.qpos[7:]
            joint_vel = mj_data.qvel[6:]
            body_quat = mj_data.qpos[3:7]
            body_angvel = mj_data.qvel[3:6]
            
            if params_container['obs_size'] == 31:
                # Old format: joint_pos(12) + joint_vel(12) + quat(4) + angvel(3) = 31
                obs = jnp.concatenate([
                    jnp.array(joint_pos),
                    jnp.array(joint_vel),
                    jnp.array(body_quat),
                    jnp.array(body_angvel),
                ])
            else:
                # New format: + body_linvel(3) + prev_action(12) = 46
                body_linvel = mj_data.qvel[0:3]
                obs = jnp.concatenate([
                    jnp.array(joint_pos),
                    jnp.array(joint_vel),
                    jnp.array(body_quat),
                    jnp.array(body_angvel),
                    jnp.array(body_linvel),
                    jnp.array(prev_action),
                ])
            
            # Get action
            rng, key = random.split(rng)
            action = get_action_jit(key, obs, params_container['params'])
            action_np = np.array(action)
            prev_action = action_np
            
            # Apply action
            mj_data.ctrl[:] = action_np
            
            # Step simulation
            for _ in range(4):
                mujoco.mj_step(mj_model, mj_data)
            
            step += 1
            
            # Check termination (tilt > 45°)
            quat = mj_data.qpos[3:7]
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            z_up = 1.0 - 2.0 * (x*x + y*y)
            tilted = z_up < 0.707  # cos(45°)
            
            if tilted or step > 1000:
                # Episode ended
                episode_distance = mj_data.qpos[0] - start_x
                episode += 1
                
                reason = "TILTED" if tilted else "TIMEOUT"
                print(f"Episode {episode} | Steps: {step} | "
                      f"Distance: {episode_distance:.3f}m | {reason}")
                
                # Reset
                mujoco.mj_resetData(mj_model, mj_data)
                mj_data.qpos[2] = 0.2
                start_x = mj_data.qpos[0]
                step = 0
                prev_action = np.zeros(action_size)
            
            # Update viewer
            viewer.sync()
            
            # Control playback speed
            time.sleep(0.008)  # ~125 fps for smooth visualization
    
    print("\nVisualization stopped.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch training progress")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--reload-interval", type=float, default=10.0,
                        help="Seconds between checkpoint reloads")
    
    args = parser.parse_args()
    
    visualize_training(
        checkpoint_dir=args.checkpoint_dir,
        reload_interval=args.reload_interval,
    )
