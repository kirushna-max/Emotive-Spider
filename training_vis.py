"""
============================================================================
Real-Time Training Visualization for Emotive-Spider
============================================================================
This script allows you to watch the spider robot learn to walk in real-time.
It runs training iterations and periodically updates a MuJoCo viewer to show
the current policy's behavior.

Usage:
    python training_vis.py

Press ESC in the viewer window to stop training.
============================================================================
"""

import os
import pickle
import time
import threading
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import numpy as np
import mujoco
import mujoco.viewer

from env import SpiderEnv, VecSpiderEnv
from train import ActorCritic, sample_action, compute_gae, ppo_loss


# ============================================================================
# VISUALIZATION THREAD
# ============================================================================

class TrainingVisualizer:
    """
    Runs MuJoCo visualization in a separate thread while training continues.
    
    This allows you to see the robot's current behavior without blocking
    the training loop.
    """
    
    def __init__(self, env: SpiderEnv):
        """
        Initialize visualizer.
        
        Args:
            env: Spider environment (for model access)
        """
        self.env = env
        self.mj_model = env.get_mj_model()
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Current policy parameters (updated from training thread)
        self.current_params = None
        self.network = None
        
        # Control flags
        self.running = True
        self.update_ready = threading.Event()
        
    def set_network(self, network: ActorCritic, params: Any):
        """Update the current policy parameters."""
        self.network = network
        self.current_params = params
        self.update_ready.set()
    
    def run_visualization(self, rng: jax.Array):
        """
        Run the visualization loop.
        
        This runs in the main thread (required by MuJoCo viewer).
        """
        print("Starting visualization...")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # Reset simulation
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            self.mj_data.qpos[2] = 0.2  # Start height
            
            episode_reward = 0.0
            episode_steps = 0
            
            while viewer.is_running() and self.running:
                if self.current_params is not None and self.network is not None:
                    # Get observation from current state
                    obs = self._get_obs()
                    
                    # Sample action from policy
                    rng, action_key = random.split(rng)
                    action, _, _ = sample_action(
                        action_key, 
                        self.current_params, 
                        self.network, 
                        obs[None, :]
                    )
                    action = action[0]
                    
                    # Apply action
                    self.mj_data.ctrl[:] = np.array(action)  # gear=10 in XML provides scaling
                    
                    # Step simulation
                    for _ in range(4):  # action_repeat
                        mujoco.mj_step(self.mj_model, self.mj_data)
                    
                    # Update viewer
                    viewer.sync()
                    
                    # Track episode
                    episode_steps += 1
                    
                    # Check termination
                    if self.mj_data.qpos[2] < 0.05 or episode_steps > 500:
                        # Reset episode
                        mujoco.mj_resetData(self.mj_model, self.mj_data)
                        self.mj_data.qpos[2] = 0.2
                        episode_steps = 0
                else:
                    # No policy yet, just step with zero control
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    viewer.sync()
                
                # Small delay for smooth visualization
                time.sleep(0.01)
        
        self.running = False
        print("Visualization stopped.")
    
    def _get_obs(self) -> jnp.ndarray:
        """Extract observation from MuJoCo data."""
        joint_pos = self.mj_data.qpos[7:]
        joint_vel = self.mj_data.qvel[6:]
        body_quat = self.mj_data.qpos[3:7]
        body_angvel = self.mj_data.qvel[3:6]
        
        return jnp.concatenate([
            jnp.array(joint_pos),
            jnp.array(joint_vel),
            jnp.array(body_quat),
            jnp.array(body_angvel),
        ])
    
    def stop(self):
        """Stop the visualization."""
        self.running = False


# ============================================================================
# TRAINING WITH VISUALIZATION
# ============================================================================

def train_with_visualization(
    num_envs: int = 32,
    num_steps: int = 128,
    num_updates: int = 500,
    num_epochs: int = 4,
    minibatch_size: int = 128,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    vis_update_freq: int = 5,
):
    """
    Run PPO training with real-time visualization.
    
    Args:
        num_envs: Number of parallel environments
        num_steps: Rollout length
        num_updates: Number of policy updates
        vis_update_freq: How often to update visualization policy
        ... (other args same as train.py)
    """
    print("=" * 60)
    print("PPO TRAINING WITH REAL-TIME VISUALIZATION")
    print("=" * 60)
    
    # Initialize random key
    rng = random.PRNGKey(seed)
    
    # Create environment
    print("\nInitializing environment...")
    base_env = SpiderEnv()
    vec_env = VecSpiderEnv(base_env, num_envs)
    
    # Create network
    print("Creating neural network...")
    network = ActorCritic(action_size=base_env.action_size)
    
    # Initialize parameters
    rng, init_key = random.split(rng)
    dummy_obs = jnp.zeros((1, base_env.obs_size))
    params = network.init(init_key, dummy_obs)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params)
    
    # Create visualizer
    visualizer = TrainingVisualizer(base_env)
    visualizer.set_network(network, params)
    
    # JIT compile functions
    @jax.jit
    def get_action(rng, params, obs):
        return sample_action(rng, params, network, obs)
    
    @jax.jit
    def compute_advantages(rewards, values, dones, last_value):
        return compute_gae(rewards, values, dones, last_value, gamma, gae_lambda)
    
    @jax.jit
    def update_step(params, opt_state, batch):
        obs, actions, old_log_probs, advantages, returns = batch
        
        def loss_fn(p):
            return ppo_loss(
                p, network, obs, actions, old_log_probs,
                advantages, returns, clip_eps, vf_coef, ent_coef
            )
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, metrics
    
    # Start training in background thread
    def training_thread():
        nonlocal params, opt_state
        
        # Reset environments
        rng_train = random.PRNGKey(seed + 1)
        rng_train, reset_key = random.split(rng_train)
        obs, env_states = vec_env.reset(reset_key)
        
        for update in range(num_updates):
            if not visualizer.running:
                break
                
            # Collect rollout
            rollout_obs = []
            rollout_actions = []
            rollout_rewards = []
            rollout_dones = []
            rollout_values = []
            rollout_log_probs = []
            
            for step in range(num_steps):
                rng_train, action_key = random.split(rng_train)
                actions, log_probs, values = get_action(action_key, params, obs)
                
                next_obs, env_states, rewards, infos = vec_env.step(env_states, actions)
                dones = env_states.done
                
                rollout_obs.append(obs)
                rollout_actions.append(actions)
                rollout_rewards.append(rewards)
                rollout_dones.append(dones)
                rollout_values.append(values)
                rollout_log_probs.append(log_probs)
                
                # Reset done envs
                rng_train, reset_key = random.split(rng_train)
                reset_obs, reset_states = vec_env.reset(reset_key)
                obs = jnp.where(dones[:, None], reset_obs, next_obs)
                
                # Helper to broadcast dones to match any array shape
                def select_state(reset_val, current_val):
                    expanded_dones = dones
                    for _ in range(reset_val.ndim - 1):
                        expanded_dones = expanded_dones[..., None]
                    return jnp.where(expanded_dones, reset_val, current_val)
                
                env_states = jax.tree.map(select_state, reset_states, env_states)
            
            # Stack and compute advantages
            rollout_obs = jnp.stack(rollout_obs)
            rollout_actions = jnp.stack(rollout_actions)
            rollout_rewards = jnp.stack(rollout_rewards)
            rollout_dones = jnp.stack(rollout_dones)
            rollout_values = jnp.stack(rollout_values)
            rollout_log_probs = jnp.stack(rollout_log_probs)
            
            _, _, last_value = get_action(rng_train, params, obs)
            advantages, returns = compute_advantages(
                rollout_rewards, rollout_values, rollout_dones, last_value
            )
            
            # Update policy
            batch_size = num_envs * num_steps
            flat_obs = rollout_obs.reshape(batch_size, -1)
            flat_actions = rollout_actions.reshape(batch_size, -1)
            flat_log_probs = rollout_log_probs.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)
            
            for epoch in range(num_epochs):
                rng_train, shuffle_key = random.split(rng_train)
                perm = random.permutation(shuffle_key, batch_size)
                
                for start_idx in range(0, batch_size, minibatch_size):
                    end_idx = start_idx + minibatch_size
                    mb_idx = perm[start_idx:end_idx]
                    
                    batch = (
                        flat_obs[mb_idx],
                        flat_actions[mb_idx],
                        flat_log_probs[mb_idx],
                        flat_advantages[mb_idx],
                        flat_returns[mb_idx],
                    )
                    
                    params, opt_state, metrics = update_step(params, opt_state, batch)
            
            # Update visualization policy
            if (update + 1) % vis_update_freq == 0:
                visualizer.set_network(network, params)
                mean_reward = float(jnp.mean(rollout_rewards))
                print(f"Update {update + 1}/{num_updates} | Mean reward: {mean_reward:.4f}")
        
        print("Training complete!")
        visualizer.stop()
    
    # Start training thread
    train_thread = threading.Thread(target=training_thread)
    train_thread.start()
    
    # Run visualization in main thread (required by MuJoCo)
    rng, vis_key = random.split(rng)
    visualizer.run_visualization(vis_key)
    
    # Wait for training to finish
    train_thread.join()
    
    # Save final model
    print("\nSaving trained model...")
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/vis_trained.pkl", 'wb') as f:
        pickle.dump({'params': jax.device_get(params)}, f)
    print("Saved to checkpoints/vis_trained.pkl")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train_with_visualization()
