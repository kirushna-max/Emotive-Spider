"""
============================================================================
PPO Training Script for Emotive-Spider Robot
============================================================================
This script implements Proximal Policy Optimization (PPO) to train the
spider robot to walk using MuJoCo MJX for accelerated physics simulation.

Key Features:
- PPO with Generalized Advantage Estimation (GAE)
- Vectorized environments for parallel rollouts
- Checkpointing for saving/resuming training
- Training metrics logging

Usage:
    python train.py                    # Standard training
    python train.py --test-mode        # Quick test (10 iterations)
    python train.py --resume checkpoint.pkl  # Resume from checkpoint
============================================================================
"""

import os
import pickle
import time
from typing import Tuple, Dict, Any, NamedTuple
from functools import partial
import argparse

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import numpy as np
import matplotlib.pyplot as plt

from env import SpiderEnv, VecSpiderEnv, EnvState
from env_stand import SpiderStandEnv, VecSpiderStandEnv


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    The actor outputs action mean and log standard deviation.
    The critic outputs a single value estimate.
    
    Architecture:
    - Shared: None (separate networks for actor and critic)
    - Actor: 2 hidden layers (256, 256) -> action mean + learned log_std
    - Critic: 2 hidden layers (256, 256) -> value
    """
    action_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through actor-critic network.
        
        Args:
            x: Observation tensor, shape (batch, obs_size)
        
        Returns:
            action_mean: Mean of action distribution, shape (batch, action_size)
            action_log_std: Log std of action distribution, shape (action_size,)
            value: Value estimate, shape (batch, 1)
        """
        # ====================================================================
        # ACTOR NETWORK
        # ====================================================================
        
        actor = nn.Dense(
            256, 
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        actor = nn.tanh(actor)
        
        actor = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(actor)
        actor = nn.tanh(actor)
        
        # Action mean output
        action_mean = nn.Dense(
            self.action_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor)
        
        # Learnable log standard deviation (state-independent)
        # Initialize to -0.5 for reasonable initial exploration
        action_log_std = self.param(
            'log_std',
            lambda key, shape: jnp.full(shape, -0.5),
            (self.action_size,)
        )
        
        # CRITICAL: Clamp log_std to prevent explosion or collapse
        # Min -2.0 (std=0.135) prevents over-confidence
        # Max 0.5 (std=1.65) prevents too much randomness
        action_log_std = jnp.clip(action_log_std, -2.0, 0.5)
        
        # ====================================================================
        # CRITIC NETWORK
        # ====================================================================
        
        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        critic = nn.tanh(critic)
        
        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(critic)
        critic = nn.tanh(critic)
        
        # Value output
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)
        
        return action_mean, action_log_std, value.squeeze(-1)


# ============================================================================
# TRAINING DATA STRUCTURES
# ============================================================================

class TrainState(NamedTuple):
    """Training state container."""
    params: Any
    opt_state: Any
    step: int


class RolloutBuffer(NamedTuple):
    """Buffer for storing rollout data."""
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray


# ============================================================================
# PPO TRAINING FUNCTIONS
# ============================================================================

def sample_action(
    rng: jax.Array,
    params: Any,
    network: ActorCritic,
    obs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample action from policy.
    
    Args:
        rng: Random key
        params: Network parameters
        network: ActorCritic network
        obs: Observation
    
    Returns:
        action: Sampled action
        log_prob: Log probability of action
        value: Value estimate
    """
    action_mean, action_log_std, value = network.apply(params, obs)
    
    # SAFETY: Clamp log_std to prevent numerical issues
    action_log_std = jnp.clip(action_log_std, -2.0, 0.5)
    action_std = jnp.exp(action_log_std)
    
    # Sample from Gaussian
    noise = random.normal(rng, action_mean.shape)
    action = action_mean + noise * action_std
    
    # Clip action to valid range
    action = jnp.clip(action, -1.0, 1.0)
    
    # Compute log probability (with epsilon for numerical stability)
    log_prob = -0.5 * jnp.sum(
        jnp.square((action - action_mean) / (action_std + 1e-8)) + 
        2 * action_log_std + 
        jnp.log(2 * jnp.pi),
        axis=-1
    )
    
    return action, log_prob, value


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Rewards from rollout, shape (num_steps, num_envs)
        values: Value estimates, shape (num_steps, num_envs)
        dones: Done flags, shape (num_steps, num_envs)
        last_value: Value at final state, shape (num_envs,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Advantage estimates, shape (num_steps, num_envs)
        returns: Return estimates, shape (num_steps, num_envs)
    """
    num_steps = rewards.shape[0]
    
    # Append last value for bootstrapping
    values_with_last = jnp.concatenate([values, last_value[None, :]], axis=0)
    
    def compute_gae_step(carry, t):
        gae = carry
        delta = (
            rewards[t] + 
            gamma * values_with_last[t + 1] * (1 - dones[t]) - 
            values_with_last[t]
        )
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae
    
    # Scan backwards through time
    _, advantages = jax.lax.scan(
        compute_gae_step,
        jnp.zeros_like(last_value),
        jnp.arange(num_steps - 1, -1, -1)
    )
    
    # Reverse to get correct time order
    advantages = advantages[::-1]
    returns = advantages + values
    
    return advantages, returns


def ppo_loss(
    params: Any,
    network: ActorCritic,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute PPO loss.
    
    Args:
        params: Network parameters
        network: ActorCritic network
        obs: Observations
        actions: Actions taken
        old_log_probs: Log probs from rollout
        advantages: Advantage estimates
        returns: Return estimates
        clip_eps: PPO clipping epsilon
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient
    
    Returns:
        loss: Total loss
        metrics: Dictionary of loss components
    """
    # Forward pass
    action_mean, action_log_std, values = network.apply(params, obs)
    
    # SAFETY: Clamp log_std to prevent numerical issues
    action_log_std = jnp.clip(action_log_std, -2.0, 0.5)
    action_std = jnp.exp(action_log_std)
    
    # Compute new log probabilities
    log_probs = -0.5 * jnp.sum(
        jnp.square((actions - action_mean) / (action_std + 1e-8)) + 
        2 * action_log_std + 
        jnp.log(2 * jnp.pi),
        axis=-1
    )
    
    # SAFETY: Clip log probability difference to prevent ratio explosion
    log_ratio = jnp.clip(log_probs - old_log_probs, -10.0, 10.0)
    ratio = jnp.exp(log_ratio)
    
    # SAFETY: Clip ratio explicitly as additional safeguard
    ratio = jnp.clip(ratio, 0.0, 10.0)
    
    # Normalize advantages with robust scaling
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages)
    advantages_normalized = (advantages - adv_mean) / (adv_std + 1e-8)
    
    # SAFETY: Clip normalized advantages to prevent extreme values
    advantages_normalized = jnp.clip(advantages_normalized, -10.0, 10.0)
    
    pg_loss1 = -advantages_normalized * ratio
    pg_loss2 = -advantages_normalized * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
    
    # SAFETY: Clip policy loss to prevent explosion
    policy_loss = jnp.clip(policy_loss, -100.0, 100.0)
    
    # Value function loss (no clipping to see actual values)
    value_loss = jnp.mean(jnp.square(values - returns))
    
    # Entropy bonus (encourage exploration)
    entropy = 0.5 * jnp.sum(1 + 2 * action_log_std + jnp.log(2 * jnp.pi))
    
    # Total loss
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    
    # SAFETY: Replace NaN with large loss to trigger recovery
    total_loss = jnp.where(jnp.isnan(total_loss), 100.0, total_loss)
    policy_loss = jnp.where(jnp.isnan(policy_loss), 100.0, policy_loss)
    value_loss = jnp.where(jnp.isnan(value_loss), 100.0, value_loss)
    
    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'total_loss': total_loss,
    }
    
    return total_loss, metrics


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(
    num_envs: int = 64,
    num_steps: int = 256,
    num_updates: int = 1000,
    num_epochs: int = 4,
    minibatch_size: int = 256,
    learning_rate: float = 1e-4,  # Reduced from 3e-4 to prevent gradient explosion
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 10,
    show_plot: bool = True,
    test_mode: bool = False,
    resume_checkpoint: str = None,
    mode: str = "walk",  # "walk" or "stand"
):
    """
    Main PPO training function.
    
    Args:
        num_envs: Number of parallel environments
        num_steps: Rollout length
        num_updates: Number of policy updates
        num_epochs: PPO epochs per update
        minibatch_size: Minibatch size for updates
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_eps: PPO clipping epsilon
        vf_coef: Value loss coefficient
        ent_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm
        seed: Random seed
        checkpoint_dir: Directory for checkpoints
        checkpoint_freq: Checkpoint frequency
        test_mode: If True, run only 10 iterations
        resume_checkpoint: Path to checkpoint file to resume from
    """
    print("=" * 60)
    print(f"PPO TRAINING FOR EMOTIVE-SPIDER ({mode.upper()} MODE)")
    print("=" * 60)
    
    # Validate mode
    if mode not in ["walk", "stand"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'walk' or 'stand'")
    
    if test_mode:
        num_updates = 10
        print("TEST MODE: Running only 10 updates")
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize random key
    rng = random.PRNGKey(seed)
    
    # Create environment based on mode
    print(f"\nInitializing {mode} environment...")
    if mode == "walk":
        base_env = SpiderEnv()
        vec_env = VecSpiderEnv(base_env, num_envs)
    else:  # mode == "stand"
        base_env = SpiderStandEnv()
        vec_env = VecSpiderStandEnv(base_env, num_envs)
    
    # Create network
    print("Creating neural network...")
    network = ActorCritic(action_size=base_env.action_size)
    
    # Initialize network parameters
    rng, init_key = random.split(rng)
    dummy_obs = jnp.zeros((1, base_env.obs_size))
    params = network.init(init_key, dummy_obs)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params)
    
    # Resume from checkpoint if specified
    start_update = 0
    total_episodes = 0
    total_rewards = []
    
    if resume_checkpoint is not None:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        with open(resume_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        opt_state = optimizer.init(params)  # Re-init optimizer with loaded params
        start_update = checkpoint.get('update', 0)
        total_episodes = checkpoint.get('total_episodes', 0)
        total_rewards = checkpoint.get('rewards', [])
        print(f"Resumed from update {start_update}, {total_episodes} episodes completed")
    
    # ========================================================================
    # JIT COMPILE FUNCTIONS
    # ========================================================================
    
    print("JIT compiling training functions...")
    
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
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\nStarting training...")
    start_time = time.time()
    
    # Reset environments
    rng, reset_key = random.split(rng)
    obs, env_states = vec_env.reset(reset_key)
    
    # Initialize counters if not resuming
    if resume_checkpoint is None:
        total_episodes = 0
        total_rewards = []
    
    # Initialize plotting data
    plot_rewards = []
    plot_policy_loss = []
    plot_value_loss = []
    
    # Setup live plotting
    if show_plot:
        plt.ion()  # Enable interactive mode
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Emotive-Spider Training Progress ({mode.upper()} mode)')
        
        # Initialize empty lines for each subplot
        line_reward, = axes[0].plot([], [], 'b-', linewidth=1.5)
        axes[0].set_xlabel('Update')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Reward per Update')
        axes[0].grid(True, alpha=0.3)
        
        line_policy, = axes[1].plot([], [], 'g-', linewidth=1.5)
        axes[1].set_xlabel('Update')
        axes[1].set_ylabel('Policy Loss')
        axes[1].set_title('Policy Loss')
        axes[1].grid(True, alpha=0.3)
        
        line_value, = axes[2].plot([], [], 'r-', linewidth=1.5)
        axes[2].set_xlabel('Update')
        axes[2].set_ylabel('Value Loss')
        axes[2].set_title('Value Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    for update in range(start_update, num_updates):
        update_start = time.time()
        
        # ====================================================================
        # COLLECT ROLLOUT
        # ====================================================================
        
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        rollout_log_probs = []
        
        for step in range(num_steps):
            # Sample actions
            rng, action_key = random.split(rng)
            actions, log_probs, values = get_action(action_key, params, obs)
            
            # Step environments
            next_obs, env_states, rewards, infos = vec_env.step(env_states, actions)
            dones = env_states.done
            
            # Store transition
            rollout_obs.append(obs)
            rollout_actions.append(actions)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)
            rollout_values.append(values)
            rollout_log_probs.append(log_probs)
            
            # Handle episode termination (reset done envs)
            rng, reset_key = random.split(rng)
            reset_obs, reset_states = vec_env.reset(reset_key)
            
            # Only reset environments that are done
            obs = jnp.where(dones[:, None], reset_obs, next_obs)
            
            # Helper to broadcast dones to match any array shape
            def select_state(reset_val, current_val):
                # Expand dones to match the number of dimensions
                expanded_dones = dones
                for _ in range(reset_val.ndim - 1):
                    expanded_dones = expanded_dones[..., None]
                return jnp.where(expanded_dones, reset_val, current_val)
            
            env_states = jax.tree.map(select_state, reset_states, env_states)
            
            # Track episode completions
            num_done = int(jnp.sum(dones))
            if num_done > 0:
                total_episodes += num_done
        
        # Stack rollout data
        rollout_obs = jnp.stack(rollout_obs)
        rollout_actions = jnp.stack(rollout_actions)
        rollout_rewards = jnp.stack(rollout_rewards)
        rollout_dones = jnp.stack(rollout_dones)
        rollout_values = jnp.stack(rollout_values)
        rollout_log_probs = jnp.stack(rollout_log_probs)
        
        # Get value of final state
        _, _, last_value = get_action(rng, params, obs)
        
        # ====================================================================
        # COMPUTE ADVANTAGES
        # ====================================================================
        
        advantages, returns = compute_advantages(
            rollout_rewards, rollout_values, rollout_dones, last_value
        )
        
        # ====================================================================
        # UPDATE POLICY
        # ====================================================================
        
        # Flatten batch
        batch_size = num_envs * num_steps
        flat_obs = rollout_obs.reshape(batch_size, -1)
        flat_actions = rollout_actions.reshape(batch_size, -1)
        flat_log_probs = rollout_log_probs.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)
        
        # Multiple epochs of updates
        all_metrics = []
        for epoch in range(num_epochs):
            # Shuffle data
            rng, shuffle_key = random.split(rng)
            perm = random.permutation(shuffle_key, batch_size)
            
            # Update in minibatches
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
                all_metrics.append(metrics)
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        
        update_time = time.time() - update_start
        mean_reward = float(jnp.mean(rollout_rewards))
        total_rewards.append(mean_reward)
        
        if True:  # Print every update
            avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            print(f"Update {update + 1}/{num_updates} | "
                  f"Reward: {mean_reward:.4f} | "
                  f"Policy Loss: {avg_metrics['policy_loss']:.4f} | "
                  f"Value Loss: {avg_metrics['value_loss']:.4f} | "
                  f"Episodes: {total_episodes} | "
                  f"Time: {update_time:.2f}s")
            
            # Update plot data
            plot_rewards.append(mean_reward)
            plot_policy_loss.append(float(avg_metrics['policy_loss']))
            plot_value_loss.append(float(avg_metrics['value_loss']))
            
            # Update live plot
            if show_plot and (update + 1) % 1 == 0:  # Update plot every update
                x_data = list(range(1, len(plot_rewards) + 1))
                
                # Update reward plot
                line_reward.set_data(x_data, plot_rewards)
                axes[0].relim()
                axes[0].autoscale_view()
                
                # Update policy loss plot
                line_policy.set_data(x_data, plot_policy_loss)
                axes[1].relim()
                axes[1].autoscale_view()
                
                # Update value loss plot  
                line_value.set_data(x_data, plot_value_loss)
                axes[2].relim()
                axes[2].autoscale_view()
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
        
        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        
        if (update + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update + 1}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': jax.device_get(params),
                    'opt_state': jax.device_get(opt_state),
                    'update': update + 1,
                    'total_episodes': total_episodes,
                    'rewards': total_rewards,
                }, f)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Also save latest
            latest_path = os.path.join(checkpoint_dir, "latest.pkl")
            with open(latest_path, 'wb') as f:
                pickle.dump({
                    'params': jax.device_get(params),
                    'opt_state': jax.device_get(opt_state),
                    'update': update + 1,
                    'total_episodes': total_episodes,
                    'rewards': total_rewards,
                }, f)
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total episodes: {total_episodes}")
    print(f"Final mean reward: {np.mean(total_rewards[-10:]):.4f}")
    
    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final.pkl")
    with open(final_path, 'wb') as f:
        pickle.dump({
            'params': jax.device_get(params),
            'opt_state': jax.device_get(opt_state),
            'update': num_updates,
            'total_episodes': total_episodes,
            'rewards': total_rewards,
        }, f)
    print(f"Saved final model to {final_path}")
    
    # Save final plot
    if show_plot:
        plot_path = os.path.join(checkpoint_dir, "training_progress.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {plot_path}")
        plt.ioff()
        plt.show()  # Keep plot open at end


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotive-Spider with PPO")
    parser.add_argument("--mode", type=str, default="walk", choices=["walk", "stand"],
                        help="Training mode: 'walk' for walking, 'stand' for standing")
    parser.add_argument("--test-mode", action="store_true", 
                        help="Run in test mode (10 iterations)")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="Rollout length")
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="Number of policy updates")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: checkpoints_walk or checkpoints_stand)")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Checkpoint frequency (updates)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable live plotting")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    
    args = parser.parse_args()
    
    # Set default checkpoint directory based on mode if not specified
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints_{args.mode}"
    
    train(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        show_plot=not args.no_plot,
        test_mode=args.test_mode,
        resume_checkpoint=args.resume,
        mode=args.mode,
    )
