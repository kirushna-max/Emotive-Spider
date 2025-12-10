"""
============================================================================
Population Based Reinforcement Learning (PBRL) for Emotive-Spider Robot
============================================================================
This script implements Population Based Training (PBT) combined with PPO
to automatically discover optimal hyperparameters while training.

Key Concepts:
- Population: N agents train in parallel with different hyperparameters
- Exploitation: Bottom performers copy weights from top performers
- Exploration: Copied agents get mutated hyperparameters

Algorithm Flow:
1. Initialize population of agents with varied hyperparameters
2. Train each agent using PPO for T steps
3. Every E steps: evaluate fitness (mean reward)
4. Exploit: Bottom 20% copy weights from top 20%
5. Explore: Mutate hyperparameters of copied agents
6. Repeat until convergence

Usage:
    python train_pbrl.py --stage 1 --population-size 8
    python train_pbrl.py --stage 1 --test-mode --population-size 4
    python train_pbrl.py --resume checkpoints/population_latest.pkl
============================================================================
"""

import os
import pickle
import time
from typing import Tuple, Dict, Any, NamedTuple, List
from functools import partial
from dataclasses import dataclass, field
import argparse
import copy

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import numpy as np
import matplotlib.pyplot as plt

from env_stand import SpiderStandEnv, VecSpiderStandEnv, EnvState


# ============================================================================
# NEURAL NETWORK ARCHITECTURE (same as train.py)
# ============================================================================

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Architecture:
    - Actor: 2 hidden layers (256, 256) -> action mean + learned log_std
    - Critic: 2 hidden layers (256, 256) -> value
    """
    action_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Actor network
        actor = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = nn.tanh(actor)
        
        action_mean = nn.Dense(
            self.action_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor)
        
        action_log_std = self.param(
            'log_std',
            lambda key, shape: jnp.full(shape, -0.5),
            (self.action_size,)
        )
        action_log_std = jnp.clip(action_log_std, -2.0, 0.5)
        
        # Critic network
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        
        return action_mean, action_log_std, value.squeeze(-1)


# ============================================================================
# PBRL DATA STRUCTURES
# ============================================================================

@dataclass
class AgentConfig:
    """
    Mutable hyperparameters for each population member.
    These are the parameters that PBRL will optimize.
    """
    learning_rate: float = 3e-4
    ent_coef: float = 0.02
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Bounds for mutations
    lr_bounds: Tuple[float, float] = field(default=(1e-5, 1e-3), repr=False)
    ent_bounds: Tuple[float, float] = field(default=(0.001, 0.1), repr=False)
    clip_bounds: Tuple[float, float] = field(default=(0.05, 0.4), repr=False)
    gamma_bounds: Tuple[float, float] = field(default=(0.9, 0.999), repr=False)


class PopulationMember(NamedTuple):
    """A single member of the population."""
    agent_id: int
    params: Any  # Network parameters
    opt_state: Any  # Optimizer state
    config: AgentConfig  # Mutable hyperparameters
    fitness_history: List[float]  # Reward history for fitness evaluation
    total_updates: int  # Number of PPO updates completed


# ============================================================================
# PBRL CORE FUNCTIONS
# ============================================================================

def create_initial_population(
    rng: jax.Array,
    network: ActorCritic,
    base_env: SpiderStandEnv,
    population_size: int,
    max_grad_norm: float = 0.5,
) -> List[PopulationMember]:
    """
    Create initial population with varied hyperparameters.
    
    Args:
        rng: Random key
        network: ActorCritic network architecture
        base_env: Environment for observation size
        population_size: Number of agents in population
        max_grad_norm: Gradient clipping norm
    
    Returns:
        List of PopulationMember with varied hyperparameters
    """
    population = []
    
    for i in range(population_size):
        # Split key for each agent
        rng, init_key, hp_key = random.split(rng, 3)
        
        # Initialize network parameters
        dummy_obs = jnp.zeros((1, base_env.obs_size))
        params = network.init(init_key, dummy_obs)
        
        # Create varied hyperparameters for each agent
        hp_keys = random.split(hp_key, 4)
        
        # Sample hyperparameters from log-uniform or uniform distributions
        learning_rate = float(10 ** random.uniform(hp_keys[0], (), minval=-4, maxval=-3))  # 1e-4 to 1e-3
        ent_coef = float(10 ** random.uniform(hp_keys[1], (), minval=-2.5, maxval=-1))  # ~0.003 to 0.1
        clip_eps = float(random.uniform(hp_keys[2], (), minval=0.1, maxval=0.3))
        gamma = float(random.uniform(hp_keys[3], (), minval=0.95, maxval=0.995))
        
        config = AgentConfig(
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_eps=clip_eps,
            gamma=gamma,
        )
        
        # Create optimizer with this agent's learning rate
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(config.learning_rate),
        )
        opt_state = optimizer.init(params)
        
        member = PopulationMember(
            agent_id=i,
            params=params,
            opt_state=opt_state,
            config=config,
            fitness_history=[],
            total_updates=0,
        )
        population.append(member)
        
        print(f"  Agent {i}: lr={config.learning_rate:.2e}, ent={config.ent_coef:.3f}, "
              f"clip={config.clip_eps:.2f}, gamma={config.gamma:.4f}")
    
    return population


def evaluate_fitness(member: PopulationMember, window: int = 10) -> float:
    """
    Evaluate fitness of a population member.
    
    Uses mean of recent rewards as fitness metric.
    
    Args:
        member: Population member to evaluate
        window: Number of recent rewards to consider
    
    Returns:
        Fitness score (higher is better)
    """
    if not member.fitness_history:
        return float('-inf')
    
    recent = member.fitness_history[-window:]
    return np.mean(recent)


def exploit(
    population: List[PopulationMember],
    truncation_ratio: float = 0.2,
    max_grad_norm: float = 0.5,
) -> List[PopulationMember]:
    """
    Exploitation step: Bottom performers copy from top performers.
    
    Uses truncation selection: bottom X% copy weights from top X%.
    
    Args:
        population: Current population
        truncation_ratio: Fraction of population to replace
    
    Returns:
        Updated population with copied weights
    """
    pop_size = len(population)
    num_to_replace = max(1, int(pop_size * truncation_ratio))
    
    # Rank by fitness
    fitness_scores = [(i, evaluate_fitness(m)) for i, m in enumerate(population)]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)  # High to low
    
    top_indices = [idx for idx, _ in fitness_scores[:num_to_replace]]
    bottom_indices = [idx for idx, _ in fitness_scores[-num_to_replace:]]
    
    # Report rankings
    print("\n  [EXPLOIT] Population Fitness Ranking:")
    for rank, (idx, score) in enumerate(fitness_scores):
        status = "TOP" if idx in top_indices else ("BOTTOM" if idx in bottom_indices else "")
        print(f"    Rank {rank+1}: Agent {idx} -> fitness={score:.4f} {status}")
    
    # Copy weights from top to bottom
    new_population = list(population)
    for bottom_idx, top_idx in zip(bottom_indices, top_indices):
        top_member = population[top_idx]
        bottom_member = population[bottom_idx]
        
        print(f"  [EXPLOIT] Agent {bottom_idx} copies weights from Agent {top_idx}")
        
        # Create new optimizer state for the copied params
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(bottom_member.config.learning_rate),
        )
        new_opt_state = optimizer.init(top_member.params)
        
        new_population[bottom_idx] = PopulationMember(
            agent_id=bottom_member.agent_id,
            params=copy.deepcopy(jax.device_get(top_member.params)),  # Copy weights
            opt_state=new_opt_state,
            config=bottom_member.config,  # Keep own config (will be mutated in explore)
            fitness_history=[],  # Reset history after copying
            total_updates=bottom_member.total_updates,
        )
    
    return new_population


def explore(
    rng: jax.Array,
    population: List[PopulationMember],
    copied_indices: List[int],
    mutation_strength: float = 0.2,
    max_grad_norm: float = 0.5,
) -> List[PopulationMember]:
    """
    Exploration step: Mutate hyperparameters of copied agents.
    
    Args:
        rng: Random key
        population: Current population
        copied_indices: Indices of agents that were copied (need mutation)
        mutation_strength: Relative mutation factor
    
    Returns:
        Updated population with mutated hyperparameters
    """
    new_population = list(population)
    
    for idx in copied_indices:
        rng, key = random.split(rng)
        member = population[idx]
        config = member.config
        
        # Mutate each hyperparameter
        keys = random.split(key, 4)
        
        # Learning rate: multiply by random factor
        lr_factor = float(np.exp(random.normal(keys[0]) * mutation_strength))
        new_lr = np.clip(config.learning_rate * lr_factor, *config.lr_bounds)
        
        # Entropy coefficient: multiply by random factor
        ent_factor = float(np.exp(random.normal(keys[1]) * mutation_strength))
        new_ent = np.clip(config.ent_coef * ent_factor, *config.ent_bounds)
        
        # Clip epsilon: add small perturbation
        clip_delta = float(random.normal(keys[2]) * 0.05)
        new_clip = np.clip(config.clip_eps + clip_delta, *config.clip_bounds)
        
        # Gamma: add small perturbation
        gamma_delta = float(random.normal(keys[3]) * 0.01)
        new_gamma = np.clip(config.gamma + gamma_delta, *config.gamma_bounds)
        
        new_config = AgentConfig(
            learning_rate=new_lr,
            ent_coef=new_ent,
            clip_eps=new_clip,
            gamma=new_gamma,
        )
        
        print(f"  [EXPLORE] Agent {idx} hyperparameters mutated:")
        print(f"    lr: {config.learning_rate:.2e} -> {new_lr:.2e}")
        print(f"    ent: {config.ent_coef:.3f} -> {new_ent:.3f}")
        print(f"    clip: {config.clip_eps:.2f} -> {new_clip:.2f}")
        print(f"    gamma: {config.gamma:.4f} -> {new_gamma:.4f}")
        
        # Create new optimizer with updated learning rate
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(new_config.learning_rate),
        )
        new_opt_state = optimizer.init(member.params)
        
        new_population[idx] = PopulationMember(
            agent_id=member.agent_id,
            params=member.params,
            opt_state=new_opt_state,
            config=new_config,
            fitness_history=member.fitness_history,
            total_updates=member.total_updates,
        )
    
    return new_population


# ============================================================================
# PPO TRAINING FUNCTIONS (adapted from train.py)
# ============================================================================

def sample_action(
    rng: jax.Array,
    params: Any,
    network: ActorCritic,
    obs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample action from policy."""
    action_mean, action_log_std, value = network.apply(params, obs)
    action_log_std = jnp.clip(action_log_std, -2.0, 0.5)
    action_std = jnp.exp(action_log_std)
    
    noise = random.normal(rng, action_mean.shape)
    action = action_mean + noise * action_std
    action = jnp.clip(action, -1.0, 1.0)
    
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
    """Compute Generalized Advantage Estimation."""
    num_steps = rewards.shape[0]
    values_with_last = jnp.concatenate([values, last_value[None, :]], axis=0)
    
    def compute_gae_step(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values_with_last[t + 1] * (1 - dones[t]) - values_with_last[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae
    
    _, advantages = jax.lax.scan(
        compute_gae_step,
        jnp.zeros_like(last_value),
        jnp.arange(num_steps - 1, -1, -1)
    )
    
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
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute PPO loss with agent-specific hyperparameters."""
    action_mean, action_log_std, values = network.apply(params, obs)
    action_log_std = jnp.clip(action_log_std, -2.0, 0.5)
    action_std = jnp.exp(action_log_std)
    
    log_probs = -0.5 * jnp.sum(
        jnp.square((actions - action_mean) / (action_std + 1e-8)) + 
        2 * action_log_std + 
        jnp.log(2 * jnp.pi),
        axis=-1
    )
    
    log_ratio = jnp.clip(log_probs - old_log_probs, -10.0, 10.0)
    ratio = jnp.clip(jnp.exp(log_ratio), 0.0, 10.0)
    
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages)
    advantages_normalized = jnp.clip((advantages - adv_mean) / (adv_std + 1e-8), -10.0, 10.0)
    
    pg_loss1 = -advantages_normalized * ratio
    pg_loss2 = -advantages_normalized * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = jnp.clip(jnp.mean(jnp.maximum(pg_loss1, pg_loss2)), -100.0, 100.0)
    
    value_loss = jnp.mean(jnp.square(values - returns))
    entropy = 0.5 * jnp.sum(1 + 2 * action_log_std + jnp.log(2 * jnp.pi))
    
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    total_loss = jnp.where(jnp.isnan(total_loss), 100.0, total_loss)
    
    return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy}


# ============================================================================
# MAIN PBRL TRAINING LOOP
# ============================================================================

def train_population(
    population_size: int = 8,
    num_envs_per_agent: int = 64,
    num_steps: int = 256,
    num_updates: int = 500,
    num_epochs: int = 4,
    minibatch_size: int = 256,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    exploit_interval: int = 20,  # Exploit/explore every N updates
    truncation_ratio: float = 0.2,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_freq: int = 50,
    show_plot: bool = True,
    test_mode: bool = False,
    stage: int = 1,
    resume_checkpoint: str = None,
):
    """
    Main PBRL training function.
    
    Args:
        population_size: Number of agents in population
        num_envs_per_agent: Parallel environments per agent
        num_steps: Rollout length
        num_updates: Total PPO updates per agent
        num_epochs: PPO epochs per update
        minibatch_size: Minibatch size
        vf_coef: Value loss coefficient
        max_grad_norm: Gradient clipping
        exploit_interval: Updates between exploit/explore cycles
        truncation_ratio: Fraction of population to replace
        seed: Random seed
        checkpoint_dir: Directory for checkpoints
        checkpoint_freq: Checkpoint frequency
        show_plot: Show live plot
        test_mode: Quick test with few updates
        stage: Curriculum stage (1=stand, 2=walk)
        resume_checkpoint: Path to resume from
    """
    print("=" * 70)
    print("POPULATION BASED REINFORCEMENT LEARNING (PBRL)")
    print(f"Stage {stage}: {'STAND' if stage == 1 else 'WALK'}")
    print("=" * 70)
    
    if test_mode:
        num_updates = 15
        exploit_interval = 5
        print("TEST MODE: Running 15 updates with exploit every 5")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    rng = random.PRNGKey(seed)
    
    # Create base environment
    print("\nInitializing environment...")
    base_env = SpiderStandEnv()
    vec_env = VecSpiderStandEnv(base_env, num_envs_per_agent)
    
    # Create network
    network = ActorCritic(action_size=base_env.action_size)
    
    # Initialize population
    print(f"\nInitializing population of {population_size} agents:")
    rng, pop_key = random.split(rng)
    
    if resume_checkpoint:
        print(f"Loading population from: {resume_checkpoint}")
        with open(resume_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        population = checkpoint['population']
        start_update = checkpoint.get('update', 0)
        print(f"Resumed from update {start_update}")
    else:
        population = create_initial_population(pop_key, network, base_env, population_size, max_grad_norm)
        start_update = 0
    
    # JIT compile functions
    print("\nJIT compiling training functions...")
    
    @jax.jit
    def get_action(rng, params, obs):
        return sample_action(rng, params, network, obs)
    
    # Create env state for each agent
    rng_keys = random.split(rng, population_size + 1)
    rng = rng_keys[0]
    env_states = [vec_env.reset(rng_keys[i + 1]) for i in range(population_size)]
    
    # Plotting setup
    all_fitness = {i: [] for i in range(population_size)}
    
    if show_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel('Update')
        ax.set_ylabel('Mean Reward (Fitness)')
        ax.set_title('PBRL Training Progress')
        ax.grid(True, alpha=0.3)
        lines = {}
        for i in range(population_size):
            lines[i], = ax.plot([], [], label=f'Agent {i}', alpha=0.7)
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    # Main training loop
    print("\nStarting PBRL training...")
    start_time = time.time()
    
    for update in range(start_update, num_updates):
        update_start = time.time()
        
        # Train each agent for one update
        for agent_idx, member in enumerate(population):
            obs, env_state = env_states[agent_idx]
            config = member.config
            
            # Create optimizer for this agent
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(config.learning_rate),
            )
            
            @jax.jit
            def compute_advantages(rewards, values, dones, last_value):
                return compute_gae(rewards, values, dones, last_value, config.gamma, config.gae_lambda)
            
            # Collect rollout
            rollout_obs, rollout_actions, rollout_rewards = [], [], []
            rollout_dones, rollout_values, rollout_log_probs = [], [], []
            
            params = member.params
            
            for step in range(num_steps):
                rng, action_key = random.split(rng)
                actions, log_probs, values = get_action(action_key, params, obs)
                
                next_obs, env_state, rewards, infos = vec_env.step(env_state, actions)
                dones = env_state.done
                
                rollout_obs.append(obs)
                rollout_actions.append(actions)
                rollout_rewards.append(rewards)
                rollout_dones.append(dones)
                rollout_values.append(values)
                rollout_log_probs.append(log_probs)
                
                # Handle resets
                rng, reset_key = random.split(rng)
                reset_obs, reset_states = vec_env.reset(reset_key)
                obs = jnp.where(dones[:, None], reset_obs, next_obs)
                
                def select_state(reset_val, current_val):
                    expanded_dones = dones
                    for _ in range(reset_val.ndim - 1):
                        expanded_dones = expanded_dones[..., None]
                    return jnp.where(expanded_dones, reset_val, current_val)
                
                env_state = jax.tree.map(select_state, reset_states, env_state)
            
            env_states[agent_idx] = (obs, env_state)
            
            # Stack rollout
            rollout_obs = jnp.stack(rollout_obs)
            rollout_actions = jnp.stack(rollout_actions)
            rollout_rewards = jnp.stack(rollout_rewards)
            rollout_dones = jnp.stack(rollout_dones)
            rollout_values = jnp.stack(rollout_values)
            rollout_log_probs = jnp.stack(rollout_log_probs)
            
            _, _, last_value = get_action(rng, params, obs)
            advantages, returns = compute_advantages(rollout_rewards, rollout_values, rollout_dones, last_value)
            
            # PPO update
            batch_size = num_envs_per_agent * num_steps
            flat_obs = rollout_obs.reshape(batch_size, -1)
            flat_actions = rollout_actions.reshape(batch_size, -1)
            flat_log_probs = rollout_log_probs.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)
            
            opt_state = member.opt_state
            
            for epoch in range(num_epochs):
                rng, shuffle_key = random.split(rng)
                perm = random.permutation(shuffle_key, batch_size)
                
                for start_idx in range(0, batch_size, minibatch_size):
                    mb_idx = perm[start_idx:start_idx + minibatch_size]
                    
                    def loss_fn(p):
                        return ppo_loss(
                            p, network, 
                            flat_obs[mb_idx], flat_actions[mb_idx],
                            flat_log_probs[mb_idx], flat_advantages[mb_idx],
                            flat_returns[mb_idx], config.clip_eps, vf_coef, config.ent_coef
                        )
                    
                    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
            
            # Update member
            mean_reward = float(jnp.mean(rollout_rewards))
            new_history = list(member.fitness_history) + [mean_reward]
            
            population[agent_idx] = PopulationMember(
                agent_id=agent_idx,
                params=params,
                opt_state=opt_state,
                config=config,
                fitness_history=new_history,
                total_updates=member.total_updates + 1,
            )
            
            all_fitness[agent_idx].append(mean_reward)
        
        # Logging
        update_time = time.time() - update_start
        fitnesses = [evaluate_fitness(m) for m in population]
        best_idx = int(np.argmax(fitnesses))
        
        print(f"Update {update + 1}/{num_updates} | "
              f"Best: Agent {best_idx} ({fitnesses[best_idx]:.4f}) | "
              f"Pop avg: {np.mean(fitnesses):.4f} | "
              f"Time: {update_time:.2f}s")
        
        # Update plot
        if show_plot and (update + 1) % 1 == 0:
            for i in range(population_size):
                x_data = list(range(1, len(all_fitness[i]) + 1))
                lines[i].set_data(x_data, all_fitness[i])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
        
        # Exploit/Explore cycle
        if (update + 1) % exploit_interval == 0 and update + 1 < num_updates:
            print(f"\n{'='*50}")
            print(f"EXPLOIT/EXPLORE CYCLE at update {update + 1}")
            print(f"{'='*50}")
            
            # Track which agents get copied
            fitness_scores = [(i, evaluate_fitness(m)) for i, m in enumerate(population)]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            num_to_replace = max(1, int(population_size * truncation_ratio))
            copied_indices = [idx for idx, _ in fitness_scores[-num_to_replace:]]
            
            # Exploit
            population = exploit(population, truncation_ratio, max_grad_norm)
            
            # Explore
            rng, explore_key = random.split(rng)
            population = explore(explore_key, population, copied_indices, max_grad_norm=max_grad_norm)
            
            print(f"{'='*50}\n")
        
        # Checkpointing
        if (update + 1) % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"population_stage{stage}_{update + 1}.pkl")
            with open(ckpt_path, 'wb') as f:
                pickle.dump({
                    'population': [(m.agent_id, jax.device_get(m.params), m.config, m.fitness_history, m.total_updates) 
                                   for m in population],
                    'update': update + 1,
                    'stage': stage,
                }, f)
            print(f"Saved checkpoint to {ckpt_path}")
            
            # Save latest
            latest_path = os.path.join(checkpoint_dir, "population_latest.pkl")
            with open(latest_path, 'wb') as f:
                pickle.dump({
                    'population': population,
                    'update': update + 1,
                    'stage': stage,
                }, f)
    
    # Save final results
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PBRL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time:.2f}s")
    
    # Find best agent
    final_fitnesses = [evaluate_fitness(m) for m in population]
    best_idx = int(np.argmax(final_fitnesses))
    best_member = population[best_idx]
    
    print(f"\nBest Agent: {best_idx}")
    print(f"  Fitness: {final_fitnesses[best_idx]:.4f}")
    print(f"  Learning rate: {best_member.config.learning_rate:.2e}")
    print(f"  Entropy coef: {best_member.config.ent_coef:.4f}")
    print(f"  Clip epsilon: {best_member.config.clip_eps:.3f}")
    print(f"  Gamma: {best_member.config.gamma:.4f}")
    
    # Save best agent as standard checkpoint (compatible with train.py)
    best_path = os.path.join(checkpoint_dir, f"pbrl_best_stage{stage}.pkl")
    with open(best_path, 'wb') as f:
        pickle.dump({
            'params': jax.device_get(best_member.params),
            'update': num_updates,
            'stage': stage,
            'best_hyperparams': {
                'learning_rate': best_member.config.learning_rate,
                'ent_coef': best_member.config.ent_coef,
                'clip_eps': best_member.config.clip_eps,
                'gamma': best_member.config.gamma,
            }
        }, f)
    print(f"\nSaved best agent to {best_path}")
    
    if show_plot:
        plot_path = os.path.join(checkpoint_dir, "pbrl_training_progress.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {plot_path}")
        plt.ioff()
        plt.show()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBRL Training for Emotive-Spider")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Curriculum stage: 1=stand, 2=walk")
    parser.add_argument("--population-size", type=int, default=8,
                        help="Number of agents in population")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Parallel environments per agent")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="Rollout length")
    parser.add_argument("--num-updates", type=int, default=500,
                        help="Total PPO updates")
    parser.add_argument("--exploit-interval", type=int, default=20,
                        help="Updates between exploit/explore cycles")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                        help="Checkpoint frequency")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable live plotting")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run quick test (15 updates)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to population checkpoint to resume")
    
    args = parser.parse_args()
    
    train_population(
        population_size=args.population_size,
        num_envs_per_agent=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        exploit_interval=args.exploit_interval,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        show_plot=not args.no_plot,
        test_mode=args.test_mode,
        stage=args.stage,
        resume_checkpoint=args.resume,
    )
