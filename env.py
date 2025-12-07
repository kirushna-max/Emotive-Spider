"""
============================================================================
Emotive-Spider Training Environment for MuJoCo MJX
============================================================================
This module defines the training environment for the 4-legged spider robot.
It uses MuJoCo MJX for JAX-based accelerated physics simulation.

The environment follows a gym-like interface but is designed for vectorized
simulation with JAX for efficient parallel training.

Key Components:
- SpiderEnv: Main environment class
- Observation: Joint positions, velocities, body orientation
- Actions: 12 motor torques (one per actuator)
- Rewards: Forward velocity + stability - joint limit penalties
============================================================================
"""

import os
import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
from typing import Tuple, Dict, Any, NamedTuple
from functools import partial


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class EnvState(NamedTuple):
    """
    Container for environment state that needs to be passed through JAX functions.
    
    Attributes:
        mjx_data: MJX simulation data containing physics state
        step_count: Current step in the episode
        done: Whether the episode has terminated
        prev_pos: Previous x-position of the robot (for velocity reward)
    """
    mjx_data: mjx.Data
    step_count: jnp.ndarray
    done: jnp.ndarray
    prev_pos: jnp.ndarray


class Transition(NamedTuple):
    """
    Container for a single environment transition (for PPO training).
    
    Attributes:
        obs: Observation at current timestep
        action: Action taken
        reward: Reward received
        next_obs: Observation at next timestep
        done: Whether episode terminated
    """
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

class SpiderEnv:
    """
    MJX-based training environment for the Emotive-Spider robot.
    
    This environment teaches the spider robot to walk forward while
    maintaining stability and respecting joint limits.
    
    Joint Configuration (12 total):
    - Leg 1: sh_roll_1, sh_yaw_1, kn_yaw_1
    - Leg 2: sh_roll_2, sh_yaw_2, kn_yaw_2
    - Leg 3: sh_roll_3, sh_yaw_3, kn_yaw_3
    - Leg 4: sh_roll_4, sh_yaw_4, kn_yaw_4
    
    Reward Function:
    1. Forward velocity reward (positive)
    2. Alive bonus (positive, for not falling)
    3. sh_roll boundary penalty (< -1.25 or > 1.25 radians)
    4. sh_yaw boundary penalty (> 1.55 radians)
    5. Symmetry penalty: |sh_roll_1 + sh_roll_3| and |sh_roll_2 + sh_roll_4|
    6. Energy penalty (optional, for smooth movements)
    """
    
    def __init__(
        self,
        urdf_path: str = None,
        episode_length: int = 1000,
        dt: float = 0.002,
        action_repeat: int = 4,
        forward_reward_weight: float = 5.0,
        alive_bonus: float = 0.01,
        sh_roll_penalty_weight: float = 1.0,
        sh_yaw_penalty_weight: float = 1.0,
        symmetry_penalty_weight: float = 0.5,
        energy_penalty_weight: float = 0.01,
    ):
        """
        Initialize the Spider training environment.
        
        Args:
            urdf_path: Path to the robot URDF file. If None, uses default path.
            episode_length: Maximum steps per episode.
            dt: Simulation timestep (seconds).
            action_repeat: Number of simulation steps per action.
            forward_reward_weight: Weight for forward velocity reward.
            alive_bonus: Bonus reward for staying upright.
            sh_roll_penalty_weight: Weight for sh_roll limit penalty.
            sh_yaw_penalty_weight: Weight for sh_yaw limit penalty.
            symmetry_penalty_weight: Weight for leg symmetry penalty.
            energy_penalty_weight: Weight for energy/torque penalty.
        """
        # ====================================================================
        # LOAD MUJOCO MODEL
        # ====================================================================
        
        if urdf_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(current_dir, "urdf", "robot_converted.xml")
        
        print(f"Loading MuJoCo model from: {urdf_path}")
        
        # Load the MuJoCo model from URDF
        self.mj_model = mujoco.MjModel.from_xml_path(urdf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # ====================================================================
        # CONVERT TO MJX (JAX-accelerated MuJoCo)
        # ====================================================================
        
        print("Converting to MJX for JAX acceleration...")
        self.mjx_model = mjx.put_model(self.mj_model)
        
        # ====================================================================
        # ENVIRONMENT PARAMETERS
        # ====================================================================
        
        self.episode_length = episode_length
        self.dt = dt
        self.action_repeat = action_repeat
        
        # Reward weights
        self.forward_reward_weight = forward_reward_weight
        self.alive_bonus = alive_bonus
        self.sh_roll_penalty_weight = sh_roll_penalty_weight
        self.sh_yaw_penalty_weight = sh_yaw_penalty_weight
        self.symmetry_penalty_weight = symmetry_penalty_weight
        self.energy_penalty_weight = energy_penalty_weight
        
        # ====================================================================
        # ACTION AND OBSERVATION SPACES
        # ====================================================================
        
        # 12 actuators (motors)
        self.action_size = self.mj_model.nu
        print(f"Action size: {self.action_size}")
        
        # Observation: joint positions (12) + joint velocities (12) + 
        # body orientation quaternion (4) + body angular velocity (3)
        # Total: 12 + 12 + 4 + 3 = 31
        self.obs_size = 31
        print(f"Observation size: {self.obs_size}")
        
        # ====================================================================
        # JOINT INDICES FOR REWARD COMPUTATION
        # ====================================================================
        
        # We need to identify which joints are sh_roll and sh_yaw
        # Joint order from URDF analysis:
        # The qpos indices for revolute joints (after the floating base):
        # Floating base uses 7 qpos (3 pos + 4 quat)
        # Then each revolute joint uses 1 qpos
        
        # Get joint names and find their qpos indices
        self._setup_joint_indices()
        
        print("Environment initialized successfully!")
    
    def _setup_joint_indices(self):
        """
        Setup joint indices for reward computation.
        
        This method identifies which qpos indices correspond to:
        - sh_roll_1, sh_roll_2, sh_roll_3, sh_roll_4
        - sh_yaw_1, sh_yaw_2, sh_yaw_3, sh_yaw_4
        
        These indices are used in the reward function to penalize
        positions outside the desired ranges.
        """
        # ====================================================================
        # BUILD JOINT NAME TO INDEX MAPPING
        # ====================================================================
        
        joint_names = []
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_names.append(name)
        
        print(f"Found joints: {joint_names}")
        
        # ====================================================================
        # IDENTIFY SH_ROLL JOINT INDICES
        # ====================================================================
        
        # sh_roll joints - we need their qpos indices
        # For floating base: first joint has qposadr=0, uses 7 qpos
        # Subsequent revolute joints each use 1 qpos
        
        self.sh_roll_indices = []
        self.sh_yaw_indices = []
        
        for i, name in enumerate(joint_names):
            if name is None:
                continue
            
            # Get qpos address for this joint
            qpos_idx = self.mj_model.jnt_qposadr[i]
            
            if 'sh_roll' in name:
                self.sh_roll_indices.append(qpos_idx)
                print(f"  sh_roll joint '{name}' at qpos index {qpos_idx}")
            elif 'sh_yaw' in name:
                self.sh_yaw_indices.append(qpos_idx)
                print(f"  sh_yaw joint '{name}' at qpos index {qpos_idx}")
        
        # Convert to JAX arrays for efficient computation
        self.sh_roll_indices = jnp.array(self.sh_roll_indices, dtype=jnp.int32)
        self.sh_yaw_indices = jnp.array(self.sh_yaw_indices, dtype=jnp.int32)
        
        print(f"sh_roll indices: {self.sh_roll_indices}")
        print(f"sh_yaw indices: {self.sh_yaw_indices}")
    
    # ========================================================================
    # CORE ENVIRONMENT METHODS
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, EnvState]:
        """
        Reset the environment to initial state.
        
        This method:
        1. Resets the robot to its initial pose
        2. Adds small random perturbations to joint positions
        3. Returns the initial observation
        
        Args:
            rng: JAX random key for reproducibility
        
        Returns:
            obs: Initial observation
            state: Initial environment state
        """
        # ====================================================================
        # CREATE FRESH MJX DATA
        # ====================================================================
        
        # Start with clean data from model
        mjx_data = mjx.make_data(self.mjx_model)
        
        # ====================================================================
        # ADD SMALL RANDOM PERTURBATIONS
        # ====================================================================
        
        # Split random key
        rng, key = random.split(rng)
        
        # Add small noise to joint positions (excluding floating base)
        # Floating base is first 7 qpos elements (3 pos + 4 quat)
        noise = random.uniform(key, (self.mj_model.nq - 7,), minval=-0.01, maxval=0.01)
        
        # Create new qpos with noise added to joint positions
        new_qpos = mjx_data.qpos.at[7:].add(noise)
        
        # Set initial height (z-position) to avoid ground collision
        new_qpos = new_qpos.at[2].set(0.2)  # Start at 20cm height
        
        # Update mjx_data with new positions
        mjx_data = mjx_data.replace(qpos=new_qpos)
        
        # ====================================================================
        # STEP SIMULATION TO SETTLE
        # ====================================================================
        
        # Do a few physics steps to settle the robot
        mjx_data = mjx.step(self.mjx_model, mjx_data)
        
        # ====================================================================
        # BUILD INITIAL STATE
        # ====================================================================
        
        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.array(0),
            done=jnp.array(False),
            prev_pos=mjx_data.qpos[0],  # x-position of base
        )
        
        # Get initial observation
        obs = self._get_obs(mjx_data)
        
        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        state: EnvState, 
        action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, Dict[str, Any]]:
        """
        Take one environment step.
        
        This method:
        1. Applies the action (motor torques)
        2. Runs physics simulation for action_repeat steps
        3. Computes reward based on new state
        4. Checks for termination conditions
        5. Returns new observation and reward
        
        Args:
            state: Current environment state
            action: Action to apply (12 motor torques, normalized to [-1, 1])
        
        Returns:
            obs: New observation
            new_state: Updated environment state
            reward: Reward for this transition
            info: Dictionary with additional information
        """
        # ====================================================================
        # APPLY ACTION
        # ====================================================================
        
        # Clip action to [-1, 1] range first for safety
        action = jnp.clip(action, -1.0, 1.0)
        
        # Replace any NaN values with zeros
        action = jnp.where(jnp.isnan(action), 0.0, action)
        
        # Scale actions from [-1, 1] to control range [-1, 1]
        # Note: Actuators in XML have gear=10 and ctrlrange="-10 10"
        # So ctrl values should be in [-10, 10], but the gear already amplifies
        # We use action directly (scaled to ctrl range) - the gear does the rest
        scaled_action = action * 1.0  # ctrl in [-1, 1], gear=10 gives torque [-10, 10]
        
        # Set control inputs
        mjx_data = state.mjx_data.replace(ctrl=scaled_action)
        
        # ====================================================================
        # RUN PHYSICS SIMULATION
        # ====================================================================
        
        # Run multiple simulation steps per action for stability
        def step_fn(mjx_data, _):
            return mjx.step(self.mjx_model, mjx_data), None
        
        mjx_data, _ = jax.lax.scan(
            step_fn, 
            mjx_data, 
            None, 
            length=self.action_repeat
        )
        
        # ====================================================================
        # COMPUTE REWARD
        # ====================================================================
        
        reward = self._compute_reward(
            mjx_data, 
            state.prev_pos, 
            action
        )
        
        # ====================================================================
        # CHECK TERMINATION
        # ====================================================================
        
        done = self._check_termination(mjx_data)
        
        # Also terminate if episode length exceeded
        new_step_count = state.step_count + 1
        done = done | (new_step_count >= self.episode_length)
        
        # ====================================================================
        # BUILD NEW STATE
        # ====================================================================
        
        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=new_step_count,
            done=done,
            prev_pos=mjx_data.qpos[0],  # Current x-position
        )
        
        # Get new observation
        obs = self._get_obs(mjx_data)
        
        # Build info dict
        info = {
            'step': new_step_count,
            'x_pos': mjx_data.qpos[0],
            'z_pos': mjx_data.qpos[2],
        }
        
        return obs, new_state, reward, info
    
    # ========================================================================
    # OBSERVATION COMPUTATION
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """
        Extract observation from MJX data.
        
        The observation includes:
        - Joint positions (12 values): Positions of all revolute joints
        - Joint velocities (12 values): Velocities of all revolute joints
        - Body orientation (4 values): Quaternion of the base link
        - Body angular velocity (3 values): Angular velocity of base
        
        Total: 31 values
        
        Args:
            mjx_data: MJX simulation data
        
        Returns:
            obs: Observation array of shape (31,)
        """
        # ====================================================================
        # EXTRACT JOINT POSITIONS
        # ====================================================================
        
        # qpos layout: [base_x, base_y, base_z, quat_w, quat_x, quat_y, quat_z, joint1, joint2, ...]
        # Skip first 7 (floating base) to get joint positions
        joint_positions = mjx_data.qpos[7:]  # Shape: (12,)
        
        # ====================================================================
        # EXTRACT JOINT VELOCITIES
        # ====================================================================
        
        # qvel layout: [base_vx, base_vy, base_vz, base_wx, base_wy, base_wz, joint1_vel, ...]
        # Skip first 6 (floating base velocity) to get joint velocities
        joint_velocities = mjx_data.qvel[6:]  # Shape: (12,)
        
        # ====================================================================
        # EXTRACT BODY ORIENTATION
        # ====================================================================
        
        # Quaternion of base (indices 3-6 in qpos)
        body_quaternion = mjx_data.qpos[3:7]  # Shape: (4,)
        
        # ====================================================================
        # EXTRACT BODY ANGULAR VELOCITY
        # ====================================================================
        
        # Angular velocity of base (indices 3-5 in qvel)
        body_angular_vel = mjx_data.qvel[3:6]  # Shape: (3,)
        
        # ====================================================================
        # CONCATENATE ALL OBSERVATIONS
        # ====================================================================
        
        obs = jnp.concatenate([
            joint_positions,     # 12 values
            joint_velocities,    # 12 values
            body_quaternion,     # 4 values
            body_angular_vel,    # 3 values
        ])
        
        # Safety: replace any NaN/Inf values with zeros
        obs = jnp.where(jnp.isnan(obs), 0.0, obs)
        obs = jnp.where(jnp.isinf(obs), 0.0, obs)
        
        # Clip velocities to reasonable range to prevent explosion
        obs = jnp.clip(obs, -100.0, 100.0)
        
        return obs
    
    # ========================================================================
    # REWARD COMPUTATION
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self, 
        mjx_data: mjx.Data, 
        prev_x: jnp.ndarray,
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the reward for the current state.
        
        Reward components:
        1. Forward velocity reward: Encourages walking forward
        2. Alive bonus: Small positive reward for not falling
        3. sh_roll penalty: Negative reward for sh_roll outside [-1.25, 1.25]
        4. sh_yaw penalty: Negative reward for sh_yaw > 1.55
        5. Symmetry penalty: |sh_roll_1 + sh_roll_3| and |sh_roll_2 + sh_roll_4|
        6. Energy penalty: Penalizes large control inputs
        
        Args:
            mjx_data: Current MJX simulation data
            prev_x: Previous x-position of robot base
            action: Action (control inputs) applied
        
        Returns:
            reward: Total reward value
        """
        # ====================================================================
        # 1. FORWARD VELOCITY REWARD
        # ====================================================================
        
        # Calculate forward velocity (x-direction movement)
        current_x = mjx_data.qpos[0]
        forward_velocity = (current_x - prev_x) / (self.dt * self.action_repeat)
        forward_reward = self.forward_reward_weight * forward_velocity
        
        # ====================================================================
        # 2. ALIVE BONUS
        # ====================================================================
        
        # Simple positive reward for not terminating
        alive_reward = self.alive_bonus
        
        # ====================================================================
        # 3. SH_ROLL BOUNDARY PENALTY
        # ====================================================================
        
        # Get sh_roll joint positions
        sh_roll_positions = mjx_data.qpos[self.sh_roll_indices]
        
        # Penalty for positions outside [-1.25, 1.25]
        # penalty = sum of (|pos| - 1.25) for all pos outside bounds
        sh_roll_violation_low = jnp.maximum(0.0, -1.25 - sh_roll_positions)
        sh_roll_violation_high = jnp.maximum(0.0, sh_roll_positions - 1.25)
        sh_roll_penalty = -self.sh_roll_penalty_weight * jnp.sum(
            sh_roll_violation_low + sh_roll_violation_high
        )
        
        # ====================================================================
        # 4. SH_YAW BOUNDARY PENALTY
        # ====================================================================
        
        # Get sh_yaw joint positions
        sh_yaw_positions = mjx_data.qpos[self.sh_yaw_indices]
        
        # Penalty for positions > 1.55
        sh_yaw_violation = jnp.maximum(0.0, sh_yaw_positions - 1.55)
        sh_yaw_penalty = -self.sh_yaw_penalty_weight * jnp.sum(sh_yaw_violation)
        
        # ====================================================================
        # 5. SYMMETRY PENALTY
        # ====================================================================
        
        # Leg pairs should move symmetrically
        # sh_roll_1 + sh_roll_3 should be close to 0 (opposite legs)
        # sh_roll_2 + sh_roll_4 should be close to 0 (opposite legs)
        
        # Assuming sh_roll_indices order: [sh_roll_1, sh_roll_2, sh_roll_3, sh_roll_4]
        symmetry_error_1_3 = jnp.abs(sh_roll_positions[0] + sh_roll_positions[2])
        symmetry_error_2_4 = jnp.abs(sh_roll_positions[1] + sh_roll_positions[3])
        
        # Variable penalty: larger error = larger penalty
        symmetry_penalty = -self.symmetry_penalty_weight * (
            symmetry_error_1_3 + symmetry_error_2_4
        )
        
        # ====================================================================
        # 6. ENERGY PENALTY
        # ====================================================================
        
        # Penalize large control inputs for smooth movements
        energy_penalty = -self.energy_penalty_weight * jnp.sum(jnp.square(action))
        
        # ====================================================================
        # TOTAL REWARD
        # ====================================================================
        
        total_reward = (
            forward_reward +
            alive_reward +
            sh_roll_penalty +
            sh_yaw_penalty +
            symmetry_penalty +
            energy_penalty
        )
        
        return total_reward
    
    # ========================================================================
    # TERMINATION CHECKING
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """
        Check if the episode should terminate.
        
        Termination conditions:
        1. Robot body is tilted too much (z-axis makes <75° angle with vertical)
        
        Args:
            mjx_data: Current MJX simulation data
        
        Returns:
            done: Boolean indicating termination
        """
        # ====================================================================
        # CHECK ORIENTATION (TILT ANGLE)
        # ====================================================================
        
        # Get quaternion [w, x, y, z] from qpos
        quat = mjx_data.qpos[3:7]
        
        # Calculate the z-component of the body's up vector after rotation
        # For a quaternion q = [w, x, y, z], the rotated z-axis is:
        # z_up = 1 - 2*(x^2 + y^2)  (this is the z-component of rotated [0,0,1])
        # This equals cos(tilt_angle) where tilt_angle is from vertical
        
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # z-component of the body's up vector in world frame
        # This is the dot product of body z-axis with world z-axis
        z_up = 1.0 - 2.0 * (x*x + y*y)
        
        # z_up = cos(tilt_angle_from_vertical)
        # We want to terminate if tilt > 15° (i.e., angle from vertical > 15°)
        # which means angle from horizontal (xy plane) < 75°
        # cos(15°) ≈ 0.966, so terminate if z_up < 0.966
        # But user said 75° from xy plane, so angle from vertical is 15°
        # Let's be more lenient: cos(25°) ≈ 0.906
        
        # Terminate if tilted more than 25° from upright (75° from ground would be 15°)
        # Using cos(15°) = 0.9659 for 75° from xy plane
        tilted = z_up < 0.259  # cos(75°) ≈ 0.259 - this is 75° from vertical
        
        # Actually, let's interpret correctly:
        # "z axis makes an angle of less than 75 with the xy plane"
        # means the robot's z-axis is nearly horizontal (bad)
        # When upright, z-axis is 90° from xy plane
        # So terminate if angle < 75°, meaning tilted > 15° from vertical
        # cos(15°) = 0.9659
        
        tilted = z_up < 0.9659  # Terminate if tilted more than 15° from upright
        
        return tilted
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_mj_model(self) -> mujoco.MjModel:
        """Get the original MuJoCo model (for visualization)."""
        return self.mj_model
    
    def get_mj_data(self) -> mujoco.MjData:
        """Get MuJoCo data (for visualization)."""
        return self.mj_data


# ============================================================================
# VECTORIZED ENVIRONMENT WRAPPER
# ============================================================================

class VecSpiderEnv:
    """
    Vectorized environment wrapper for parallel simulation.
    
    This wrapper allows running multiple environment instances in parallel
    using JAX's vmap, which is essential for efficient PPO training.
    """
    
    def __init__(self, env: SpiderEnv, num_envs: int):
        """
        Initialize vectorized environment.
        
        Args:
            env: Base SpiderEnv instance
            num_envs: Number of parallel environments
        """
        self.env = env
        self.num_envs = num_envs
        
        # Vectorized reset and step functions
        self._reset = jax.vmap(env.reset)
        self._step = jax.vmap(env.step)
    
    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, EnvState]:
        """
        Reset all environments.
        
        Args:
            rng: JAX random key
        
        Returns:
            obs: Observations from all envs, shape (num_envs, obs_size)
            states: States for all envs
        """
        # Split random key for each environment
        keys = random.split(rng, self.num_envs)
        return self._reset(keys)
    
    def step(
        self, 
        states: EnvState, 
        actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, Dict]:
        """
        Step all environments.
        
        Args:
            states: States for all envs
            actions: Actions for all envs, shape (num_envs, action_size)
        
        Returns:
            obs: New observations, shape (num_envs, obs_size)
            new_states: New states
            rewards: Rewards, shape (num_envs,)
            infos: Info dicts
        """
        return self._step(states, actions)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    """
    Test the environment to verify it works correctly.
    """
    print("=" * 60)
    print("Testing SpiderEnv")
    print("=" * 60)
    
    # Create environment
    env = SpiderEnv()
    
    # Initialize random key
    rng = random.PRNGKey(0)
    
    # Test reset
    print("\nTesting reset...")
    rng, key = random.split(rng)
    obs, state = env.reset(key)
    print(f"Initial observation shape: {obs.shape}")
    
    # Test step with random actions
    print("\nTesting step with random actions...")
    for i in range(5):
        rng, key = random.split(rng)
        action = random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
        obs, state, reward, info = env.step(state, action)
        print(f"Step {i+1}: reward = {float(reward):.4f}")
    
    print("\n" + "=" * 60)
    print("Environment test completed successfully!")
    print("=" * 60)
