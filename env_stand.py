"""
============================================================================
Emotive-Spider Standing Training Environment for MuJoCo MJX
============================================================================
This module defines the standing training environment for the 4-legged spider robot.
It uses MuJoCo MJX for JAX-based accelerated physics simulation.

The environment teaches the robot to STAND at a target height (10cm) while
keeping its base perpendicular to the ground.

Key Components:
- SpiderStandEnv: Main environment class
- Observation: Joint positions, velocities, body orientation
- Actions: 12 motor torques (one per actuator)
- Rewards: Standing-focused reward function

Reward Function:
1. Height reward: Reward for keeping base at 10cm from ground
2. Orientation reward: Reward for keeping base perpendicular to ground
3. All penalties from env.py maintained
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
        prev_pos: Previous x,y-position of the robot (for velocity tracking)
        prev_qpos: Previous joint positions (for smoothness penalties)
        prev_qvel: Previous joint velocities (for jerk penalty)
    """
    mjx_data: mjx.Data
    step_count: jnp.ndarray
    done: jnp.ndarray
    prev_pos: jnp.ndarray  # Now stores [x, y] position
    prev_qpos: jnp.ndarray  # Previous joint positions
    prev_qvel: jnp.ndarray  # Previous joint velocities


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

class SpiderStandEnv:
    """
    MJX-based training environment for the Emotive-Spider robot (STANDING).
    
    This environment teaches the spider robot to stand at a target height
    while maintaining a perpendicular orientation to the ground.
    
    Joint Configuration (12 total):
    - Leg 1: sh_roll_1, sh_yaw_1, kn_yaw_1
    - Leg 2: sh_roll_2, sh_yaw_2, kn_yaw_2
    - Leg 3: sh_roll_3, sh_yaw_3, kn_yaw_3
    - Leg 4: sh_roll_4, sh_yaw_4, kn_yaw_4
    
    Reward Function Components:
    1. Height reward (positive): Reward for being close to 10cm target height
    2. Orientation reward (positive): Reward for keeping base perpendicular to ground
    3. Alive bonus (positive): For not falling
    4. All penalties from env.py:
       - Motor effort penalty
       - Joint velocity change penalty
       - Jerk penalty
       - sh_roll boundary penalty
       - sh_roll symmetry penalty
    """
    
    def __init__(
        self,
        urdf_path: str = None,
        episode_length: int = 1000,
        dt: float = 0.002,
        action_repeat: int = 4,
        # Target height for standing
        target_height: float = 0.10,  # 10cm from ground
        # Reward weights (positive = reward, used as multipliers)
        height_reward_weight: float = 10.0,  # Main objective - reward for height
        orientation_reward_weight: float = 5.0,  # Reward for being perpendicular
        alive_bonus: float = 1.0,  # Encourage staying up
        # Penalty weights (these will be negated in reward computation)
        velocity_tracking_weight: float = 0.5,  # Penalize movement (should stay still)
        yaw_rate_tracking_weight: float = 0.5,
        base_orientation_weight: float = 2.0,  # Keep body level (additional)
        motor_effort_weight: float = 0.01,  # Small energy penalty
        joint_velocity_change_weight: float = 0.1,  # Smooth movements
        jerk_weight: float = 0.05,
        # sh_roll constraints
        sh_roll_boundary_weight: float = 5.0,  # Penalty for sh_roll outside [-1.26, 1.26]
        sh_roll_symmetry_weight: float = 2.0,  # Penalty for asymmetric sh_roll
        # Angle rewards
        kn_yaw_angle_reward_weight: float = 0.3,
        sh_yaw_angle_reward_weight: float = 0.3,
        # Death penalty
        death_penalty: float = 10.0,
    ):
        """
        Initialize the Spider standing training environment.
        
        Args:
            urdf_path: Path to the robot URDF file. If None, uses default path.
            episode_length: Maximum steps per episode.
            dt: Simulation timestep (seconds).
            action_repeat: Number of simulation steps per action.
            target_height: Target height for the base (meters).
            height_reward_weight: Weight for height reward (closer to 10cm = higher).
            orientation_reward_weight: Weight for perpendicular orientation reward.
            alive_bonus: Bonus reward for staying upright.
            velocity_tracking_weight: Penalty weight for velocity deviation.
            yaw_rate_tracking_weight: Penalty weight for yaw rate deviation.
            base_orientation_weight: Penalty weight for non-flat base.
            motor_effort_weight: Penalty weight for high motor effort.
            joint_velocity_change_weight: Penalty for rapid joint position changes.
            jerk_weight: Penalty for jerky movements (acceleration changes).
            kn_yaw_angle_reward_weight: Reward for kn_yaw < 0.7 rad.
            sh_yaw_angle_reward_weight: Reward for sh_yaw < 1.0 rad.
            death_penalty: Penalty applied on termination.
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
        
        # Target height for standing
        self.target_height = target_height
        
        # Reward weights
        self.height_reward_weight = height_reward_weight
        self.orientation_reward_weight = orientation_reward_weight
        self.alive_bonus = alive_bonus
        
        # Penalty weights
        self.velocity_tracking_weight = velocity_tracking_weight
        self.yaw_rate_tracking_weight = yaw_rate_tracking_weight
        self.base_orientation_weight = base_orientation_weight
        self.motor_effort_weight = motor_effort_weight
        self.joint_velocity_change_weight = joint_velocity_change_weight
        self.jerk_weight = jerk_weight
        
        # sh_roll constraints
        self.sh_roll_boundary_weight = sh_roll_boundary_weight
        self.sh_roll_symmetry_weight = sh_roll_symmetry_weight
        
        # Angle rewards
        self.kn_yaw_angle_reward_weight = kn_yaw_angle_reward_weight
        self.sh_yaw_angle_reward_weight = sh_yaw_angle_reward_weight
        
        # Death penalty
        self.death_penalty = death_penalty
        
        # ====================================================================
        # ACTION AND OBSERVATION SPACES
        # ====================================================================
        
        # 12 actuators (motors)
        self.action_size = self.mj_model.nu
        print(f"Action size: {self.action_size}")
        
        # Observation: joint positions (12) + joint velocities (12) + 
        # body orientation quaternion (4) + body angular velocity (3) +
        # base linear velocity (3) + previous actions (12) + height (1)
        # Total: 12 + 12 + 4 + 3 + 3 + 12 + 1 = 47
        self.obs_size = 47
        print(f"Observation size: {self.obs_size}")
        
        # ====================================================================
        # JOINT AND BODY INDICES FOR REWARD COMPUTATION
        # ====================================================================
        
        self._setup_joint_indices()
        self._setup_body_indices()
        
        print(f"Standing environment initialized with target height: {self.target_height}m")
    
    def _setup_joint_indices(self):
        """
        Setup joint indices for reward computation.
        
        This method identifies which qpos indices correspond to:
        - sh_roll_1, sh_roll_2, sh_roll_3, sh_roll_4
        - sh_yaw_1, sh_yaw_2, sh_yaw_3, sh_yaw_4
        - kn_yaw_1, kn_yaw_2, kn_yaw_3, kn_yaw_4
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
        # IDENTIFY JOINT INDICES
        # ====================================================================
        
        self.sh_roll_indices = []
        self.sh_yaw_indices = []
        self.kn_yaw_indices = []
        
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
            elif 'kn_yaw' in name:
                self.kn_yaw_indices.append(qpos_idx)
                print(f"  kn_yaw joint '{name}' at qpos index {qpos_idx}")
        
        # Convert to JAX arrays for efficient computation
        self.sh_roll_indices = jnp.array(self.sh_roll_indices, dtype=jnp.int32)
        self.sh_yaw_indices = jnp.array(self.sh_yaw_indices, dtype=jnp.int32)
        self.kn_yaw_indices = jnp.array(self.kn_yaw_indices, dtype=jnp.int32)
        
        print(f"sh_roll indices: {self.sh_roll_indices}")
        print(f"sh_yaw indices: {self.sh_yaw_indices}")
        print(f"kn_yaw indices: {self.kn_yaw_indices}")
    
    def _setup_body_indices(self):
        """
        Setup body indices for contact checking.
        
        Identifies which bodies are allowed to touch ground (feet):
        - part_1_4 (leg 1 foot)
        - part_1_7 (leg 2 foot)
        - part_1_10 (leg 3 foot)
        - part_1_13 (leg 4 foot)
        """
        # ====================================================================
        # IDENTIFY FOOT BODY INDICES
        # ====================================================================
        
        allowed_contact_bodies = ['part_1_4', 'part_1_7', 'part_1_10', 'part_1_13']
        self.foot_body_ids = []
        
        print("Setting up body indices for contact checking...")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name in allowed_contact_bodies:
                self.foot_body_ids.append(i)
                print(f"  Allowed contact body '{name}' has id {i}")
        
        self.foot_body_ids = jnp.array(self.foot_body_ids, dtype=jnp.int32)
        
        # Get floor geom id
        self.floor_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        print(f"Floor geom id: {self.floor_geom_id}")
        
        # Get all geom ids that belong to foot bodies (allowed to contact floor)
        self.allowed_contact_geom_ids = []
        for body_id in self.foot_body_ids:
            # Find geoms belonging to this body
            for geom_id in range(self.mj_model.ngeom):
                if self.mj_model.geom_bodyid[geom_id] == body_id:
                    self.allowed_contact_geom_ids.append(geom_id)
                    geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    print(f"  Allowed contact geom '{geom_name}' (id {geom_id}) on body {body_id}")
        
        self.allowed_contact_geom_ids = jnp.array(self.allowed_contact_geom_ids, dtype=jnp.int32)
        print(f"Allowed contact geom ids: {self.allowed_contact_geom_ids}")
    
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
        
        # Set initial height (z-position) to be near target
        new_qpos = new_qpos.at[2].set(0.15)  # Start at 15cm height
        
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
        
        # Initialize previous positions and velocities
        prev_pos = jnp.array([mjx_data.qpos[0], mjx_data.qpos[1]])  # [x, y]
        prev_qpos = mjx_data.qpos[7:]  # Joint positions
        prev_qvel = mjx_data.qvel[6:]  # Joint velocities
        
        state = EnvState(
            mjx_data=mjx_data,
            step_count=jnp.array(0),
            done=jnp.array(False),
            prev_pos=prev_pos,
            prev_qpos=prev_qpos,
            prev_qvel=prev_qvel,
        )
        
        # Get initial observation (with zero previous actions)
        zero_actions = jnp.zeros(self.action_size)
        obs = self._get_obs(mjx_data, zero_actions)
        
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
        
        # Scale actions for control
        scaled_action = action * 1.0
        
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
            state.prev_qpos,
            state.prev_qvel,
            action
        )
        
        # ====================================================================
        # CHECK TERMINATION
        # ====================================================================
        
        tilted = self._check_termination(mjx_data)
        
        # Also terminate if episode length exceeded
        new_step_count = state.step_count + 1
        timeout = new_step_count >= self.episode_length
        done = tilted | timeout
        
        # Apply death penalty if terminated due to tilt (not timeout)
        death_penalty = -self.death_penalty * tilted.astype(jnp.float32)
        reward = reward + death_penalty
        
        # ====================================================================
        # BUILD NEW STATE
        # ====================================================================
        
        new_prev_pos = jnp.array([mjx_data.qpos[0], mjx_data.qpos[1]])
        new_prev_qpos = mjx_data.qpos[7:]
        new_prev_qvel = mjx_data.qvel[6:]
        
        new_state = EnvState(
            mjx_data=mjx_data,
            step_count=new_step_count,
            done=done,
            prev_pos=new_prev_pos,
            prev_qpos=new_prev_qpos,
            prev_qvel=new_prev_qvel,
        )
        
        # Get new observation
        obs = self._get_obs(mjx_data, action)
        
        # Build info dict
        info = {
            'step': new_step_count,
            'x_pos': mjx_data.qpos[0],
            'z_pos': mjx_data.qpos[2],
            'height_from_target': jnp.abs(mjx_data.qpos[2] - self.target_height),
        }
        
        return obs, new_state, reward, info
    
    # ========================================================================
    # OBSERVATION COMPUTATION
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, mjx_data: mjx.Data, prev_action: jnp.ndarray) -> jnp.ndarray:
        """
        Extract observation from MJX data.
        
        The observation includes:
        - Joint positions (12 values): Positions of all revolute joints
        - Joint velocities (12 values): Velocities of all revolute joints
        - Body orientation (4 values): Quaternion of the base link
        - Body angular velocity (3 values): Angular velocity of base
        - Body linear velocity (3 values): Linear velocity of base
        - Previous actions (12 values): Actions from last step
        - Current height (1 value): Height of base from ground
        
        Total: 47 values
        
        Args:
            mjx_data: MJX simulation data
            prev_action: Previous action applied
        
        Returns:
            obs: Observation array of shape (47,)
        """
        # ====================================================================
        # EXTRACT JOINT POSITIONS
        # ====================================================================
        
        # qpos layout: [base_x, base_y, base_z, quat_w, quat_x, quat_y, quat_z, joint1, joint2, ...]
        joint_positions = mjx_data.qpos[7:]  # Shape: (12,)
        
        # ====================================================================
        # EXTRACT JOINT VELOCITIES
        # ====================================================================
        
        # qvel layout: [base_vx, base_vy, base_vz, base_wx, base_wy, base_wz, joint1_vel, ...]
        joint_velocities = mjx_data.qvel[6:]  # Shape: (12,)
        
        # ====================================================================
        # EXTRACT BODY ORIENTATION
        # ====================================================================
        
        body_quaternion = mjx_data.qpos[3:7]  # Shape: (4,)
        
        # ====================================================================
        # EXTRACT BODY VELOCITIES
        # ====================================================================
        
        body_angular_vel = mjx_data.qvel[3:6]  # Shape: (3,)
        body_linear_vel = mjx_data.qvel[0:3]   # Shape: (3,)
        
        # ====================================================================
        # EXTRACT CURRENT HEIGHT
        # ====================================================================
        
        current_height = mjx_data.qpos[2:3]  # Shape: (1,)
        
        # ====================================================================
        # CONCATENATE ALL OBSERVATIONS
        # ====================================================================
        
        obs = jnp.concatenate([
            joint_positions,     # 12 values
            joint_velocities,    # 12 values
            body_quaternion,     # 4 values
            body_angular_vel,    # 3 values
            body_linear_vel,     # 3 values
            prev_action,         # 12 values
            current_height,      # 1 value
        ])
        
        # Safety: replace any NaN/Inf values with zeros
        obs = jnp.where(jnp.isnan(obs), 0.0, obs)
        obs = jnp.where(jnp.isinf(obs), 0.0, obs)
        
        # Clip to reasonable range to prevent explosion
        obs = jnp.clip(obs, -100.0, 100.0)
        
        return obs
    
    # ========================================================================
    # REWARD COMPUTATION
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self, 
        mjx_data: mjx.Data, 
        prev_pos: jnp.ndarray,
        prev_qpos: jnp.ndarray,
        prev_qvel: jnp.ndarray,
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the reward for the current state (STANDING task).
        
        Reward components:
        1. Height reward: Reward for being close to 10cm target height
        2. Orientation reward: Reward for keeping base perpendicular to ground
        3. Alive bonus: Small positive reward for not falling
        4. Velocity penalty: Penalizes movement (should stay still)
        5. Yaw rate penalty: Penalizes deviation from target yaw rate
        6. Motor effort penalty: Penalizes high motor effort
        7. Joint velocity change penalty: Penalizes rapid joint position changes
        8. Jerk penalty: Penalizes jerky movements (velocity acceleration)
        9. sh_roll boundary penalty: Penalizes sh_roll outside bounds
        10. sh_roll symmetry penalty: Penalizes asymmetric sh_roll
        11. kn_yaw angle reward: Rewards kn_yaw < 0.7 radians
        12. sh_yaw angle reward: Rewards sh_yaw < 1.0 radians
        
        Args:
            mjx_data: Current MJX simulation data
            prev_pos: Previous [x, y] position of robot base
            prev_qpos: Previous joint positions
            prev_qvel: Previous joint velocities
            action: Action (control inputs) applied
        
        Returns:
            reward: Total reward value
        """
        delta_t = self.dt * self.action_repeat
        
        # ====================================================================
        # 1. HEIGHT REWARD (primary reward for standing)
        # ====================================================================
        
        # Current height of the base
        current_height = mjx_data.qpos[2]
        
        # Compute distance from target height (10cm = 0.1m)
        height_error = jnp.abs(current_height - self.target_height)
        
        # Reward is inversely proportional to distance from target
        # Using exponential decay for smooth reward gradient
        # max reward of 1.0 when exactly at target height
        height_reward = self.height_reward_weight * jnp.exp(-10.0 * height_error)
        
        # ====================================================================
        # 2. ORIENTATION REWARD (reward for being perpendicular to ground)
        # ====================================================================
        
        # Get quaternion [w, x, y, z] from qpos
        quat = mjx_data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Calculate roll and pitch from quaternion
        # Roll (rotation around x-axis)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation around y-axis)
        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.where(
            jnp.abs(sinp) >= 1,
            jnp.sign(sinp) * jnp.pi / 2,
            jnp.arcsin(sinp)
        )
        
        # Compute total orientation deviation from perpendicular
        # Perpendicular = roll and pitch both = 0
        orientation_error = jnp.sqrt(roll**2 + pitch**2)
        
        # Reward for being close to perpendicular (exponential decay)
        orientation_reward = self.orientation_reward_weight * jnp.exp(-5.0 * orientation_error)
        
        # ====================================================================
        # 3. ALIVE BONUS
        # ====================================================================
        
        alive_reward = self.alive_bonus
        
        # ====================================================================
        # 4. VELOCITY PENALTY (should stay still for standing)
        # ====================================================================
        
        # Get current XY velocities from qvel (should be minimal for standing)
        current_vel_x = mjx_data.qvel[0]
        current_vel_y = mjx_data.qvel[1]
        
        # Penalize any horizontal movement (target velocity = 0 for standing)
        velocity_penalty = -self.velocity_tracking_weight * (
            jnp.abs(current_vel_x) + jnp.abs(current_vel_y)
        )
        
        # ====================================================================
        # 5. YAW RATE TRACKING PENALTY
        # ====================================================================
        
        # Yaw rate is angular velocity around z-axis (qvel[5])
        current_yaw_rate = mjx_data.qvel[5]
        yaw_rate_penalty = -self.yaw_rate_tracking_weight * jnp.abs(current_yaw_rate)
        
        # ====================================================================
        # 6. BASE ORIENTATION PENALTY (additional penalty for non-flat base)
        # ====================================================================
        
        # Penalty for any roll or pitch (should be flat, so both should be 0)
        orientation_error_penalty = jnp.abs(roll) + jnp.abs(pitch)
        base_orientation_penalty = -self.base_orientation_weight * orientation_error_penalty
        
        # ====================================================================
        # 7. MOTOR EFFORT PENALTY (high torque usage)
        # ====================================================================
        
        # Penalize squared control inputs (proportional to torque)
        motor_effort_penalty = -self.motor_effort_weight * jnp.sum(jnp.square(action))
        
        # ====================================================================
        # 8. JOINT VELOCITY CHANGE PENALTY (rapid changes in joint positions)
        # ====================================================================
        
        # Current joint positions
        current_qpos = mjx_data.qpos[7:]
        
        # Change in joint positions
        joint_pos_change = current_qpos - prev_qpos
        
        # Penalize rapid changes
        joint_velocity_change_penalty = -self.joint_velocity_change_weight * jnp.sum(jnp.square(joint_pos_change))
        
        # ====================================================================
        # 9. JERK PENALTY (jerky movements - change in velocity)
        # ====================================================================
        
        # Current joint velocities
        current_qvel = mjx_data.qvel[6:]
        
        # Change in joint velocities (acceleration)
        joint_vel_change = current_qvel - prev_qvel
        
        # Penalize jerky movements
        jerk_penalty = -self.jerk_weight * jnp.sum(jnp.square(joint_vel_change))
        
        # ====================================================================
        # 10. SH_ROLL BOUNDARY PENALTY (penalize outside [-1.26, 1.26])
        # ====================================================================
        
        sh_roll_positions = mjx_data.qpos[self.sh_roll_indices]
        
        # Penalty for exceeding upper bound (1.26)
        upper_violation = jnp.maximum(0.0, sh_roll_positions - 1.26)
        # Penalty for exceeding lower bound (-1.26)
        lower_violation = jnp.maximum(0.0, -1.26 - sh_roll_positions)
        
        sh_roll_boundary_penalty = -self.sh_roll_boundary_weight * jnp.sum(
            jnp.square(upper_violation) + jnp.square(lower_violation)
        )
        
        # ====================================================================
        # 11. SH_ROLL SYMMETRY PENALTY
        # ====================================================================
        
        # sh_roll pairs should be symmetric (opposite legs move together)
        # |sh_roll1 + sh_roll2| + |sh_roll3 + sh_roll4| should be minimized
        sh_roll_1 = sh_roll_positions[0]  # sh_roll_1
        sh_roll_2 = sh_roll_positions[1]  # sh_roll_2
        sh_roll_3 = sh_roll_positions[2]  # sh_roll_3
        sh_roll_4 = sh_roll_positions[3]  # sh_roll_4
        
        # Diagonal pairs should move opposite (sum should be ~0)
        symmetry_error_12 = jnp.abs(sh_roll_1 + sh_roll_2)
        symmetry_error_34 = jnp.abs(sh_roll_3 + sh_roll_4)
        
        sh_roll_symmetry_penalty = -self.sh_roll_symmetry_weight * (
            symmetry_error_12 + symmetry_error_34
        )
        
        # ====================================================================
        # 12. KN_YAW ANGLE REWARD (reward for angles < 0.7 radians)
        # ====================================================================
        
        # Get kn_yaw joint positions (absolute values)
        kn_yaw_positions = mjx_data.qpos[self.kn_yaw_indices]
        kn_yaw_abs = jnp.abs(kn_yaw_positions)
        
        # Reward when angle magnitude is less than 0.7 radians
        # Reward = sum of (0.7 - abs(angle)) for angles below threshold
        kn_yaw_below_threshold = jnp.maximum(0.0, 0.7 - kn_yaw_abs)
        kn_yaw_reward = self.kn_yaw_angle_reward_weight * jnp.sum(kn_yaw_below_threshold)
        
        # ====================================================================
        # 13. SH_YAW ANGLE REWARD (reward for angles < 1.0 radians)
        # ====================================================================
        
        # Get sh_yaw joint positions (absolute values)
        sh_yaw_positions = mjx_data.qpos[self.sh_yaw_indices]
        sh_yaw_abs = jnp.abs(sh_yaw_positions)
        
        # Reward when angle magnitude is less than 1.0 radians
        sh_yaw_below_threshold = jnp.maximum(0.0, 1.0 - sh_yaw_abs)
        sh_yaw_reward = self.sh_yaw_angle_reward_weight * jnp.sum(sh_yaw_below_threshold)
        
        # ====================================================================
        # TOTAL REWARD
        # ====================================================================
        
        total_reward = (
            height_reward +               # Primary: keep at 10cm height
            orientation_reward +          # Primary: stay perpendicular
            alive_reward +
            velocity_penalty +            # Penalize movement
            yaw_rate_penalty +
            base_orientation_penalty +
            motor_effort_penalty +
            joint_velocity_change_penalty +
            jerk_penalty +
            sh_roll_boundary_penalty +
            sh_roll_symmetry_penalty +
            kn_yaw_reward +
            sh_yaw_reward
        )
        
        return total_reward
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_illegal_contact_penalty(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """
        Compute penalty for illegal contacts (non-foot parts touching ground).
        
        Only the following parts are allowed to touch ground:
        - part_1_4 (leg 1 foot)
        - part_1_7 (leg 2 foot)
        - part_1_10 (leg 3 foot)
        - part_1_13 (leg 4 foot)
        
        This uses a height-based approach: check if non-foot body parts are
        too close to the ground (z < threshold).
        
        Args:
            mjx_data: Current MJX simulation data
        
        Returns:
            penalty: Negative penalty value (0 if no illegal contacts)
        """
        # Get body positions from xpos (world frame positions)
        # xpos has shape (nbody, 3) - [x, y, z] for each body
        body_positions = mjx_data.xpos
        
        # Ground threshold - bodies below this height are considered in contact
        ground_threshold = 0.03  # 3cm threshold
        
        # Check each body that is NOT a foot
        # Foot body ids: 4 (part_1_4), 7 (part_1_7), 10 (part_1_10), 13 (part_1_13)
        # We need to check: body (id 0-3), upper legs, mid legs
        
        # Get z-positions of all bodies
        z_positions = body_positions[:, 2]
        
        # Create mask for non-foot bodies
        # Bodies 0 is world, 1 is base (part_1)
        # Bodies 2, 3, 4 are leg 1 (upper, mid, foot) - only 4 is allowed
        # Bodies 5, 6, 7 are leg 2 (upper, mid, foot) - only 7 is allowed
        # Bodies 8, 9, 10 are leg 3 (upper, mid, foot) - only 10 is allowed  
        # Bodies 11, 12, 13 are leg 4 (upper, mid, foot) - only 13 is allowed
        
        # Check main body (part_1, id=1)
        body_too_low = (z_positions[1] < ground_threshold).astype(jnp.float32)
        
        # Check upper leg segments (ids 2, 5, 8, 11)
        upper_leg_violations = (
            (z_positions[2] < ground_threshold).astype(jnp.float32) +
            (z_positions[5] < ground_threshold).astype(jnp.float32) +
            (z_positions[8] < ground_threshold).astype(jnp.float32) +
            (z_positions[11] < ground_threshold).astype(jnp.float32)
        )
        
        # Check middle leg segments (ids 3, 6, 9, 12)
        mid_leg_violations = (
            (z_positions[3] < ground_threshold).astype(jnp.float32) +
            (z_positions[6] < ground_threshold).astype(jnp.float32) +
            (z_positions[9] < ground_threshold).astype(jnp.float32) +
            (z_positions[12] < ground_threshold).astype(jnp.float32)
        )
        
        # Total violations
        total_violations = body_too_low + upper_leg_violations + mid_leg_violations
        
        # Apply heavy penalty (using a default weight since illegal_contact_weight is not defined)
        penalty = -10.0 * total_violations
        
        return penalty
    
    # ========================================================================
    # TERMINATION CHECKING
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """
        Check if the episode should terminate.
        
        Termination conditions:
        1. Robot body is tilted too much (z-axis makes <45° angle with vertical)
        2. Robot height is too low (body touching or near ground)
        3. Non-foot body parts touching ground (illegal contact)
        
        Args:
            mjx_data: Current MJX simulation data
        
        Returns:
            done: Boolean indicating termination
        """
        # ====================================================================
        # CHECK ORIENTATION (TILT ANGLE)
        # ====================================================================
        
        quat = mjx_data.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # z-component of the body's up vector in world frame
        z_up = 1.0 - 2.0 * (x*x + y*y)
        
        # Terminate if tilted more than 45° from upright
        tilted = z_up < 0.707  # cos(45°) ≈ 0.707
        
        # ====================================================================
        # CHECK HEIGHT
        # ====================================================================
        
        body_height = mjx_data.qpos[2]
        too_low = body_height < 0.03  # Less than 3cm from ground (stricter for standing)
        
        # ====================================================================
        # CHECK ILLEGAL BODY CONTACT
        # ====================================================================
        
        # Get z-positions of all bodies
        z_positions = mjx_data.xpos[:, 2]
        ground_threshold = 0.02  # 2cm - if any non-foot body is this low, terminate
        
        # Foot bodies are 4, 7, 10, 13 - all others should NOT touch ground
        # Check main body (id=1)
        body_contact = z_positions[1] < ground_threshold
        
        # Check upper legs (ids 2, 5, 8, 11)
        upper_leg_contact = (
            (z_positions[2] < ground_threshold) |
            (z_positions[5] < ground_threshold) |
            (z_positions[8] < ground_threshold) |
            (z_positions[11] < ground_threshold)
        )
        
        # Check mid legs (ids 3, 6, 9, 12)
        mid_leg_contact = (
            (z_positions[3] < ground_threshold) |
            (z_positions[6] < ground_threshold) |
            (z_positions[9] < ground_threshold) |
            (z_positions[12] < ground_threshold)
        )
        
        illegal_contact = body_contact | upper_leg_contact | mid_leg_contact
        
        return tilted | too_low | illegal_contact
    
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

class VecSpiderStandEnv:
    """
    Vectorized environment wrapper for parallel simulation.
    
    This wrapper allows running multiple environment instances in parallel
    using JAX's vmap, which is essential for efficient PPO training.
    """
    
    def __init__(self, env: SpiderStandEnv, num_envs: int):
        """
        Initialize vectorized environment.
        
        Args:
            env: Base SpiderStandEnv instance
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
    print("Testing SpiderStandEnv (Standing Environment)")
    print("=" * 60)
    
    # Create environment
    env = SpiderStandEnv()
    
    # Initialize random key
    rng = random.PRNGKey(0)
    
    # Test reset
    print("\nTesting reset...")
    rng, key = random.split(rng)
    obs, state = env.reset(key)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Expected observation size: {env.obs_size}")
    print(f"Target height: {env.target_height}m")
    
    # Test step with random actions
    print("\nTesting step with random actions...")
    for i in range(5):
        rng, key = random.split(rng)
        action = random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
        obs, state, reward, info = env.step(state, action)
        height = float(state.mjx_data.qpos[2])
        print(f"Step {i+1}: reward = {float(reward):.4f}, height = {height:.4f}m, height_error = {float(info['height_from_target']):.4f}")
    
    print("\n" + "=" * 60)
    print("Standing environment test completed successfully!")
    print("=" * 60)
