# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class progress_to_goal(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        
        super().__init__(cfg, env)
        
        # Get command name from params
        self._command_name = cfg.params.get("command_name", "goal_pose")
        
        # Initialize previous distance buffer for all environments
        self._prev_distance = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_distance.zero_()
            self._initialized.zero_()
        else:
            self._prev_distance[env_ids] = 0.0
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "goal_pose",
    ) -> torch.Tensor:
        # Get robot position (world frame, xy only)
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # [N, 2]
        
        # Get goal position from command
        command = env.command_manager.get_command(self._command_name)
        goal_pos = command[:, :2]  # [N, 2]
        
        # Calculate current distance
        current_distance = torch.norm(goal_pos - robot_pos, dim=-1)
        
        # Calculate progress (positive = getting closer)
        # For first step, return 0 (no previous distance to compare)
        progress = torch.where(
            self._initialized,
            self._prev_distance - current_distance,
            torch.zeros_like(current_distance)
        )
        
        # Update state for next step
        self._prev_distance = current_distance.clone()
        self._initialized[:] = True
        
        return progress

class potential_field_progress(ManagerTermBase):
    """
    Potential field-based progress reward.
    
    Uses attractive potential: U = 0.5 * mu * distance²
    Reward = U_prev - U_curr (decrease in potential = positive reward)
    
    This naturally scales reward with distance (larger steps when far, smaller when close).
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        self._command_name = cfg.params.get("command_name", "goal_pose")
        self._mu = cfg.params.get("mu", 1.0)
        
        self._prev_potential = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_potential.zero_()
            self._initialized.zero_()
        else:
            self._prev_potential[env_ids] = 0.0
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "goal_pose",
        mu: float = 1.0,
    ) -> torch.Tensor:
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
        command = env.command_manager.get_command(self._command_name)
        goal_pos = command[:, :2]
        
        current_distance = torch.norm(goal_pos - robot_pos, dim=-1)
        current_potential = 0.5 * self._mu * current_distance ** 2
        
        # Reward is decrease in potential (closer = lower potential = positive reward)
        reward = torch.where(
            self._initialized,
            self._prev_potential - current_potential,
            torch.zeros_like(current_potential)
        )
        
        self._prev_potential = current_potential.clone()
        self._initialized[:] = True
        
        return reward

def goal_reached_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
    threshold: float = 0.6,
) -> torch.Tensor:
    
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    return (distance < threshold).float()

def heading_alignment_to_goal(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
) -> torch.Tensor:
    """Reward for facing toward the goal.
    
    This is STATELESS - computes from current state only.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    robot_quat = env.scene["robot"].data.root_quat_w
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    # Direction to goal
    to_goal = goal_pos - robot_pos
    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True).clamp(min=1e-6)
    to_goal_dir = to_goal / to_goal_norm
    
    # Robot heading from quaternion
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    heading_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    
    # Dot product = cos(angle)
    alignment = torch.sum(heading_dir * to_goal_dir, dim=-1)
    return alignment

def distance_to_goal_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
) -> torch.Tensor:
    """Penalty based on current distance to goal.
    
    This is STATELESS - just returns current distance.
    NOTE: Consider using progress_to_goal (stateful) instead for better learning.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    return torch.norm(goal_pos - robot_pos, dim=-1)

def collision_penalty(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    sudden_decel_threshold: float = 0.25,
    displacement_threshold: float = 1.0,
    angular_velocity_threshold: float = 0.15,
    goal_proximity_threshold: float = 1.0,
    startup_grace_steps: int = 45,
    chassis_sensor_cfg = None,
    roof_sensor_cfg = None,
) -> torch.Tensor:
    """Collision penalty using same detection logic as termination.
    
    Returns 1.0 when collision detected, 0.0 otherwise.
    Uses identical multi-layer detection as is_collided() termination function.
    
    Thresholds based on 200-episode statistical analysis:
    - sudden_decel_threshold: 0.25 m/s (83% detection, 0.053% false positive)
    - displacement_threshold: 1.0 m over 2 seconds
    - angular_velocity_threshold: 0.15 rad/s
    - goal_proximity_threshold: 1.0 m
    
    Args:
        env: The environment instance.
        force_threshold: Contact force threshold (N).
        sudden_decel_threshold: Velocity drop for collision detection (m/s).
        displacement_threshold: Minimum displacement over 2s (m).
        angular_velocity_threshold: Minimum angular velocity for turning (rad/s).
        goal_proximity_threshold: Distance to goal for proximity zone (m).
        startup_grace_steps: Steps before checking collision (3s at 15Hz).
        chassis_sensor_cfg: Chassis contact sensor config.
        roof_sensor_cfg: Roof contact sensor config.
    
    Returns:
        Penalty value (1.0 for collision, 0.0 for no collision).
    """
    
    # ========== STEP 1: Startup Grace Period ==========
    startup_grace = env.episode_length_buf < startup_grace_steps
    
    # ========== STEP 2: Sudden Deceleration Detection ==========
    if not hasattr(env, "_prev_velocity_reward"):
        env._prev_velocity_reward = torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_w
    current_speed = torch.norm(lin_vel[:, :2], dim=-1)

    velocity_change = env._prev_velocity_reward - current_speed

    # Get current action to check if robot was commanding forward motion
    # Action shape: (num_envs, 2) where [0]=left_wheel, [1]=right_wheel
    action = env.action_manager.action
    both_wheels_positive = (action[:, 0] > 0) & (action[:, 1] > 0)

    # Sudden deceleration indicates collision ONLY if robot was commanding forward
    sudden_collision = (velocity_change > sudden_decel_threshold) & both_wheels_positive

    env._prev_velocity_reward = current_speed.clone()
    
    # ========== STEP 3: Contact Force Detection ==========
    chassis_sensor = env.scene[chassis_sensor_cfg.name]
    chassis_force = chassis_sensor.data.force_matrix_w
    if chassis_force is not None:
        chassis_mag = torch.norm(chassis_force, dim=-1)
        max_chassis = torch.max(chassis_mag.view(env.num_envs, -1), dim=-1)[0]
        chassis_collision = max_chassis > force_threshold
    else:
        chassis_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    roof_sensor = env.scene[roof_sensor_cfg.name]
    roof_force = roof_sensor.data.force_matrix_w
    if roof_force is not None:
        roof_mag = torch.norm(roof_force, dim=-1)
        max_roof = torch.max(roof_mag.view(env.num_envs, -1), dim=-1)[0]
        roof_collision = max_roof > force_threshold
    else:
        roof_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    force_collision = chassis_collision | roof_collision
    
    # ========== STEP 4: Goal Proximity Check ==========
    robot_pos = robot.data.root_pos_w[:, :2]
    command = env.command_manager.get_command("goal_pose")
    goal_pos = command[:, :2]
    distance_to_goal = torch.norm(goal_pos - robot_pos, dim=-1)
    
    near_goal = distance_to_goal < goal_proximity_threshold
    
    # ========== STEP 5: Progress Tracking ==========
    if not hasattr(env, "_distance_history_reward"):
        env._distance_history_reward = torch.zeros(env.num_envs, 30, device=env.device)
        env._history_index_reward = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    env._distance_history_reward[torch.arange(env.num_envs), env._history_index_reward] = distance_to_goal
    
    distance_2s_ago_index = (env._history_index_reward - 29) % 30
    distance_2s_ago = env._distance_history_reward[torch.arange(env.num_envs), distance_2s_ago_index]
    
    env._history_index_reward = (env._history_index_reward + 1) % 30
    
    distance_change_2s = distance_to_goal - distance_2s_ago
    making_progress = distance_change_2s < 0
    
    # ========== STEP 6: Net Displacement Check ==========
    if not hasattr(env, "_position_history_reward"):
        env._position_history_reward = torch.zeros(env.num_envs, 30, 2, device=env.device)
        env._position_index_reward = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    current_pos = robot_pos.clone()
    env._position_history_reward[torch.arange(env.num_envs), env._position_index_reward] = current_pos
    
    position_2s_ago_index = (env._position_index_reward - 29) % 30
    position_2s_ago = env._position_history_reward[torch.arange(env.num_envs), position_2s_ago_index]
    
    env._position_index_reward = (env._position_index_reward + 1) % 30
    
    net_displacement = torch.norm(current_pos - position_2s_ago, dim=-1)
    is_moving = net_displacement >= displacement_threshold
    
    # ========== STEP 7: Turning Check ==========
    ang_vel = robot.data.root_ang_vel_w
    yaw_rate = torch.abs(ang_vel[:, 2])
    is_turning = yaw_rate > angular_velocity_threshold
    
    # ========== STEP 8: Combine All Checks ==========
    immediate_collision = sudden_collision | force_collision
    
    stuck_collision = (
        ~making_progress &
        ~is_moving &
        ~is_turning &
        ~near_goal
    )
    
    collision_detected = (immediate_collision | stuck_collision) & ~startup_grace
    
    return collision_detected.float()