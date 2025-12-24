# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for YHBot navigation task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_goal_reached(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
    threshold: float = 0.5,
    
) -> torch.Tensor:
    """Termination condition when goal is reached.
    
    Args:
        env: The environment instance.
        command_name: Name of the command term for goal position.
        threshold: Distance threshold for goal reached.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    return distance < threshold

def is_collided(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    velocity_threshold: float = 0.05,
    action_threshold: float = 0.3,
    stuck_time_threshold: float = 0.5,
    chassis_sensor_cfg: "SceneEntityCfg" = None,
    roof_sensor_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    # ========== PART 1: Contact Force Detection ==========
    
    # Check chassis
    chassis_sensor = env.scene[chassis_sensor_cfg.name]
    chassis_force = chassis_sensor.data.force_matrix_w
    if chassis_force is not None:
        chassis_mag = torch.norm(chassis_force, dim=-1)
        max_chassis = torch.max(chassis_mag.view(env.num_envs, -1), dim=-1)[0]
        chassis_collision = max_chassis > force_threshold
    else:
        chassis_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check roof
    roof_sensor = env.scene[roof_sensor_cfg.name]
    roof_force = roof_sensor.data.force_matrix_w
    if roof_force is not None:
        roof_mag = torch.norm(roof_force, dim=-1)
        max_roof = torch.max(roof_mag.view(env.num_envs, -1), dim=-1)[0]
        roof_collision = max_roof > force_threshold
    else:
        roof_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Combine contact force detections
    force_collision = chassis_collision | roof_collision
    
    # ========== PART 2: Stuck Detection ==========
    
    # Get robot velocity
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    
    # Get action magnitude
    action_magnitude = torch.norm(env.action_manager.action, dim=-1)
    
    # Initialize stuck counter if not exists
    if not hasattr(env, "_stuck_counter"):
        env._stuck_counter = torch.zeros(env.num_envs, device=env.device)
    
    # Reset counters for fresh episodes (first 3 steps)
    fresh_episodes = env.episode_length_buf < 3
    env._stuck_counter[fresh_episodes] = 0.0
    
    # Check stuck conditions
    is_slow = speed < velocity_threshold
    is_trying_to_move = action_magnitude > action_threshold
    is_stuck_condition = is_slow & is_trying_to_move
    
    # Update stuck counter
    env._stuck_counter[is_stuck_condition] += env.step_dt
    env._stuck_counter[~is_stuck_condition] = 0.0
    
    # Stuck if condition persists
    stuck_collision = env._stuck_counter > stuck_time_threshold
    
    # ========== PART 3: Combine ==========
    return force_collision | stuck_collision