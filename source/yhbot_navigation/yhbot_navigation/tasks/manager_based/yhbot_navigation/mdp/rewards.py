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
        # Get robot position 
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2] 
        
        # Get goal position from command
        command = env.command_manager.get_command(self._command_name)
        goal_pos = command[:, :2] 
        
        # Calculate current distance
        current_distance = torch.norm(goal_pos - robot_pos, dim=-1)
        
        # Calculate progress 
        # For first step, return 0 
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
        
        # Reward is decrease in potential 
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
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    return torch.norm(goal_pos - robot_pos, dim=-1)

def collision_penalty(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    startup_grace_steps: int = 45,
    chassis_sensor_cfg = None,
    roof_sensor_cfg = None,
) -> torch.Tensor:
    startup_grace = env.episode_length_buf < startup_grace_steps
    
    chassis_sensor = env.scene[chassis_sensor_cfg.name]
    chassis_force = chassis_sensor.data.net_forces_w
    if chassis_force is not None:

        chassis_horizontal = torch.norm(chassis_force[:, :, :2], dim=-1) 

        chassis_collision = torch.any(chassis_horizontal > force_threshold, dim=-1)
    else:
        chassis_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check roof contact
    roof_sensor = env.scene[roof_sensor_cfg.name]
    roof_force = roof_sensor.data.net_forces_w
    if roof_force is not None:
        roof_mag = torch.norm(roof_force, dim=-1)
        max_roof = torch.max(roof_mag.view(env.num_envs, -1), dim=-1)[0]
        roof_collision = max_roof > force_threshold
    else:
        roof_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Combine contact detections
    force_collision = chassis_collision | roof_collision
    
    # Apply startup grace period
    collision_detected = force_collision & ~startup_grace
    
    return collision_detected.float()