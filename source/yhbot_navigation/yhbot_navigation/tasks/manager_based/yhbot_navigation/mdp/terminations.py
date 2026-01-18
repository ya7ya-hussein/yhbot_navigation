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
    threshold: float = 0.6,
    
) -> torch.Tensor:
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    return distance < threshold



def is_collided(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    startup_grace_steps: int = 45,
    chassis_sensor_cfg: "SceneEntityCfg" = None,
    roof_sensor_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    startup_grace = env.episode_length_buf < startup_grace_steps
    
    chassis_sensor = env.scene[chassis_sensor_cfg.name]
    chassis_force = chassis_sensor.data.net_forces_w

    if chassis_force is not None:
        # Split into horizontal (XY) and vertical (Z) components
        chassis_horizontal = torch.norm(chassis_force[:, :, :2], dim=-1)  
        chassis_vertical = torch.abs(chassis_force[:, :, 2])  
        
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
    
    
    collision_detected = force_collision & ~startup_grace
    
    return collision_detected