# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # Get ray hit distances
    ray_origins = sensor.data.pos_w.unsqueeze(1)  
    ray_hits = sensor.data.ray_hits_w  
    
    # Compute distances
    distances = torch.norm(ray_hits - ray_origins, dim=-1)
    
    # Handle inf values 
    distances = torch.where(torch.isinf(distances), torch.tensor(sensor.cfg.max_distance, device=env.device), distances)
    
    # Normalize to [0, 1]
    distances = distances / sensor.cfg.max_distance
    
    return distances


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def goal_position_robot_frame(env: ManagerBasedRLEnv, command_name: str = "goal_command") -> torch.Tensor:
    # Get goal command from command manager
    goal_command = env.command_manager.get_command(command_name) 
    goal_pos_world = goal_command[:, :2]  
    
    # Get robot state
    robot: Articulation = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  
    robot_quat = robot.data.root_quat_w  
    
    # Compute relative position in world frame
    goal_rel_world = goal_pos_world - robot_pos  
    
    # Convert to robot frame using yaw rotation
    # Extract yaw from quaternion
    yaw = torch.atan2(
        2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    )
    
    # Rotation matrix 
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    # Transform to robot frame
    goal_x_robot = cos_yaw * goal_rel_world[:, 0] + sin_yaw * goal_rel_world[:, 1]
    goal_y_robot = -sin_yaw * goal_rel_world[:, 0] + cos_yaw * goal_rel_world[:, 1]
    
    return torch.stack([goal_x_robot, goal_y_robot], dim=-1)


def distance_to_goal(env: ManagerBasedRLEnv, command_name: str = "goal_command") -> torch.Tensor:
    # Get goal command from command manager
    goal_command = env.command_manager.get_command(command_name) 
    goal_pos_world = goal_command[:, :2]  
    
    # Get robot position
    robot: Articulation = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  
    
    # Compute distance
    distance = torch.norm(goal_pos_world - robot_pos, dim=-1, keepdim=True)
    
    return distance


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.action