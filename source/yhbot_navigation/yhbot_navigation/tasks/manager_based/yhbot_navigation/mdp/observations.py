# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for YHBot navigation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")) -> torch.Tensor:
    """LiDAR observations - 360 distance measurements.
    
    Returns normalized distances from ray-caster sensor.
    Shape: (num_envs, 360)
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # Get ray hit distances
    # ray_hits_w shape: (num_envs, num_rays, 3) - hit positions in world frame
    # Calculate distance from sensor origin to hit point
    ray_origins = sensor.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    ray_hits = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)
    
    # Compute distances
    distances = torch.norm(ray_hits - ray_origins, dim=-1)  # (num_envs, num_rays)
    
    # Handle inf values (no hit) - replace with max distance
    distances = torch.where(torch.isinf(distances), torch.tensor(sensor.cfg.max_distance, device=env.device), distances)
    
    # Normalize to [0, 1]
    distances = distances / sensor.cfg.max_distance
    
    return distances


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in robot's base frame.
    
    Shape: (num_envs, 3) - [vx, vy, vz] in m/s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in robot's base frame.
    
    Shape: (num_envs, 3) - [wx, wy, wz] in rad/s
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def goal_position_robot_frame(env: ManagerBasedRLEnv, command_name: str = "goal_command") -> torch.Tensor:
    """Goal position in robot's local frame.
    
    Returns 2D position (x, y) of goal relative to robot.
    Shape: (num_envs, 2)
    """
    # Get goal command from command manager
    goal_command = env.command_manager.get_command(command_name)  # (num_envs, 3) - [x, y, heading]
    goal_pos_world = goal_command[:, :2]  # (num_envs, 2) - only x, y ✅ SLICE IT!
    
    # Get robot state
    robot: Articulation = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  # (num_envs, 2) - [x, y] in world
    robot_quat = robot.data.root_quat_w  # (num_envs, 4) - [w, x, y, z]
    
    # Compute relative position in world frame
    goal_rel_world = goal_pos_world - robot_pos  # (num_envs, 2)
    
    # Convert to robot frame using yaw rotation
    # Extract yaw from quaternion
    yaw = torch.atan2(
        2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    )
    
    # Rotation matrix (2D)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    # Transform to robot frame
    goal_x_robot = cos_yaw * goal_rel_world[:, 0] + sin_yaw * goal_rel_world[:, 1]
    goal_y_robot = -sin_yaw * goal_rel_world[:, 0] + cos_yaw * goal_rel_world[:, 1]
    
    return torch.stack([goal_x_robot, goal_y_robot], dim=-1)


def distance_to_goal(env: ManagerBasedRLEnv, command_name: str = "goal_command") -> torch.Tensor:
    """Euclidean distance to goal position.
    
    Shape: (num_envs, 1)
    """
    # Get goal command from command manager
    goal_command = env.command_manager.get_command(command_name)  # (num_envs, 3) - [x, y, heading]
    goal_pos_world = goal_command[:, :2]  # (num_envs, 2) - only x, y ✅ SLICE IT!
    
    # Get robot position
    robot: Articulation = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute distance
    distance = torch.norm(goal_pos_world - robot_pos, dim=-1, keepdim=True)
    
    return distance


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action taken by the agent.
    
    Shape: (num_envs, action_dim) - typically (num_envs, 2) for differential drive
    """
    return env.action_manager.action