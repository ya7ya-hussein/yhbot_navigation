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
    threshold: float = 1.0,
    sensor_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Termination condition when collision occurs.
    
    Args:
        env: The environment instance.
        threshold: Force threshold for collision detection.
        sensor_cfg: Configuration for contact sensor.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating collision.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    force_magnitude = torch.norm(net_forces[:, 0, :, :], dim=-1)
    max_force = torch.max(force_magnitude, dim=-1)[0]
    
    return max_force > threshold

def is_robot_flipped(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episode if robot flips over.
    
    Returns True when robot's z-axis points downward (flipped).
    This is a FAILURE termination.
    
    Args:
        env: The environment.
        threshold: Threshold for z-component of up vector. If < threshold, robot is flipped.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Boolean tensor of shape (num_envs,). True = terminate episode.
    """
    # Get robot orientation
    robot: Articulation = env.scene[asset_cfg.name]
    robot_quat_w = robot.data.root_quat_w  
    
    # Extract z-component of robot's up vector
    # Up vector in body frame is [0, 0, 1]
    # Transform to world frame using quaternion
    w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
    
    # Z-component of up vector in world frame
    up_z = 1.0 - 2.0 * (x * x + y * y)
    
    # Return True if robot is flipped (up vector pointing down)
    return up_z < threshold


def is_out_of_bounds(
    env: ManagerBasedRLEnv,
    bounds: tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episode if robot goes out of bounds.
    
    Returns True when robot position is outside the specified rectangular bounds.
    This is a FAILURE termination.
    
    Args:
        env: The environment.
        bounds: Boundary limits (min_x, max_x, min_y, max_y) in meters.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Boolean tensor of shape (num_envs,). True = terminate episode.
    """
    # Get robot position
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  
    
    # Unpack bounds
    min_x, max_x, min_y, max_y = bounds
    
    # Check if out of bounds
    out_of_bounds = (
        (robot_pos_w[:, 0] < min_x) | 
        (robot_pos_w[:, 0] > max_x) | 
        (robot_pos_w[:, 1] < min_y) |  
        (robot_pos_w[:, 1] > max_y)    
    )
    
    return out_of_bounds


def is_stuck(
    env: ManagerBasedRLEnv,
    threshold: float = 0.01,
    time_threshold: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episode if robot is stuck (not moving).
    
    Returns True when robot velocity is below threshold for extended time.
    This is a FAILURE termination (robot is stuck, not making progress).
    
    Args:
        env: The environment.
        threshold: Velocity threshold below which robot is considered stuck (m/s).
        time_threshold: Time duration to be stuck before terminating (seconds).
        asset_cfg: Robot asset configuration.
        
    Returns:
        Boolean tensor of shape (num_envs,). True = terminate episode.
    """
    # Get robot velocity
    robot: Articulation = env.scene[asset_cfg.name]
    lin_vel = robot.data.root_lin_vel_w  
    
    # Compute speed (magnitude of velocity)
    speed = torch.norm(lin_vel[:, :2], dim=-1)  
    
    # Initialize stuck counter if not exists
    if not hasattr(env, "_stuck_counter"):
        env._stuck_counter = torch.zeros(env.num_envs, device=env.device)
    
    # Update stuck counter
    is_moving_slowly = speed < threshold
    env._stuck_counter[is_moving_slowly] += env.step_dt
    env._stuck_counter[~is_moving_slowly] = 0.0  
    
    # Return True if stuck for too long
    return env._stuck_counter > time_threshold