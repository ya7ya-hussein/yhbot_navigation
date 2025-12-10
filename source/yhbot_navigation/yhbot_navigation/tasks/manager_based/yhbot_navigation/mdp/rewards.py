# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for YHBot navigation task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus reward when robot reaches goal.
    
    Returns 1.0 when distance to goal < threshold, else 0.0.
    This is a terminal reward - episode should end when goal is reached.
    
    Args:
        env: The environment.
        command_name: Name of the goal command.
        threshold: Distance threshold to consider goal reached (in meters).
        asset_cfg: Robot asset configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get goal position from command manager
    goal_command = env.command_manager.get_command(command_name)
    goal_pos_w = goal_command[:, :2]  # (num_envs, 2) - [x, y]
    
    # Get robot position
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute distance to goal
    distance = torch.norm(goal_pos_w - robot_pos_w, dim=-1)
    
    # Return 1.0 if within threshold, else 0.0
    return (distance < threshold).float()


def progress_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for making progress toward the goal.
    
    Returns the distance reduction from previous step to current step.
    Positive reward for moving closer, negative for moving farther.
    
    Args:
        env: The environment.
        command_name: Name of the goal command.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get goal position from command manager
    goal_command = env.command_manager.get_command(command_name)
    goal_pos_w = goal_command[:, :2]  # (num_envs, 2)
    
    # Get robot position
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Compute current distance
    current_distance = torch.norm(goal_pos_w - robot_pos_w, dim=-1)
    
    # Get previous distance (stored in environment buffer)
    # Initialize buffer on first call
    if not hasattr(env, "_previous_goal_distance"):
        env._previous_goal_distance = current_distance.clone()
    
    # Compute progress: positive if moving closer, negative if moving away
    progress = env._previous_goal_distance - current_distance
    
    # Update buffer for next step
    env._previous_goal_distance = current_distance.clone()
    
    return progress


def collision_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalty for collision with obstacles or walls.
    
    Returns 1.0 when collision is detected (contact force > threshold), else 0.0.
    This should trigger episode termination.
    
    Args:
        env: The environment.
        threshold: Contact force threshold to consider collision (in Newtons).
        sensor_cfg: Contact sensor configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get net contact forces on the robot body
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # Compute maximum force magnitude across all bodies
    force_magnitude = torch.max(torch.norm(net_contact_forces[:, :, 0], dim=-1), dim=1)[0]
    
    # Return 1.0 if force exceeds threshold (collision detected)
    return (force_magnitude > threshold).float()


def heading_alignment_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for aligning robot heading toward goal.
    
    Returns cosine of angle between robot heading and direction to goal.
    Returns value in range [-1, 1]: +1 when perfectly aligned, -1 when opposite.
    
    Args:
        env: The environment.
        command_name: Name of the goal command.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get goal position from command manager
    goal_command = env.command_manager.get_command(command_name)
    goal_pos_w = goal_command[:, :2]  # (num_envs, 2)
    
    # Get robot state
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4) - [w, x, y, z]
    
    # Compute direction to goal (normalized)
    direction_to_goal = goal_pos_w - robot_pos_w
    distance = torch.norm(direction_to_goal, dim=-1, keepdim=True)
    direction_to_goal = direction_to_goal / (distance + 1e-6)  # Normalize
    
    # Extract robot heading from quaternion (yaw angle)
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Robot heading vector (forward direction in world frame)
    robot_heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    
    # Compute cosine of angle (dot product)
    alignment = torch.sum(robot_heading * direction_to_goal, dim=-1)
    
    return alignment


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for jerky/abrupt actions (action changes).
    
    Computes L2 norm of action change: ||action_current - action_previous||^2
    Encourages smooth control and helps with real-world transfer.
    
    Args:
        env: The environment.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get current action from action manager
    current_action = env.action_manager.action
    
    # Get previous action (stored in environment buffer)
    # Initialize buffer on first call
    if not hasattr(env, "_previous_action"):
        env._previous_action = torch.zeros_like(current_action)
    
    # Compute L2 norm of action change
    action_diff = current_action - env._previous_action
    smoothness_penalty = torch.sum(action_diff ** 2, dim=-1)
    
    # Update buffer for next step
    env._previous_action = current_action.clone()
    
    return smoothness_penalty


def orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for robot tilting (roll/pitch angles).
    
    Penalizes non-upright orientation to encourage stable navigation.
    Useful for preventing the robot from flipping or tilting excessively.
    
    Args:
        env: The environment.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get robot orientation
    robot: Articulation = env.scene[asset_cfg.name]
    robot_quat_w = robot.data.root_quat_w  # (num_envs, 4) - [w, x, y, z]
    
    # Extract roll and pitch from quaternion
    w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
    
    # Roll (rotation around x-axis)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    
    # Pitch (rotation around y-axis)
    pitch = torch.asin(2.0 * (w * y - z * x))
    
    # Penalty is sum of squared angles (zero when perfectly upright)
    orientation_penalty = roll ** 2 + pitch ** 2
    
    return orientation_penalty


def linear_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for excessive linear velocity.
    
    Penalizes very high speeds to encourage safe navigation.
    Can help prevent collision due to high speed.
    
    Args:
        env: The environment.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get robot velocity
    robot: Articulation = env.scene[asset_cfg.name]
    lin_vel = robot.data.root_lin_vel_w  # (num_envs, 3)
    
    # Compute speed (magnitude of velocity)
    speed = torch.norm(lin_vel, dim=-1)
    
    # Quadratic penalty for high speeds (zero at low speeds)
    velocity_penalty = speed ** 2
    
    return velocity_penalty


def angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for excessive angular velocity (spinning).
    
    Penalizes very high rotation speeds to encourage smooth turning.
    
    Args:
        env: The environment.
        asset_cfg: Robot asset configuration.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get robot angular velocity
    robot: Articulation = env.scene[asset_cfg.name]
    ang_vel = robot.data.root_ang_vel_w  # (num_envs, 3)
    
    # We only care about yaw rate (rotation around z-axis)
    yaw_rate = ang_vel[:, 2]
    
    # Quadratic penalty for high rotation rates
    angular_penalty = yaw_rate ** 2
    
    return angular_penalty