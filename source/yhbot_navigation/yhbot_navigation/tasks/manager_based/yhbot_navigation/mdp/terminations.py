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
    sudden_decel_threshold: float = 0.25,
    displacement_threshold: float = 1.0,
    angular_velocity_threshold: float = 0.15,
    goal_proximity_threshold: float = 1.0,
    startup_grace_steps: int = 45,
    chassis_sensor_cfg: "SceneEntityCfg" = None,
    roof_sensor_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Multi-layer collision detection based on 200-episode data analysis.
    
    Detection strategy:
    1. Startup grace period (first 3 seconds)
    2. Sudden deceleration detection (velocity drop >0.25 m/s WHILE commanding forward)
    3. Contact force detection (backup)
    4. Goal proximity check (allow slow movement near goal)
    5. Progress tracking (is robot getting closer to goal?)
    6. Movement check (displacement over 2 seconds)
    7. Turning check (angular velocity)
    
    Key insight for collision detection (from 60-episode analysis):
    - When robot intentionally brakes: it commands opposite wheel signs (turning/reversing)
      → Decelerations up to 0.37 m/s are NORMAL and should NOT terminate
    - When robot collides: it was commanding both wheels positive (forward)
      → Any decel >0.25 m/s while commanding forward = COLLISION
    - Max observed decel while commanding forward: 0.013 m/s (normal physics)
    
    Thresholds based on statistical analysis:
    - sudden_decel_threshold: 0.25 m/s (only triggers when both wheels positive)
    - displacement_threshold: 1.0 m (normal navigation: 1.59-1.88m, stuck: 0.3-1.5m)
    - angular_velocity_threshold: 0.15 rad/s (turning shows 0.5-0.58 rad/s)
    - goal_proximity_threshold: 1.0 m (allow adjustment behavior)
    
    Args:
        env: The environment instance.
        force_threshold: Contact force threshold (N).
        sudden_decel_threshold: Velocity drop threshold for collision detection (m/s).
        displacement_threshold: Minimum displacement over 2s for movement check (m).
        angular_velocity_threshold: Minimum angular velocity for turning detection (rad/s).
        goal_proximity_threshold: Distance to goal for proximity zone (m).
        startup_grace_steps: Steps to wait before checking collision (3 seconds at 15Hz).
        chassis_sensor_cfg: Chassis contact sensor configuration.
        roof_sensor_cfg: Roof contact sensor configuration.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating collision termination.
    """
    
    # ========== STEP 1: Startup Grace Period ==========
    # Don't check collision during first 3 seconds (robot accelerating from rest)
    startup_grace = env.episode_length_buf < startup_grace_steps
    
    # ========== STEP 2: Sudden Deceleration Detection ==========
    # Initialize previous velocity tracker
    if not hasattr(env, "_prev_velocity_collision"):
        env._prev_velocity_collision = torch.zeros(env.num_envs, device=env.device)
    
    # Get current velocity
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_w
    current_speed = torch.norm(lin_vel[:, :2], dim=-1)
    
    # Calculate velocity change (positive = deceleration)
    velocity_change = env._prev_velocity_collision - current_speed
    
    # Get current action to check if robot was commanding forward motion
    # Action shape: (num_envs, 2) where [0]=left_wheel, [1]=right_wheel
    action = env.action_manager.action
    both_wheels_positive = (action[:, 0] > 0) & (action[:, 1] > 0)
    
    # Sudden deceleration indicates collision ONLY if robot was commanding forward
    # Key insight from 60-episode analysis:
    # - Intentional braking: robot commands opposite wheel signs (turning/reversing) → decel up to 0.37 m/s is NORMAL
    # - Collision: robot commands both wheels positive (forward) but velocity drops → COLLISION
    # - Max observed decel while commanding forward: 0.013 m/s (normal physics)
    # - Any significant decel (>0.25 m/s) while commanding forward = external force stopped robot
    sudden_collision = (velocity_change > sudden_decel_threshold) & both_wheels_positive
    
    # Update previous velocity for next step
    env._prev_velocity_collision = current_speed.clone()
    
    # ========== STEP 3: Contact Force Detection (Backup) ==========
    # Check chassis contact
    chassis_sensor = env.scene[chassis_sensor_cfg.name]
    chassis_force = chassis_sensor.data.force_matrix_w
    if chassis_force is not None:
        chassis_mag = torch.norm(chassis_force, dim=-1)
        max_chassis = torch.max(chassis_mag.view(env.num_envs, -1), dim=-1)[0]
        chassis_collision = max_chassis > force_threshold
    else:
        chassis_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check roof contact
    roof_sensor = env.scene[roof_sensor_cfg.name]
    roof_force = roof_sensor.data.force_matrix_w
    if roof_force is not None:
        roof_mag = torch.norm(roof_force, dim=-1)
        max_roof = torch.max(roof_mag.view(env.num_envs, -1), dim=-1)[0]
        roof_collision = max_roof > force_threshold
    else:
        roof_collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Combine contact detections (unreliable but use as backup)
    force_collision = chassis_collision | roof_collision
    
    # ========== STEP 4: Goal Proximity Check ==========
    # Allow slow movement and adjustments when near goal
    robot_pos = robot.data.root_pos_w[:, :2]
    command = env.command_manager.get_command("goal_pose")
    goal_pos = command[:, :2]
    distance_to_goal = torch.norm(goal_pos - robot_pos, dim=-1)
    
    near_goal = distance_to_goal < goal_proximity_threshold
    
    # ========== STEP 5: Progress Tracking (2-second window) ==========
    # Initialize distance history buffer (30 steps = 2 seconds at 15Hz)
    if not hasattr(env, "_distance_history"):
        env._distance_history = torch.zeros(env.num_envs, 30, device=env.device)
        env._history_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Store current distance at circular buffer index
    env._distance_history[torch.arange(env.num_envs), env._history_index] = distance_to_goal
    
    # Get distance from 2 seconds ago (30 steps back in circular buffer)
    distance_2s_ago_index = (env._history_index - 29) % 30
    distance_2s_ago = env._distance_history[torch.arange(env.num_envs), distance_2s_ago_index]
    
    # Update circular buffer index
    env._history_index = (env._history_index + 1) % 30
    
    # Calculate progress (negative = getting closer to goal)
    distance_change_2s = distance_to_goal - distance_2s_ago
    making_progress = distance_change_2s < 0  # True if getting closer
    
    # ========== STEP 6: Net Displacement Check (2-second window) ==========
    # Initialize position history buffer
    if not hasattr(env, "_position_history"):
        env._position_history = torch.zeros(env.num_envs, 30, 2, device=env.device)
        env._position_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Store current position
    current_pos = robot_pos.clone()
    env._position_history[torch.arange(env.num_envs), env._position_index] = current_pos
    
    # Get position from 2 seconds ago
    position_2s_ago_index = (env._position_index - 29) % 30
    position_2s_ago = env._position_history[torch.arange(env.num_envs), position_2s_ago_index]
    
    # Update position buffer index
    env._position_index = (env._position_index + 1) % 30
    
    # Calculate net displacement (actual distance traveled)
    net_displacement = torch.norm(current_pos - position_2s_ago, dim=-1)
    
    # Robot is moving if displacement exceeds threshold
    # Normal navigation: 1.59-1.88m, Stuck episodes: 0.3-1.5m (from analysis)
    is_moving = net_displacement >= displacement_threshold
    
    # ========== STEP 7: Turning Check ==========
    # Get angular velocity (yaw rate)
    ang_vel = robot.data.root_ang_vel_w
    yaw_rate = torch.abs(ang_vel[:, 2])  # Z-axis rotation
    
    # Robot is turning if angular velocity exceeds threshold
    # Turning episodes show 0.50-0.58 rad/s (from analysis)
    is_turning = yaw_rate > angular_velocity_threshold
    
    # ========== STEP 8: Combine All Checks ==========
    # Immediate terminations (high confidence)
    immediate_collision = sudden_collision | force_collision
    
    # Stuck detection (all conditions must be met)
    # - NOT making progress toward goal
    # - NOT moving significantly
    # - NOT turning
    # - NOT near goal (where slow movement is expected)
    stuck_collision = (
        ~making_progress &  # Not getting closer to goal
        ~is_moving &        # Not moving significantly
        ~is_turning &       # Not rotating
        ~near_goal          # Not in goal adjustment zone
    )
    
    # Combine immediate and stuck detections, but exclude startup period
    collision_detected = (immediate_collision | stuck_collision) & ~startup_grace
    
    return collision_detected