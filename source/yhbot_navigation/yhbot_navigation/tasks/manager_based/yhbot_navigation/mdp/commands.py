# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import UniformPose2dCommandCfg


class UniformPose2dCommand(CommandTerm):

    cfg: UniformPose2dCommandCfg

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Get the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Create buffers to store command
        self.pos_command_w = torch.zeros(env.num_envs, 2, device=env.device)
        self.heading_command_w = torch.zeros(env.num_envs, device=env.device)

        # Track metrics
        self.metrics["error_pos"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics["error_heading"] = torch.zeros(env.num_envs, device=env.device)

    def __str__(self) -> str:
        msg = "UniformPose2dCommand:\n"
        msg += f"\tCommand dimension: 3 (x, y, heading)\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg


    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.pos_command_w, self.heading_command_w.unsqueeze(-1)], dim=-1)


    def _resample_command(self, env_ids: Sequence[int]):
        num_envs = len(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, device=self.pos_command_w.device, dtype=torch.long)
        env_origins = self._env.scene.env_origins[env_ids, :2]
        
        # Obstacle names for rejection sampling
        obstacle_names = ["obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4", "obstacle_5"]
        safety_radius = 1.5
        max_attempts = 50
        
        # Get obstacle positions for the environments being resampled
        obstacle_positions = []
        for obs_name in obstacle_names:
            if obs_name in self._env.scene.rigid_objects:
                obs_pos = self._env.scene.rigid_objects[obs_name].data.root_pos_w[env_ids, :2]
                obstacle_positions.append(obs_pos)
        
        # Initialize valid mask
        needs_resample = torch.ones(num_envs, dtype=torch.bool, device=self.pos_command_w.device)
        final_pos_local = torch.zeros(num_envs, 2, device=self.pos_command_w.device)
        
        for attempt in range(max_attempts):
            if not needs_resample.any():
                break
            
            n_resample = needs_resample.sum().item()
            
            # Sample random positions for environments that still need valid goals
            pos_local = torch.zeros(n_resample, 2, device=self.pos_command_w.device)
            pos_local[:, 0] = torch.rand(n_resample, device=self.pos_command_w.device) * \
                            (self.cfg.ranges.pos_x[1] - self.cfg.ranges.pos_x[0]) + self.cfg.ranges.pos_x[0]
            pos_local[:, 1] = torch.rand(n_resample, device=self.pos_command_w.device) * \
                            (self.cfg.ranges.pos_y[1] - self.cfg.ranges.pos_y[0]) + self.cfg.ranges.pos_y[0]
            
            # Convert to world frame
            pos_world = pos_local + env_origins[needs_resample]
            
            # Check distance to all obstacles
            valid = torch.ones(n_resample, dtype=torch.bool, device=self.pos_command_w.device)
            for obs_pos in obstacle_positions:
                obs_pos_subset = obs_pos[needs_resample]
                dist = torch.norm(pos_world - obs_pos_subset, dim=-1)
                valid = valid & (dist > safety_radius)
            
            # Update final positions for valid samples
            resample_indices = torch.where(needs_resample)[0]
            valid_indices = resample_indices[valid]
            final_pos_local[valid_indices] = pos_local[valid]
            needs_resample[valid_indices] = False
        
        # Fallback: use environment center for any remaining invalid positions
        if needs_resample.any():
            final_pos_local[needs_resample] = 0.0
        
        # Set final positions in world frame
        self.pos_command_w[env_ids] = final_pos_local + env_origins
        
        # Sample heading
        if self.cfg.simple_heading:
            robot_pos = self.robot.data.root_pos_w[env_ids, :2]
            direction = self.pos_command_w[env_ids] - robot_pos
            self.heading_command_w[env_ids] = torch.atan2(direction[:, 1], direction[:, 0])
        else:
            self.heading_command_w[env_ids] = torch.rand(num_envs, device=self.heading_command_w.device) * \
                                            (self.cfg.ranges.heading[1] - self.cfg.ranges.heading[0]) + \
                                            self.cfg.ranges.heading[0]

    def _update_command(self):
        # The goals only change when _resample_command() is called
        pass

    def _update_metrics(self):
        """Update the command metrics based on current state."""
        # Get current robot position in world frame
        robot_pos_w = self.robot.data.root_pos_w[:, :2]  
        robot_heading_w = self._compute_heading(self.robot.data.root_quat_w)

        # Compute position error (Euclidean distance)
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - robot_pos_w, dim=-1)
        
        # Compute heading error (angular difference)
        heading_error = self.heading_command_w - robot_heading_w
        # Normalize to [-pi, pi]
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        self.metrics["error_heading"] = torch.abs(heading_error)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create or remove visualization markers
        if debug_vis and self.cfg.goal_pose_visualizer_cfg is not None:
            if not hasattr(self, "goal_visualizer"):
                # Create the goal pose visualizer
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
        else:
            if hasattr(self, "goal_visualizer"):
                # Remove the visualizer
                delattr(self, "goal_visualizer")

    def _debug_vis_callback(self, event):
        # Update goal visualization markers
        if hasattr(self, "goal_visualizer"):
            # Create 3D positions for visualization 
            goal_pos_3d = torch.cat([
                self.pos_command_w,
                torch.ones(self._env.num_envs, 1, device=self.pos_command_w.device) * 0.2
            ], dim=-1)
            
            # Create quaternions from heading angles
            goal_quat = self._quat_from_yaw(self.heading_command_w)
            
            # Visualize the markers
            self.goal_visualizer.visualize(
                translations=goal_pos_3d,
                orientations=goal_quat
            )

    def _compute_heading(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _quat_from_yaw(self, yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros(len(yaw), 4, device=yaw.device)
        quat[:, 0] = torch.cos(yaw / 2.0)  
        quat[:, 3] = torch.sin(yaw / 2.0)  
        return quat