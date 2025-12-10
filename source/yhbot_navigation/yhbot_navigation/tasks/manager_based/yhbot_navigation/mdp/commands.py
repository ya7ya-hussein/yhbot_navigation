# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command generators for YHBot navigation task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import UniformPose2dCommandCfg


class UniformPose2dCommand(CommandTerm):
    """Command term that generates uniform random 2D pose commands for navigation.
    
    This command generator samples random (x, y) goal positions within specified ranges
    and optionally a heading angle. The goals are generated in the world frame and
    resampled after a specified time interval.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: Configuration for the command generator.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Get the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Create buffers to store command
        # Command: [pos_x, pos_y, heading]
        self.pos_command_w = torch.zeros(env.num_envs, 2, device=env.device)
        self.heading_command_w = torch.zeros(env.num_envs, device=env.device)

        # Track metrics
        self.metrics["error_pos"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics["error_heading"] = torch.zeros(env.num_envs, device=env.device)

    def __str__(self) -> str:
        """String representation of the command generator."""
        msg = "UniformPose2dCommand:\n"
        msg += f"\tCommand dimension: 3 (x, y, heading)\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose command in world frame. Shape is (num_envs, 3)."""
        return torch.cat([self.pos_command_w, self.heading_command_w.unsqueeze(-1)], dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update the command metrics based on current state."""
        # Get current robot position in world frame
        robot_pos_w = self.robot.data.root_pos_w[:, :2]  # Only x, y
        robot_heading_w = self._compute_heading(self.robot.data.root_quat_w)

        # Compute position error (Euclidean distance)
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - robot_pos_w, dim=-1)
        
        # Compute heading error (angular difference)
        heading_error = self.heading_command_w - robot_heading_w
        # Normalize to [-pi, pi]
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        self.metrics["error_heading"] = torch.abs(heading_error)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments.

        Args:
            env_ids: Environment indices to resample commands for.
        """
        # Sample random positions within the specified range
        self.pos_command_w[env_ids, 0] = torch.rand(
            len(env_ids), device=self.pos_command_w.device
        ) * (self.cfg.ranges.pos_x[1] - self.cfg.ranges.pos_x[0]) + self.cfg.ranges.pos_x[0]
        
        self.pos_command_w[env_ids, 1] = torch.rand(
            len(env_ids), device=self.pos_command_w.device
        ) * (self.cfg.ranges.pos_y[1] - self.cfg.ranges.pos_y[0]) + self.cfg.ranges.pos_y[0]

        # Sample heading
        if self.cfg.simple_heading:
            # Simple heading: point towards the goal from current position
            robot_pos = self.robot.data.root_pos_w[env_ids, :2]
            direction = self.pos_command_w[env_ids] - robot_pos
            self.heading_command_w[env_ids] = torch.atan2(direction[:, 1], direction[:, 0])
        else:
            # Random heading within specified range
            self.heading_command_w[env_ids] = torch.rand(
                len(env_ids), device=self.heading_command_w.device
            ) * (self.cfg.ranges.heading[1] - self.cfg.ranges.heading[0]) + self.cfg.ranges.heading[0]

    def _update_command(self):
        """Update the command based on current state.
        
        For navigation tasks, we don't need to update the command during the episode.
        The command stays fixed until it's resampled.
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization for goal markers.

        Args:
            debug_vis: Whether to enable debug visualization.
        """
        # Create or destroy markers based on flag
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                # Create goal marker (sphere)
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # Set visibility
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        # Check if scene still exists (prevents shutdown errors)
        if not hasattr(self._env, "scene") or self._env.scene is None:
            return
            
        # Update marker positions
        self._goal_markers.visualize(
            translations=self.pos_command_w[:, :3],
            orientations=self.heading_command_w,
            scales=torch.ones(self._env.num_envs, 1, device=self.pos_command_w.device) * 0.2
        )
    """
    Helper functions.
    """

    def _compute_heading(self, quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle (heading) from quaternion.

        Args:
            quat: Quaternion tensor in (w, x, y, z) format. Shape (num_envs, 4).

        Returns:
            Yaw angles in radians. Shape (num_envs,).
        """
        # Extract yaw from quaternion
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _quat_from_yaw(self, yaw: torch.Tensor) -> torch.Tensor:
        """Create quaternion from yaw angle (rotation around z-axis).

        Args:
            yaw: Yaw angles in radians. Shape (num_envs,).

        Returns:
            Quaternions in (w, x, y, z) format. Shape (num_envs, 4).
        """
        quat = torch.zeros(len(yaw), 4, device=yaw.device)
        quat[:, 0] = torch.cos(yaw / 2.0)  # w
        quat[:, 3] = torch.sin(yaw / 2.0)  # z
        return quat