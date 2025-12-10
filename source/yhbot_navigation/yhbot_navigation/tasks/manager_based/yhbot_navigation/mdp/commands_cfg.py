# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for custom command generators."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .commands import UniformPose2dCommand


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for uniform 2D pose command generator for navigation.
    
    This command generator samples random (x, y) goal positions within specified ranges
    and optionally a heading angle for navigation tasks.
    """

    class_type: type = UniformPose2dCommand
    """Class of the command generator term. Default is UniformPose2dCommand."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = True
    """Whether to use simple heading or not. Default is True.
    
    If True, the heading points toward the goal position from the robot's current position.
    Otherwise, the heading is sampled uniformly from the specified range.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = (-3.14159, 3.14159)
        """Heading range for the position commands (in rad).
        
        Used only if :attr:`simple_heading` is False. Default is (-pi, pi).
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    # ✅ NEW: Use sphere marker for goal visualization
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_pose",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
            ),
        },
    )
    """The configuration for the goal pose visualization marker (green sphere)."""