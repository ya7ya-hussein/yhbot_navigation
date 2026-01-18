# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for custom command generators."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .commands import UniformPose2dCommand


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING

    simple_heading: bool = True

    @configclass
    class Ranges:

        pos_x: tuple[float, float] = MISSING

        pos_y: tuple[float, float] = MISSING

        heading: tuple[float, float] = (-3.14159, 3.14159)

    ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )

    # Set the scale
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)