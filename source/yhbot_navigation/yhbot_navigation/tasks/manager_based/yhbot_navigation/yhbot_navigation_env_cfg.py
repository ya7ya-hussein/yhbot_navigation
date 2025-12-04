# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# Import YHBot robot configuration from assets folder
from .assets.yh_bot_cfg import YHBOT_CFG  # isort:skip


##
# Scene definition
##


@configclass
class YhbotNavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for YHBot navigation scene (Grid environment)."""

    # Ground plane - simple flat terrain for Grid environment
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # YHBot differential drive robot
    robot: ArticulationCfg = YHBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting - dome light for uniform illumination
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    # LiDAR sensor - 360 rays, 360° coverage, 10m range
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="yaw",  # Fixed: was attach_yaw_only (deprecated)
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=10.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,  # Disabled to avoid visualization errors
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensor - for collision detection
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/.*",  # yh_bot becomes Robot!
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Temporary: Reference real wheel joints (will be configured properly in Step 3)
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=["middle_left_wheel_joint", "middle_right_wheel_joint"], 
        scale=10.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # Events will be added in Step 12
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # Rewards will be configured in Steps 8-9
    # For now, just give alive reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # Terminations will be configured in Steps 10-11
    # For now, just timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class YhbotNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for YHBot navigation environment (Grid)."""
    
    # Scene settings
    scene: YhbotNavigationSceneCfg = YhbotNavigationSceneCfg(num_envs=4096, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 4  # Control frequency: 60Hz / 4 = 15Hz policy updates
        self.episode_length_s = 120.0  # 120 seconds max episode duration
        
        # Viewer settings
        self.viewer.eye = (10.0, 10.0, 8.0)  # Camera position for better view of navigation
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Simulation settings
        self.sim.dt = 1 / 60  # 60 Hz physics simulation
        self.sim.render_interval = self.decimation  # Render every decimation steps