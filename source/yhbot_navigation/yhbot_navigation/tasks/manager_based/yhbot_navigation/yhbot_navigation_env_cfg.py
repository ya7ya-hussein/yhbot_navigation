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

# Import YHBot robot configuration from assets folder
from .assets.yh_bot_cfg import YHBOT_CFG  


##
# Scene definition
##

@configclass
class YhbotNavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for YHBot navigation scene (Grid environment)."""

    # Ground plane 
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # YHBot differential drive robot
    robot: ArticulationCfg = YHBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting 
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    # LiDAR sensor
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="yaw",  
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-179.5, 180.0), 
            horizontal_res=10.0,
        ),
        max_distance=10.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,  
        mesh_prim_paths=["/World/ground"],
    )

    # # Contact sensor - for collision detection
    # contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     update_period=0.0,
    #     history_length=1,
    #     track_air_time=False,
    #     filter_prim_paths_expr=["/World/ground"],
    # )


##
# MDP settings
##


@configclass
class ActionsCfg:    
    # Direct wheel velocity control: 2D action [left_wheel_vel, right_wheel_vel]
    # Agent learns to control wheels directly
    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["middle_left_wheel_joint", "middle_right_wheel_joint"],
        scale=10,  
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # LiDAR observations
        lidar = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")}
        )
        
        # Robot velocities in base frame
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  
        
        # Goal information 
        goal_position_robot_frame = ObsTerm(
            func=mdp.goal_position_robot_frame,
            params={"command_name": "goal_pose"} 
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_pose"}  
        )
        
        # Action history
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    goal_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,  
        resampling_time_range=(1000.0, 1000.0),  
        debug_vis=False,  
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-10.0, 10.0),  
            pos_y=(-10.0, 10.0),  
            heading=(-3.14159, 3.14159),  
        ),
    )


@configclass
class EventCfg:
    # Randomize robot starting pose
    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-10.0, 10.0),  
                "y": (-10.0, 10.0),
                "yaw": (-3.14159, 3.14159),  
            },
            "velocity_range": {
                "x": (0.0, 0.0),   
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:    
    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=500.0,  
        params={"command_name": "goal_pose", "threshold": 0.5}
    )
    
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal,
        weight=10.0,
        params={"command_name": "goal_pose"}
    )
    
    heading_alignment = RewTerm(
        func=mdp.heading_alignment_to_goal,
        weight=0.3, 
        params={"command_name": "goal_pose"}
    )
    
    # collision = RewTerm(
    #     func=mdp.collision_penalty,
    #     weight=-100.0,
    #     params={"threshold": 10.0, "sensor_cfg": SceneEntityCfg("contact_sensor")}  # ← Changed from 1.0 to 10.0
    # )
    
    alive_penalty = RewTerm(
        func=mdp.is_alive,
        weight=-0.01,  
    )
    
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

@configclass
class TerminationsCfg:
    
    # TIME OUT
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    
    # GOAL REACHED: Success termination 
    goal_reached = DoneTerm(
        func=mdp.is_goal_reached,
        params={"command_name": "goal_pose", "threshold": 0.5}
    )
    
    # # 3. COLLISION: Failure termination (contact force > threshold)
    # collision = DoneTerm(
    #     func=mdp.is_collided,
    #     params={"threshold": 10.0, "sensor_cfg": SceneEntityCfg("contact_sensor")}  # ← Changed from 1.0 to 10.0
    # )

##
# Environment configuration
##


@configclass
class YhbotNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for YHBot navigation environment"""
    
    # Scene settings
    scene: YhbotNavigationSceneCfg = YhbotNavigationSceneCfg(num_envs=8192, env_spacing=25.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 4  
        self.episode_length_s = 50
        
        # Viewer settings
        self.viewer.eye = (20.0, 20.0, 15.0) 
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Simulation settings
        self.sim.dt = 1 / 60  
        self.sim.render_interval = self.decimation 