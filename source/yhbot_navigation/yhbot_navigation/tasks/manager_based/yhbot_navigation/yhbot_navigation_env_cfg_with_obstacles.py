# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


# YHBot robot configuration
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-10.0, 4.0, 0.0)),
    )

    # YHBot differential drive robot
    robot: ArticulationCfg = YHBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting 
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    # Static obstacles 
    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    obstacle_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_3",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    obstacle_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_4",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    # LiDAR sensor
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/chassis_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0), 
            horizontal_res=3.0,
        ),
        max_distance=3.0,
        drift_range=(0.0, 0.0),
        mesh_prim_paths=[
            "/World/ground",
            "{ENV_REGEX_NS}/obstacle_1",
            "{ENV_REGEX_NS}/obstacle_2",
            "{ENV_REGEX_NS}/obstacle_3",
            "{ENV_REGEX_NS}/obstacle_4",
        ],
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )

    contact_forces_roof = ContactSensorCfg(  
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/roof_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False, 
    )



##
# MDP settings
##


@configclass
class ActionsCfg:    
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

# for training
@configclass 
class CommandsCfg:
    goal_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,  
        resampling_time_range=(1000.0, 1000.0),  
        debug_vis=True,  
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-13.0, 13.0),  
            pos_y=(-25.0, 25.0),
            heading=(-3.14159, 3.14159),  
        ),
    )



@configclass
class EventCfg:
    # Randomize robot starting pose for training
    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-13.0, 13.0),
                "y": (-25.0, 25.0),
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

    # Randomize obstacle positions 
    reset_obstacle_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-11.0, 11.0), "y": (-23.0, 23.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )

    reset_obstacle_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-11.0, 11.0), "y": (-23.0, 23.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_2"),
        },
    )

    reset_obstacle_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-11.0, 11.0), "y": (-23.0, 23.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_3"),
        },
    )

    reset_obstacle_4 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-11.0, 11.0), "y": (-23.0, 23.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_4"),
        },
    )

@configclass
class RewardsCfg:    
    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=500.0,  
        params={"command_name": "goal_pose", "threshold": 0.6}
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
        
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-100.0,
        params={"force_threshold": 0.5, "startup_grace_steps": 45, "chassis_sensor_cfg": SceneEntityCfg("contact_forces"), "roof_sensor_cfg": SceneEntityCfg("contact_forces_roof"),},
    )

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
        params={"command_name": "goal_pose", "threshold": 0.6}, 
        time_out=False
    )
    
    # COLLISION: Failure termination 
    collision = DoneTerm(
        func=mdp.is_collided,
        params={"force_threshold": 0.5, "startup_grace_steps": 45, "chassis_sensor_cfg": SceneEntityCfg("contact_forces"), "roof_sensor_cfg": SceneEntityCfg("contact_forces_roof"),},
        time_out=False
    )
##
# Environment configuration
##


@configclass
class YhbotNavigationObsEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for YHBot navigation environment"""
    
    # Scene settings
    scene: YhbotNavigationSceneCfg = YhbotNavigationSceneCfg(num_envs=8192, env_spacing=60.0)
    
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
        self.episode_length_s = 70
        
        # Viewer settings
        self.viewer.eye = (20.0, 20.0, 15.0) 
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Simulation settings
        self.sim.dt = 1 / 60  
        self.sim.render_interval = self.decimation 