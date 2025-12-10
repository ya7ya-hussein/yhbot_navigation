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
1000000

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

    # LiDAR sensor - 360 rays, 360Â° coverage, 10m range
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="yaw",  
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-179.5, 180.0), 
            horizontal_res=1.0,
        ),
        max_distance=10.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,  
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensor - for collision detection
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
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
    
    # Direct wheel velocity control: 2D action [left_wheel_vel, right_wheel_vel]
    # Agent learns to control wheels directly (differential drive kinematics implicit in learning)
    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["middle_left_wheel_joint", "middle_right_wheel_joint"],
        scale=10,  # Max velocity: Â±0.5 m/s per wheel
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # LiDAR observations - 360 distance measurements
        lidar = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")}
        )
        
        # Robot velocities in base frame
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 3D linear velocity
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 3D angular velocity
        
        # Goal information 
        goal_position_robot_frame = ObsTerm(
            func=mdp.goal_position_robot_frame,
            params={"command_name": "goal_pose"}  # ✅ Matches CommandsCfg
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_pose"}  # ✅ Matches CommandsCfg
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
    """Command terms for the MDP."""
    
    goal_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,  # Always face toward goal
        resampling_time_range=(1000.0, 1000.0),  # Resample every 10 seconds
        debug_vis=True,  # Show goal markers
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-4.0, 4.0),  # Safe range: ±4m from center
            pos_y=(-4.0, 4.0),  # Safe range: ±4m from center
            heading=(-3.14159, 3.14159),  # Full rotation (unused if simple_heading=True)
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""
    
    # On reset: Randomize robot starting pose
    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-4.0, 4.0),   # Spawn anywhere within 8m×8m area (Grid is 10m×10m)
                "y": (-4.0, 4.0),
                "yaw": (-3.14159, 3.14159),  # Random orientation
            },
            "velocity_range": {
                "x": (0.0, 0.0),     # Start from rest
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
    """Reward terms for the MDP."""
    
    # 1. PRIMARY: Goal Reached - Large bonus when reaching goal
    goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=500.0,
        params={"command_name": "goal_pose", "threshold": 0.5}
    )
    
    # 2. PROGRESS: Continuous reward for moving toward goal
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal,
        weight=10.0,
        params={"command_name": "goal_pose"}
    )
    
    # 3. COLLISION: Severe penalty for hitting obstacles (triggers termination)
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-100.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_sensor")} 
    )
    
    # 4. TIME: Small penalty per step (encourages efficiency)
    alive_penalty = RewTerm(
        func=mdp.is_alive,
        weight=-0.01
    )
    
    # 5. HEADING: Bonus for facing toward goal
    heading_alignment = RewTerm(
        func=mdp.heading_alignment_to_goal,
        weight=5.0,
        params={"command_name": "goal_pose"}
    )
    
    # 6. SMOOTHNESS: Penalty for jerky actions
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.5
    )
    
    # 7. ORIENTATION: Penalty for tilting (keep robot upright)
    # Optional - uncomment if robot tips over easily
    # orientation_penalty = RewTerm(
    #     func=mdp.orientation_penalty,
    #     weight=-1.0
    # )
    
    # 8. VELOCITY: Penalty for excessive speed
    # Optional - uncomment if robot moves too fast
    # velocity_penalty = RewTerm(
    #     func=mdp.linear_velocity_penalty,
    #     weight=-0.1
    # )
    
    # 9. ANGULAR VELOCITY: Penalty for excessive spinning
    # Optional - uncomment if robot spins too much
    # angular_penalty = RewTerm(
    #     func=mdp.angular_velocity_penalty,
    #     weight=-0.1
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    # 1. TIME OUT: Maximum episode duration (120 seconds)
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    
    # 2. GOAL REACHED: Success termination (distance < 0.5m)
    goal_reached = DoneTerm(
        func=mdp.is_goal_reached,
        params={"command_name": "goal_pose", "threshold": 0.5}
    )
    
    # 3. COLLISION: Failure termination (contact force > threshold)
    collision = DoneTerm(
        func=mdp.is_collided,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_sensor")}
    )
    
    # 4. ROBOT FLIPPED: Failure termination (optional - if robot can flip)
    # Uncomment if your robot can tip over
    # robot_flipped = DoneTerm(
    #     func=mdp.is_robot_flipped,
    #     params={"threshold": 0.5}
    # )
    
    # 5. OUT OF BOUNDS: Failure termination (optional - for bounded environments)
    # Uncomment if you want to enforce boundaries
    # out_of_bounds = DoneTerm(
    #     func=mdp.is_out_of_bounds,
    #     params={"bounds": (-5.0, 5.0, -5.0, 5.0)}  # (min_x, max_x, min_y, max_y)
    # )
    
    # 6. STUCK: Failure termination (optional - if robot gets stuck)
    # Uncomment if you want to detect stuck robots
    # stuck = DoneTerm(
    #     func=mdp.is_stuck,
    #     params={"threshold": 0.01, "time_threshold": 5.0}
    # )


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
        self.decimation = 4  # Control frequency: 60Hz / 4 = 15Hz policy updates
        self.episode_length_s = 120.0  # 120 seconds max episode duration
        
        # Viewer settings
        self.viewer.eye = (10.0, 10.0, 8.0)  # Camera position for better view of navigation
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Simulation settings
        self.sim.dt = 1 / 60  # 60 Hz physics simulation
        self.sim.render_interval = self.decimation  # Render every decimation steps