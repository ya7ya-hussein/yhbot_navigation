# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import MultiMeshRayCasterCfg

from . import mdp
from .assets.yh_bot_cfg import YHBOT_CFG


# Asset paths
WAREHOUSE_ASSETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets/Simple_Warehouse"
)

WAREHOUSE_USD_PATH = os.path.join(WAREHOUSE_ASSETS_PATH, "full_warehouse.usd")

##
# Scene definition
##

@configclass
class YhbotWarehouseTestSceneCfg(InteractiveSceneCfg):
    """Configuration for YHBot warehouse test scene with real USD assets."""

    # Ground plane 
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(60.0, 80.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Full warehouse USD 
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WAREHOUSE_USD_PATH,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # YHBot differential drive robot
    robot: ArticulationCfg = YHBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    # South wall collision proxy
    wall_south_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_south_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(32.0, 0.5, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.5, -23.25, 1.0)),
    )
    
    # North wall collision proxy
    wall_north_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_north_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(32.0, 0.5, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.5, 31.25, 1.0)),
    )
    
    # West wall collision proxy
    wall_west_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_west_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 55.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-26.25, 4.0, 1.0)),
    )
    
    # East wall collision proxy
    wall_east_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_east_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 55.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.25, 4.0, 1.0)),
    )

    # Rack
    rack_1_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_1_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-25.0, 17.0, 1.25)),
    )
    
    rack_2_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_2_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-20.4, 17.0, 1.25)),
    )
    
    rack_3_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_3_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-15.4, 17.0, 1.25)),
    )
    
    rack_4_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_4_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.5, 17.0, 1.25)),
    )
    
    rack_5_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_5_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.5, 17.0, 1.25)),
    )
    
    rack_6_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_6_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 17.0, 1.25)),
    )
    
    rack_7_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_7_proxy",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.4, 17.0, 1.25)),
    )

    # Visible obstacles 
    static_barrel: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_barrel",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=200.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.27, 0.27, 0.27)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.5)),
    )

    static_cone: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/static_cone",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=200.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.27, 0.27, 0.27)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-15.0, 0.0, 0.4)),
    )

    dynamic_obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacle_1",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 2.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=200.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-19.0, 17.0, 0.75)),
    )

    dynamic_obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacle_2",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 2.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=200.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.9, 17.0, 0.75)),
    )

    
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/chassis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0), 
            horizontal_res=3.0,
        ),
        max_distance=3.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
            # Wall proxies
            "{ENV_REGEX_NS}/wall_south_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/wall_north_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/wall_west_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/wall_east_proxy/geometry/mesh",
            # Rack proxies
            "{ENV_REGEX_NS}/rack_1_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/rack_2_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/rack_3_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/rack_4_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/rack_5_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/rack_6_proxy/geometry/mesh",
            "{ENV_REGEX_NS}/rack_7_proxy/geometry/mesh",
            # Visible obstacles
            "{ENV_REGEX_NS}/static_barrel/geometry/mesh",
            "{ENV_REGEX_NS}/static_cone/geometry/mesh",
            "{ENV_REGEX_NS}/dynamic_obstacle_1/geometry/mesh",
            "{ENV_REGEX_NS}/dynamic_obstacle_2/geometry/mesh",
        ],
    )

    # Contact sensors
    contact_forces_chassis = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Warehouse",
            "{ENV_REGEX_NS}/wall_south_proxy",
            "{ENV_REGEX_NS}/wall_north_proxy",
            "{ENV_REGEX_NS}/wall_west_proxy",
            "{ENV_REGEX_NS}/wall_east_proxy",
            "{ENV_REGEX_NS}/rack_1_proxy",
            "{ENV_REGEX_NS}/rack_2_proxy",
            "{ENV_REGEX_NS}/rack_3_proxy",
            "{ENV_REGEX_NS}/rack_4_proxy",
            "{ENV_REGEX_NS}/rack_5_proxy",
            "{ENV_REGEX_NS}/rack_6_proxy",
            "{ENV_REGEX_NS}/rack_7_proxy",
            "{ENV_REGEX_NS}/static_barrel",
            "{ENV_REGEX_NS}/static_cone",
            "{ENV_REGEX_NS}/dynamic_obstacle_1",
            "{ENV_REGEX_NS}/dynamic_obstacle_2",
        ],
    )

    contact_forces_roof = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bcr_bot/bcr_bot/roof_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Warehouse",
            "{ENV_REGEX_NS}/wall_south_proxy",
            "{ENV_REGEX_NS}/wall_north_proxy",
            "{ENV_REGEX_NS}/wall_west_proxy",
            "{ENV_REGEX_NS}/wall_east_proxy",
            "{ENV_REGEX_NS}/rack_1_proxy",
            "{ENV_REGEX_NS}/rack_2_proxy",
            "{ENV_REGEX_NS}/rack_3_proxy",
            "{ENV_REGEX_NS}/rack_4_proxy",
            "{ENV_REGEX_NS}/rack_5_proxy",
            "{ENV_REGEX_NS}/rack_6_proxy",
            "{ENV_REGEX_NS}/rack_7_proxy",
            "{ENV_REGEX_NS}/static_barrel",
            "{ENV_REGEX_NS}/static_cone",
            "{ENV_REGEX_NS}/dynamic_obstacle_1",
            "{ENV_REGEX_NS}/dynamic_obstacle_2",
        ],
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Direct wheel velocity control."""
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

        lidar = ObsTerm(
            func=mdp.lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar")}
        )
        
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        goal_position_robot_frame = ObsTerm(
            func=mdp.goal_position_robot_frame,
            params={"command_name": "goal_pose"}
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"command_name": "goal_pose"}
        )
        
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Goal spawns in NORTH zone (Y: 26 to 29) - past the racks."""
    goal_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-23.0, 3.0),   
            pos_y=(26.0, 29.0),   
            heading=(-3.14159, 3.14159),
        ),
    )


@configclass
class EventCfg:
    """Robot spawns in SOUTH zone, obstacles positioned for testing."""
    
    reset_robot_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-24.0, 4.0),     
                "y": (-20.0, -20.0),   
                "yaw": (0.0, 3.14159), 
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

    # Static obstacles 
    reset_static_barrel = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("static_barrel"),
        },
    )

    reset_static_cone = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("static_cone"),
        },
    )


    # Dynamic obstacles
    reset_dynamic_obstacle_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-3.0, 3.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_1"),
        },
    )

    push_dynamic_obstacle_1 = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "velocity_range": {"y": (0.5, 0.5)},  
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_1"),
        },
    )

    reset_dynamic_obstacle_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-3.0, 3.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_2"),
        },
    )

    push_dynamic_obstacle_2 = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "velocity_range": {"y": (-0.5, -0.5)},  
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_2"),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration - same as warehouse grid training."""
    
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
        params={
            "force_threshold": 0.5,
            "startup_grace_steps": 45,
            "chassis_sensor_cfg": SceneEntityCfg("contact_forces_chassis"),
            "roof_sensor_cfg": SceneEntityCfg("contact_forces_roof"),
        }
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
    """Termination conditions."""
    
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )

    goal_reached = DoneTerm(
        func=mdp.is_goal_reached,
        params={"command_name": "goal_pose", "threshold": 0.6},
        time_out=False
    )

    collision = DoneTerm(
        func=mdp.is_collided,
        params={
            "force_threshold": 0.5,
            "startup_grace_steps": 45,
            "chassis_sensor_cfg": SceneEntityCfg("contact_forces_chassis"),
            "roof_sensor_cfg": SceneEntityCfg("contact_forces_roof"),
        },
        time_out=False
    )


##
# Environment configuration
##

@configclass
class YhbotWarehouseTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for YHBot warehouse TEST environment (1 env only)."""

    scene: YhbotWarehouseTestSceneCfg = YhbotWarehouseTestSceneCfg(num_envs=1, env_spacing=70.0)

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 70  

        self.viewer.eye = (0.0, -3.7, 2.0)      
        self.viewer.lookat = (0.0, 3.0, 0.5)    
        self.viewer.origin_type = "asset_root"  
        self.viewer.asset_name = "robot"        

        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation