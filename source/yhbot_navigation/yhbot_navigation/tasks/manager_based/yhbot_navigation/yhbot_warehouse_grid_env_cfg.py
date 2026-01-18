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
from .assets.yh_bot_cfg import YHBOT_CFG


##
# Scene definition
##

@configclass
class YhbotWarehouseGridSceneCfg(InteractiveSceneCfg):
    """Configuration for YHBot warehouse grid scene with walls and racks."""

    # Ground plane 
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(5000.0, 5000.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # YHBot differential drive robot
    robot: ArticulationCfg = YHBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    
    # South wall
    wall_south: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(32.0, 0.5, 2.0),  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.5, -23.25, 1.0)),
    )
    
    # North wall
    wall_north: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(32.0, 0.5, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.5, 31.25, 1.0)),
    )
    
    # West wall 
    wall_west: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 55.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-26.25, 4.0, 1.0)),
    )
    
    # East wall 
    wall_east: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 55.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.25, 4.0, 1.0)),
    )
    
    # Rack 1
    rack_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_1",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-25.0, 17.0, 1.25)),
    )
    
    # Rack 2 
    rack_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_2",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-20.4, 17.0, 1.25)),
    )
    
    # Rack 3 
    rack_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_3",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-15.4, 17.0, 1.25)),
    )
    
    # Rack 4 
    rack_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_4",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.5, 17.0, 1.25)),
    )
    
    # Rack 5
    rack_5: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_5",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.5, 17.0, 1.25)),
    )
    
    # Rack 6 
    rack_6: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_6",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 17.0, 1.25)),
    )
    
    # Rack 7 
    rack_7: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rack_7",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 16.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.4, 17.0, 1.25)),
    )
    
    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.5,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.8, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    obstacle_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.4,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-10.0, -5.0, 0.51)),
    )

    dynamic_obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacle_1",
        spawn=sim_utils.CuboidCfg(
            size=(1.3, 1.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-18.8, 17.0, 0.41)),  
    )

    dynamic_obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dynamic_obstacle_2",
        spawn=sim_utils.CuboidCfg(
            size=(1.3, 1.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.2, 17.0, 0.41)),  
    )
    
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
            # Walls
            "{ENV_REGEX_NS}/wall_south",
            "{ENV_REGEX_NS}/wall_north",
            "{ENV_REGEX_NS}/wall_west",
            "{ENV_REGEX_NS}/wall_east",
            # Racks
            "{ENV_REGEX_NS}/rack_1",
            "{ENV_REGEX_NS}/rack_2",
            "{ENV_REGEX_NS}/rack_3",
            "{ENV_REGEX_NS}/rack_4",
            "{ENV_REGEX_NS}/rack_5",
            "{ENV_REGEX_NS}/rack_6",
            "{ENV_REGEX_NS}/rack_7",
            # Obstacles
            "{ENV_REGEX_NS}/obstacle_1",
            "{ENV_REGEX_NS}/obstacle_2",
            "{ENV_REGEX_NS}/obstacle_3",
            # Dynamic obstacles
            "{ENV_REGEX_NS}/dynamic_obstacle_1",
            "{ENV_REGEX_NS}/dynamic_obstacle_2",
        ],
        debug_vis=False,
    )
    
    contact_forces_chassis = ContactSensorCfg(
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
    """Robot spawns in SOUTH zone (Y: -20), obstacles in open south area."""
    
    # Robot spawns in south zone
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

    reset_obstacle_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_1"),
        },
    )

    reset_obstacle_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_3"),
        },
    )

    reset_obstacle_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("obstacle_3"),
        },
    )
    # Dynamic obstacle 1
    reset_dynamic_obstacle_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-5.0, 5.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_1"),
        },
    )
    
    # Dynamic obstacle 1
    push_dynamic_obstacle_1 = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 0.1),  
        params={
            "velocity_range": {"y": (0.5, 0.5)},  
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_1"),
        },
    )

    # Dynamic obstacle 2
    reset_dynamic_obstacle_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-5.0, 5.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("dynamic_obstacle_2"),
        },
    )
    
    # Dynamic obstacle 2 
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
    """Reward configuration - same as grid training."""
    
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
        params={"force_threshold": 0.5, "startup_grace_steps": 45, "chassis_sensor_cfg": SceneEntityCfg("contact_forces_chassis"), "roof_sensor_cfg": SceneEntityCfg("contact_forces_roof"),},
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
    """Termination conditions - same as grid training."""
    
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
        params={"force_threshold": 0.5, "startup_grace_steps": 45, "chassis_sensor_cfg": SceneEntityCfg("contact_forces_chassis"), "roof_sensor_cfg": SceneEntityCfg("contact_forces_roof"),},
        time_out=False
    )

##
# Environment configuration
##

@configclass
class YhbotWarehouseGridEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for YHBot warehouse grid environment."""

    # Scene settings
    scene: YhbotWarehouseGridSceneCfg = YhbotWarehouseGridSceneCfg(num_envs=4096, env_spacing=70.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 70  

        # Viewer settings 
        self.viewer.eye = (-10.0, -30.0, 25.0)
        self.viewer.lookat = (-10.0, 5.0, 0.0)

        # Simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


##
# Play configuration
##

@configclass
class YhbotWarehouseGridEnvCfg_PLAY(YhbotWarehouseGridEnvCfg):
    """Play configuration with single environment for testing."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Use single environment for play/testing
        self.scene.num_envs = 1
        self.scene.env_spacing = 70.0
        # Disable randomization for deterministic testing
        self.observations.policy.enable_corruption = False
