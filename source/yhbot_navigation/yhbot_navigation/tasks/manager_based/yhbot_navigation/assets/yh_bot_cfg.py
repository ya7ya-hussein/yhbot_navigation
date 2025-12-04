import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


YHBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/yahya/Documents/fyp/yhbot_navigation/source/yhbot_navigation/yhbot_navigation/tasks/manager_based/yhbot_navigation/assets/yh-bot.usd",
        activate_contact_sensors=True,  
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1.5,  
            max_angular_velocity=2.0,  
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,  
            sleep_threshold=0.005,  
            stabilization_threshold=0.001,
        ),
        # Configure collision properties for the robot 
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.29),  
        rot=(1.0, 0.0, 0.0, 0.0),  
        # Initial joint positions 
        joint_pos={
            "middle_left_wheel_joint": 0.0,
            "middle_right_wheel_joint": 0.0,
        },
        # Initial joint velocities 
        joint_vel={
            "middle_left_wheel_joint": 0.0,
            "middle_right_wheel_joint": 0.0,
        },
    ),
    actuators={
        # Differential drive wheels 
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["middle_.*_wheel_joint"],  
            effort_limit=20000.0,  
            velocity_limit=5.0,  
            stiffness=0.0,  
            damping=10000000.0,  
        ),
    },
)