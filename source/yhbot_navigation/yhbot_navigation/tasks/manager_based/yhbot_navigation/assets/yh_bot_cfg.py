import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_YHBOT_USD_PATH = os.path.join(os.path.dirname(__file__), "yh_bot_flattened.usd")

YHBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_YHBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=5.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "middle_left_wheel_joint": 0.0,
            "middle_right_wheel_joint": 0.0,
        },
        joint_vel={
            "middle_left_wheel_joint": 0.0,
            "middle_right_wheel_joint": 0.0,
        },
    ),
    actuators={
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["middle_.*_wheel_joint"],
            stiffness=0.0,
            damping=10000000.0,
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            friction=0.0,
        ),
    },
)