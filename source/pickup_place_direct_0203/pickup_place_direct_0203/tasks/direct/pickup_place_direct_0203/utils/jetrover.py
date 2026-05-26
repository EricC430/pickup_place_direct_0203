import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

JETROVER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/test_isaaclab/Jetrover/jetrover_isaac_sim/jetrover_real_servo_heavier_gripper_camera_point.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_depenetration_velocity=3.0,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
        ),
        # [0410 ANTI-PENETRATION] Strategy 1: Per-collider contact/rest offsets for gripper
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.61086472, # 0.52359878, # 0.0, # 0.61086472,
            "joint3": 0.7853975, # 0.78539816, # 0.0, # 0.7853975,
            "joint4": 0.95993027, # 2.07694181, # 0.0, # 0.95993027,
            "joint5": 0.0,
            "r_joint": 1.569993,
            # "joint2": (-30 * math.pi/180, -30 * math.pi/180), # Shoulder (pitch) -30 deg
            # "joint3": (45 * math.pi/180, 45 * math.pi/180),   # Elbow (pitch) 45 deg
            # "joint4": (119 * math.pi/180, 119 * math.pi/180),   
        },
    ),
    # default configs
    actuators={
        "base_servo": ImplicitActuatorCfg(joint_names_expr=["joint1"], effort_limit_sim=1.37, velocity_limit_sim=5.82, stiffness=40.0, damping=4.0,),
        "arm_servos": ImplicitActuatorCfg(joint_names_expr=["joint2", "joint3", "joint4"], effort_limit_sim=2.40, velocity_limit_sim=5.82, stiffness=60.0, damping=6.0,),
        "wrist_roll": ImplicitActuatorCfg(joint_names_expr=["joint5"], effort_limit_sim=0.82, velocity_limit_sim=5.82, stiffness=20.0, damping=2.0,),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["r_joint"], effort_limit_sim=1.44, velocity_limit_sim=6.54, stiffness=80.0, damping=8.0,),
    },
    # # gain tuner (corrected key naming to avoid duplicate key overwrites)
    # actuators={
    #     "base_servo": ImplicitActuatorCfg(joint_names_expr=["joint1"], effort_limit_sim=1.37, velocity_limit_sim=3.3, stiffness=80.0, damping=4.0,),
    #     "arm_servos": ImplicitActuatorCfg(joint_names_expr=["joint2", "joint3"], effort_limit_sim=2.40, velocity_limit_sim=2.48, stiffness=120.0, damping=60.0,),
    #     "wrist_pitch": ImplicitActuatorCfg(joint_names_expr=["joint4"], effort_limit_sim=2.40, velocity_limit_sim=3.1, stiffness=60.0, damping=10.0,),
    #     "wrist_roll": ImplicitActuatorCfg(joint_names_expr=["joint5"], effort_limit_sim=0.82, velocity_limit_sim=4.1, stiffness=60.0, damping=4.0,),
    #     "gripper": ImplicitActuatorCfg(joint_names_expr=["r_joint"], effort_limit_sim=1.44, velocity_limit_sim=2.31, stiffness=60.0, damping=4.0,),
    # },
)
