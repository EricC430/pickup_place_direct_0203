# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from .utils.jetrover import JETROVER_CFG

# ObjectFolder 物體ID列表 # [39, 41]
# SELECTED_OBJECT_IDS = [22, 25, 26, 27, 31, 39, 40, 41, 62, 68, 70, 93, 95, 96, 21] # 21 bottle 29 wood 40 28
SELECTED_OBJECT_IDS = [22, 25, 26, 27, 28, 31, 39, 40, 41, 62, 68, 70, 93, 95, 96]

# 各物體摩擦係數範圍 (static_min, static_max, dynamic_min, dynamic_max)
# 數值基於物體材質的真實物理特性
OBJECT_FRICTION_MAP = {
    # 原有 5 種
    39: (0.5, 0.8, 0.4, 0.6),  # 塑膠小杯子
    22: (0.4, 0.7, 0.3, 0.5),  # 塑膠盆子 (PP/PE)
    95: (0.7, 0.9, 0.5, 0.7),  # 玻璃瓶
    68: (0.5, 0.8, 0.4, 0.6),  # 塑膠罐
    25: (0.5, 0.7, 0.4, 0.6),  # 不鏽鋼叉子

    # 新增 10 種
    26: (0.5, 0.7, 0.4, 0.6),  # 不鏽鋼湯匙
    27: (0.5, 0.7, 0.4, 0.6),  # 不鏽鋼刀子
    28: (0.5, 0.7, 0.4, 0.6),  # 鐵叉子
    31: (0.4, 0.6, 0.3, 0.5),  # 麥克筆（塑膠外殼）
    40: (0.5, 0.8, 0.4, 0.6),  # 塑膠小杯子（同 39）
    41: (0.5, 0.8, 0.4, 0.6),  # 塑膠小杯子（同 39）
    62: (0.6, 0.8, 0.4, 0.6),  # 陶瓷盤子
    70: (0.5, 0.8, 0.4, 0.6),  # 塑膠玩具
    93: (0.5, 0.8, 0.4, 0.6),  # 塑膠瓶
    96: (0.7, 0.9, 0.5, 0.7),  # 玻璃調味罐（同 95）
}
# Create a smaller frame marker for better visualization (ISSUE FIX #4)
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)


@configclass
class PickupPlaceDirect0510EnvCfg(DirectRLEnvCfg):
    """Direct workflow configuration for pickup-place task with JetRover (Task-Space Delta IK)."""

    # Wrist Camera Config (Optional, used for trajectory collection)
    wrist_camera_cfg: TiledCameraCfg | None = None

    # 環境基本設定
    decimation = 10
    episode_length_s = 5.0
    
    # Debug visualization settings
    # Debug visualization settings
    debug_vis = False
    debug_vis_settings = {
        "target": True,
        "com": False,
        "bbox": False,
    }

    # action/observation space
    action_space = 7  # 6D Task Space (XYZ, RxRyRz) + 1 Gripper
    # observation_space breakdown (49 dim):
    # - relative_joint_pos: 6 (5 arm joints + 1 gripper)
    # - joint_vel: 6
    # - object_position: 3
    # - object_bbox_corners: 24
    # - target_poses: 3
    # - actions: 7
    use_46_dim_obs = False
    observation_space = 49 
    state_space = 0

    # simulation settings
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=decimation)
    sim.physx.gpu_found_lost_aggregate_pairs_capacity = 32 * 1024 * 1024
    sim.physx.gpu_collision_stack_size = 512 * 1024 * 1024
    sim.physx.gpu_max_rigid_patch_count = 8 * 1024 * 1024
    sim.physx.gpu_heap_capacity = 512 * 1024 * 1024
    sim.physx.gpu_temp_buffer_capacity = 128 * 1024 * 1024
    sim.physx.gpu_max_rigid_contact_count = 16 * 1024 * 1024
    sim.physx.bounce_threshold_velocity = 0.2
    sim.physx.friction_correlation_distance = 0.00625
    
    # LIFTING DETECTION FIX: Anti-penetration physics parameters
    # Prevent objects from penetrating gripper deeply

    sim.physx.rest_offset = 0.0001          # Small rest offset to maintain stable contact
    sim.physx.min_position_iteration_count = 32  # COMPROMISE: 12 is very stable but 2.5x faster than 32
    sim.physx.min_velocity_iteration_count = 4   # Sufficient for stability
    

    # robot
    robot_cfg: ArticulationCfg = JETROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True

    # scene
    # Note: Using replicate_physics=False for memory efficiency
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10, env_spacing=1.5, replicate_physics=False) # env_spacing=1.5

    # object: Config for the RigidObject wrapper
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  # Regex for all environments
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.14875, 0.0, 0.15], rot=[1, 0, 0, 0]),
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                UsdFileCfg(
                    # usd_path=f"/workspace/test_isaaclab/ObjectFolder_selected/{i}/{i}.usd", # A6000 container
                    usd_path=f"/workspace/test_isaaclab/ObjectFolder_selected/{i}/{i}.usd", # 4090 container
                    scale=(0.6, 0.6, 0.6),
                )
                for i in SELECTED_OBJECT_IDS
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=3.0,
                disable_gravity=False,
            ),
        ),
    )

    # end-effector frame transformer
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.08]),
            ),
        ],
    )

    # 左夾爪接觸感測器
    left_finger_force: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_out_link",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # 右夾爪接觸感測器
    right_finger_force: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_out_link",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # action configuration (6 DOF)
    action_scale = 1.0

    # reward scales
    rew_scale_reach = 0.2
    rew_scale_lift = 0.0
    rew_scale_lift_vel = 0.0
    rew_scale_close = 0.3  # 已調降
    rew_scale_goal = 0.0
    rew_scale_goal_fine = 0.0
    rew_scale_action = 0.0  # 被拿掉
    rew_scale_joint_vel = 0.0  # 被拿掉
    rew_scale_torques = 0.0  # 被拿掉

    # randomization settings
    randomize_object_mass = True
    object_mass_range = (0.05, 0.2)  # Physical mass constraints
    randomize_friction = True
    friction_range = (0.3, 0.8)
    observation_noise_scale = 1.0

    # termination settings
    # termination settings
    workspace_range = {"x": (-0.3, 0.45), "y": (-0.4, 0.4)}

    # command generation for target pose
    target_pos_range = {
        "x": (0.05, 0.25),
        "y": (-0.2, 0.2),
        "z": (0.3, 0.55),
    }

    # Action Configuration (Task-Space Delta Control)
    action_cfg = {
        "ik_method": "dls",        # Differential IK method
        "position_scale": 0.005,   # Max 0.5cm translation per step
        "rotation_scale": 0.05,    # Max rotation per step
        "gripper_scale": 1.0,
    }

    # Arm Initial Position Randomization Settings
    # Randomize arm joint offsets at episode reset to vary the viewing angle
    # while ensuring the camera can view the object near the center of the frame
    randomize_arm_init = False
    arm_init_offset_range = {
        "joint1": (-0.3, 0.3),      # Base rotation (yaw)
        "joint2": (0.3, 1.0),       # Shoulder (pitch)
        "joint3": (0.4, 1.2),       # Elbow (pitch) 
        "joint4": (0.5, 1.3),       # Wrist (pitch)
        "joint5": (-0.5, 0.5),      # Wrist (roll)
    }

    # Object Spawn Settings - Randomization
    # Enable random yaw rotation for objects at spawn time
    randomize_object_yaw = True
    object_yaw_range = (0.0, 6.28318)  # Full 360 degrees in radians

    # Success Criteria
    success_criteria = {
        "consecutive_success_threshold": 10,
        "initial_ignore_steps": 0,
        "min_height": 0.06,
        "min_grasp_quality": 0.20,
        "velocity_threshold": 0.20,
        "min_distance_to_ee": 0.01,
        "max_distance_to_ee": 0.15,
    }

    # Reward Settings
    reward_settings = {
        "reaching_std": 0.15,
        "lifting_min_height": 0.5,
        "close_threshold": 0.08,
        "goal_std": 0.4,
        "goal_min_height": 0.10,
        "goal_fine_std": 0.05,
        "goal_fine_min_height": 0.10,
        "object_is_reached_threshold": 0.08,
    }

    # Curriculum Settings
    curriculum_settings = {
        "lifting_object": {
            "target": 3.0, #0.4, #3.0
            "threshold": 0.6, 
            "metric": "reaching_success", 
            "dependency": None
        },
        "lifting_object_velocity": {
            "target": 0.8, #0.1, #0.8
            "threshold": 0.6, 
            "metric": "reaching_success", 
            "dependency": None
        },
        "close_reward": {
            "target": 0.3, # 已調降
            "threshold": 0.6, 
            "metric": "reaching_success", 
            "dependency": None
        },
        "object_goal_tracking": {
            "target": 4.0, #0.1, #8.0
            "threshold": 0.5, #0.6,#0.4, 
            "metric": "lifting_success", 
            "dependency": "lifting_object",#None#"lifting_object",
            "increment": 0.25,  # Gradual warm-up
            "interval": 50,    # Check success every 25 steps (fast check)
            "increment_interval": 100, # But only increase weight every 100 steps (slow warm-up)
        },
        "object_goal_tracking_fine_grained": {
            "target": 1.0, #0.2, #2.0
            "threshold": 0.5, 
            "metric": "lifting_success", 
            "dependency": "lifting_object",
            "increment": 0.025, 
            "interval": 50,
            "increment_interval": 100,
        },
        # "action_rate" curriculum removed
        # "joint_vel" curriculum removed
    }

    # Manual Curriculum Weight Initialization (for resuming training)
    # If a term is present here, it will initialize with this value instead of 0.0
    # 直接從 reaching > 60% 解鎖的階段開始訓練 (跳過純 reaching 階段)
    curriculum_starting_weights = {
        "lifting_object": 3.0,
        "lifting_object_velocity": 0.8,
        "close_reward": 0.3,
        "object_goal_tracking": 4.0,
        "object_goal_tracking_fine_grained": 1.0
    }

    # Object Configuration
    object_scale = [0.6, 0.6, 0.6]

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        # 拉遠相機視角 (Pull camera further back)
        self.viewer.eye = [3.0, 3.0, 2.0]
        # 確保相機對準原點 (Ensure camera looks at origin)
        self.viewer.lookat = [0.0, 0.0, 0.0]