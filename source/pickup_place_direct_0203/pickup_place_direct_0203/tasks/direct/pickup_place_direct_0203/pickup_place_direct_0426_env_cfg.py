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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.utils import configclass
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from .jetrover import JETROVER_CFG

# ObjectFolder 物體ID列表 # [39, 41]
SELECTED_OBJECT_IDS = [25,26,27,68,70]#[22, 25, 26, 27, 31, 39, 41, 62, 68, 70, 93, 95, 96]# #[39, 41, 68, 25] #[62, 68] # [22, 25, 26, 27, 31, 39, 41, 62, 68, 70, 93, 95, 96, 21] # 21 bottle 29 wood 40 28

# Create a smaller frame marker for better visualization (ISSUE FIX #4)
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)


@configclass
class PickupPlaceDirect0426EnvCfg(DirectRLEnvCfg):
    """Direct workflow configuration for pickup-place task with JetRover.
    
    [0426 DELTA ACTION] This version uses:
    - Delta action formulation (incremental position control instead of absolute)
    - Actuator low-pass filter (EMA) to simulate real servo response latency
    - Torque penalty and Jerk penalty (action smoothness) for Sim2Real safety
    """

    # 環境基本設定
    decimation = 2
    episode_length_s = 5.0
    
    # Debug visualization settings
    # Debug visualization settings
    debug_vis = False
    debug_vis_settings = {
        "target": False,
        "com": False,
        "bbox": False,
    }

    # action/observation space
    action_space = 6  # 5 arm joints (joint1-5) + 1 gripper (r_joint)
    # observation_space breakdown (48 dim):
    # - joint_pos: 6 (arm joints + 1 gripper) ; if 46 dim, 5 (arm joints only)
    # - joint_vel: 6 (arm joints + 1 gripper) ; if 46 dim, 5 (arm joints only)
    # - object_position: 3 (in robot frame)
    # - object_bbox_corners: 24 (8 corners * 3 coords)
    # - target_poses: 3 (target position)
    # - actions: 6 (previous action)
    use_46_dim_obs = False  # Set to True to use the previously incorrect 46-dim version for testing old weights
    disable_anti_penetration = False  # Diagnostic flag to disable CCD and solve_articulation_contact_last
    disable_push_penalty = False  # Diagnostic flag to disable horizontal displacement reset
    disable_physics_reset = False  # Diagnostic flag to disable joint explosion reset
    disable_drop_reset = False  # Diagnostic flag to disable object drop reset
    observation_space = 48 
    state_space = 0

    # simulation settings
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=decimation)
    # [0419 2048 ENV UPDATE] Optimized capacities for 2k env simulation
    sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26 # 67M (Optimized from 2**27)
    sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024    # 256 MB (Optimized from 512 MB)
    sim.physx.gpu_max_rigid_patch_count = 8 * 1024 * 1024     # 8M
    sim.physx.gpu_temp_buffer_capacity = 64 * 1024 * 1024     # 64 MB (Optimized from 128 MB)
    sim.physx.gpu_heap_capacity = 256 * 1024 * 1024           # 256 MB (Optimized from 512 MB)
    sim.physx.gpu_max_rigid_contact_count = 16 * 1024 * 1024  # 16M
    sim.physx.bounce_threshold_velocity = 0.01  # [BEST PRACTICE] Low value (0.01) makes grasping more stable
    sim.physx.friction_correlation_distance = 0.00625
    
    # [0410 ANTI-PENETRATION] Strategy 2: Enable CCD to prevent tunneling through thin objects
    sim.physx.enable_ccd = True
    # [0410 ANTI-PENETRATION] Strategy 5: Solve articulation contacts last for better grasping stability
    # (Isaac Sim 5.1+ only) Prioritizes contact constraint resolution, improving grip quality
    sim.physx.solve_articulation_contact_last = True
    
    sim.physx.min_position_iteration_count = 12  # COMPROMISE: 12 is very stable but 2.5x faster than 32
    sim.physx.min_velocity_iteration_count = 2   # Sufficient for stability

    # robot
    robot_cfg: ArticulationCfg = JETROVER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    # Note: Using replicate_physics=False for memory efficiency
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=1.5, replicate_physics=False) # env_spacing=1.5 5.0

    # object: Config for the RigidObject wrapper
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  # Regex for all environments
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.14875, 0.0, 0.05], rot=[1, 0, 0, 0]),
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                UsdFileCfg(
                    usd_path=f"/workspace/test_isaaclab/ObjectFolder_selected/{i}/{i}.usd", # A6000 container
                    # usd_path=f"/root/ObjectFolder/{i}/{i}.usd", # 4090 container
                    scale=(0.6, 0.6, 0.6),
                )
                for i in SELECTED_OBJECT_IDS
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=3.0,#0.5,#
                disable_gravity=False,
            ),
            # [0410 ANTI-PENETRATION] Strategy 1: Per-collider contact/rest offsets
            # contact_offset=0.005 creates a 5mm detection buffer before actual geometric contact
            # This gives the solver more time to generate response impulses for thin objects
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.005,#0.02,#
                rest_offset=0.0,
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

    # action configuration (6 DOF)
    action_scale = 1.0

    # [0426 DELTA ACTION] Delta action parameters
    # Instead of absolute position targets, NN output is treated as incremental delta
    # target = current_pos + action * max_delta
    use_delta_actions = True
    max_delta_arm = 0.1       # Max arm movement per step (rad). At 50Hz, this = ~5.7 deg/step, ~286 deg/s max
    max_delta_gripper = 0.15  # Max gripper movement per step (rad). Slightly larger for responsive grasping

    # [0426 ACTUATOR FILTER] Low-pass filter for simulating servo response latency
    # target_filtered = alpha * target_new + (1 - alpha) * target_prev
    # alpha=1.0 = no filter, alpha=0.5 = heavy smoothing
    action_filter_alpha = 0.8

    # reward scales
    rew_scale_reach = 0.8 #1.2 #0.25#0.005#0.2
    rew_scale_lift = 0.0 #3.0 # 2.5
    rew_scale_lift_vel = 0.0 #0.3
    rew_scale_close = 0.0 #1.0 #0.3 # [0403] 0.5→2.0: 4x increase to make grasp signal competitive with reaching
    rew_scale_goal = 0.0
    rew_scale_goal_fine = 0.0
    rew_scale_lift_bonus = 0.0 #0.7 #1.2 # Bonus for stable lifting success
    rew_scale_drop = 0.0 #-5.0 # Penalty for dropping the object after it was lifted
    rew_scale_action = 0.0 
    rew_scale_action_near_goal = 0.0 #-0.01  # Penalty for trembling when the goal is reached
    rew_scale_action_approach = 0.0 #-0.05
    rew_scale_joint_vel = 0.0  # Start at 0; curriculum will ramp up after reaching_success threshold

    # [0426 NEW PENALTIES] Torque and Smoothness penalties for Sim2Real safety
    rew_scale_joint_effort = 0.0    # Torque penalty: -λ * ||τ_estimated||. Curriculum controlled.
    rew_scale_action_smoothness = 0.0  # Jerk penalty: -λ * ||a_t - 2*a_{t-1} + a_{t-2}||. Curriculum controlled.
    
    # Joint Limit and Smoothness Settings
    # [0415 ADAPTIVE VEL PENALTY] Raised threshold so gentle motion is free;
    # only fast sweeping motions (>1.0 rad/s combined) are penalized.
    # Previously 0.55 rad/s was too tight and slowed exploration.
    joint_vel_threshold = 1.0  # rad/s, only velocities above this are penalized
    action_smoothness_approach_threshold = 0.06  # m, penalty applies when distance is below this

    # randomization settings
    randomize_object_mass = True
    object_mass_range = (0.1, 0.5)
    observation_noise_scale = 1.0

    # termination settings
    # Expanded bounds to avoid instant resets when objects spawn near 0.35
    workspace_range = {"x": (-0.3, 0.45), "y": (-0.35, 0.35)}  # {"x": (-0.05, 0.35), "y": (-0.3, 0.3)}
    
    # Curriculum Settings (Iteration-Based)
    num_steps_per_iteration = 32  # Match trainer's num_steps_per_env (Changed from 24 to 32)
    # Mapping: { iteration_number: { "weight_name": value, ... } }
    reward_iteration_curriculum = {
        # "150": {"close_reward": 0.001},
        # "150": {"lifting_object": 2.0, "lifting_object_velocity": 0.5},
    }
    

    # command generation for target pose
    target_pos_range = {
        "x": (0.05, 0.25),
        "y": (-0.2, 0.2),
        "z": (0.3, 0.55),
    }

    # Action Configuration
    # [0426 NOTE] When use_delta_actions=True, arm_scale/gripper_scale are IGNORED.
    # The delta is controlled by max_delta_arm/max_delta_gripper instead.
    # These are kept for backward compatibility with absolute action mode.
    action_cfg = {
        "arm_offsets": [0.0, 0.0, 0.0, 0.0, 0.0],
        "arm_scale": 2.09,
        "gripper_scale": 0.785,
        "gripper_offset": 0.785,
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
        "consecutive_success_threshold": 50,
        "initial_ignore_steps": 0,
        "min_height": 0.1, #0.06,
        "min_grasp_quality": 0.50,    # [0421 SCOOP FIX] 0.20 → 0.50: scoop strategy only closes ~20%, true grasp needs 50%+
        "velocity_threshold": 0.30,   # [0421 SCOOP FIX] 0.50 → 0.30: scooped objects have higher velocity jitter
        "min_distance_to_ee": 0.01,
        "max_distance_to_ee": 0.08,   # [0421 SCOOP FIX] 0.15 → 0.08: object must be truly in-hand, not just near body
    }

    # Reward Settings
    reward_settings = {
        "reaching_std": 0.05, # 0.08, # 0.015,  # 縮小 std 以迫使模型極度靠近 #0.15
        "lifting_min_height": 0.5, # 0.10, # 0.2 was too aggressive — 10cm provides gradient for initial lifts
        "close_threshold": 0.05, # 0.075 was too wide — 5cm ensures gripper physically contacts object
        "goal_std": 0.10,
        "goal_min_height": 0.10,
        "goal_fine_std": 0.05,
        "goal_fine_min_height": 0.10,
        "object_is_reached_threshold": 0.05,
    }

    # Curriculum Settings
    curriculum_settings = {
        "lifting_object": {
            "target": 12.0,
            "threshold": 0.8, 
            "metric": "reaching_success", 
            "dependency": None,
            "increment": 0.5,           # Gradual warm-up to prevent policy shock
            "interval": 50,
            "increment_interval": 100,
        },
        "lifting_object_velocity": {
            "target": 0.8,
            "threshold": 0.8, 
            "metric": "reaching_success", 
            "dependency": None,
            "increment": 0.05,
            "interval": 50,
            "increment_interval": 100,
        },
        "lifting_bonus": {
            "target": 5.0,
            "threshold": 0.8, 
            "metric": "reaching_success", 
            "dependency": None,
            "increment": 0.25,
            "interval": 50,
            "increment_interval": 100,
        },
        # "close_reward": {
        #     "target": 0.025, #0.005, #0.001,
        #     "threshold": 0.6, 
        #     "metric": "reaching_success", 
        #     "dependency": None
        # },
        "object_goal_tracking": {
            "target": 15.0, # [BEST PRACTICE] Increased from 5.0 to match official high-weight goal rewards
            "threshold": 0.5, 
            "metric": "episode_lifting_success", 
            "dependency": "lifting_object",
            "increment": 0.1,#0.05,  # Gradual warm-up
            "interval": 50,    # Check success every 25 steps (fast check)
            "increment_interval": 100, # But only increase weight every 100 steps (slow warm-up)
        },
        "object_goal_tracking_fine_grained": {
            "target": 5.0,  # Increased from 2.0
            "threshold": 0.5, 
            "metric": "episode_lifting_success", 
            "dependency": "lifting_object",
            "increment": 0.05, #0.025, 
            "interval": 50,
            "increment_interval": 100,
        },
        "action_rate": {
            "target": -0.05,  # [BEST PRACTICE] Increased from -0.001 to align with official -0.1 magnitude
            "threshold": 0.5, 
            "metric": "object_goal_tracking_success", 
            "dependency": "object_goal_tracking",
            "increment": 0.0001, # Slowly increase penalty magnitude
            "interval": 50,
            "increment_interval": 100,
        },
        "action_rate_approach": {
            "target": -0.005,  # Stronger penalty when close
            "threshold": 0.5, 
            "metric": "reaching_success", # Start penalizing jitter once it learns to reach
            "dependency": None,
            "increment": 0.0005,
            "interval": 20,
            "increment_interval": 100,
        },
        "action_rate_near_goal": {
            "target": -0.005,  # Stronger penalty when close
            "threshold": 0.5, 
            "metric": "object_goal_tracking_success", # Start penalizing jitter once it learns to reach
            "dependency": "object_goal_tracking",
            "increment": 0.0005,
            "interval": 20,
            "increment_interval": 100,
        },
        "drop_penalty": {
            "target": -5.0,   # Severe red-line penalty
            "threshold": 0.5, 
            "metric": "episode_lifting_success", # Only punish drops once lifting is learned
            "dependency": "lifting_object",
            "interval": 50,
        },
        # [0415 ADAPTIVE VEL PENALTY] joint_vel 懲罰隨 reaching 成功率動態提升。
        # 設計原則：
        #   - 觸發門檻：reaching_success >= 0.5
        #     (機器人已学会接近物体，现在才开始限速，避免过早让探索陷死)
        #   - 目標值：-0.005，足以讓手臂在接觸物體前降速，但不至於讓 Agent 停止移動
        #   - increment: 每次 +0.0002 (遞減方向)，60 次 increment → 達 -0.012
        #     讓懲罰緩慢爬升，使 Agent 有時間同步調整政策
        #   - increment_interval: 200 步一次，比 lifting 的 100 步慢，
        #     給 Agent 更多緩衝時間習慣懲罰再繼續加強
        "joint_vel": {
            "target": -0.1,  # [BEST PRACTICE] Increased from -0.005 to align with official -0.1 magnitude
            "threshold": 0.50,
            "metric": "reaching_success",
            "dependency": None,         # 不依賴任何前置課程，只要 reaching 到位就觸發
            "increment": 0.0002,         # 每次增加 0.0002 (朝 -0.005 方向)
            "interval": 50,              # 每 50 步檢查一次 reaching_success
            "increment_interval": 200,   # 每 200 步才真正增加一次懲罰
        },
        # [0426 NEW] Torque penalty: penalize estimated PD torque magnitude
        # Encourages the NN to output targets close to current position (low PD error)
        "joint_effort": {
            "target": -0.001,
            "threshold": 0.3,           # Start early — even before good reaching
            "metric": "reaching_success",
            "dependency": None,
            "increment": 0.0001,
            "interval": 50,
            "increment_interval": 200,
        },
        # [0426 NEW] Action smoothness (Jerk) penalty: penalize second derivative of actions
        # Forces smooth acceleration/deceleration curves instead of jerky motion
        "action_smoothness": {
            "target": -0.002,
            "threshold": 0.3,           # Start early
            "metric": "reaching_success",
            "dependency": None,
            "increment": 0.0002,
            "interval": 50,
            "increment_interval": 200,
        },
    }

    # Manual Curriculum Weight Initialization (for resuming training)
    # If a term is present here, it will initialize with this value instead of 0.0
    curriculum_starting_weights = {
        # "lifting_object": 3.0,
        # "lifting_object_velocity": 0.3,
        # "lifting_bonus": 0.7,
        # "close_reward": 1.0,  # [0403] Match rew_scale_close
        # "object_goal_tracking": 7.0,
        # "object_goal_tracking_fine_grained": 2.0,
        # "action_rate": -0.001,
        # "action_rate_approach": -0.05,  # [0403] Match rew_scale_action_approach
        # [0415] joint_vel 現在由 curriculum 管理，從 0.0 開始遞增懲罰，不再在此指定固定初始值
        # "joint_vel": -0.001,
    }

    # Object Configuration
    object_scale = [0.6, 0.6, 0.6]

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.viewer.eye = [5.0, 5.0, 3.0]
        if self.use_46_dim_obs:
            self.observation_space = 46
        
        # [0422 DIAGNOSTIC] Disable anti-penetration features if requested
        if self.disable_anti_penetration:
            self.sim.physx.enable_ccd = False
            self.sim.physx.solve_articulation_contact_last = False