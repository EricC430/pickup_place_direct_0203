# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.utils import configclass
from .pickup_place_direct_0203_vision_env_cfg import PickupPlaceDirect0203VisionEnvCfg
from isaaclab.sensors import ContactSensorCfg

@configclass
class PickupPlaceVisionAsym0313EnvCfg(PickupPlaceDirect0203VisionEnvCfg):
    """
    Configuration for Asymmetric Vision Environment with Multi-Frame CNN Features and Point Cloud. (0313 Version)
    """
    
    # ========== CAMERA INTRINSIC PARAMETERS FOR 3D PROJECTION ==========
    # Dabai DCW 完整規格:
    #   Depth:  640×400 @30fps, FOV H79° V62° (without D2C) / H79° V55° (with D2C)
    #   Color:  1920×1080 @30fps, FOV H86° V55° (16:9) / H64° V55° (4:3)
    #   深度工作範圍: 0.2m ~ 2.5m, 精度 6mm@1m, Baseline 40mm
    #
    # Isaac Sim PinholeCameraCfg 使用方形像素 (fx = fy)
    # VFOV 由 HFOV + 長寬比決定: VFOV_sim ≈ 54.5° (真實深度 62°, 真實彩色 55°)
    #
    # 像素焦距 (pinhole, 方形像素): fx = fy = W / (2 * tan(HFOV / 2))
    # 此為「模擬相機」的內參，部署實機時需替換為真實校正參數:
    #   真實 Depth (640×400): fx≈388.3, fy≈323.9 (非方形像素)
    #   真實 Color (1920×1080): fx≈1136.0, fy≈1036.9 (非方形像素)
    _h_fov_deg: float = 79.0   # 水平視場角 (degrees) - from Dabai DCW spec
    _v_fov_deg_real: float = 62.0   # 真實垂直視場角 (degrees) - 僅供參考
    
    camera_image_height: int = 80                  # Low-res camera height
    camera_image_width: int = 128                  # Low-res camera width
    
    # PinholeCameraCfg 方形像素: fx = fy
    _pixel_focal_length: float = camera_image_width / (2 * math.tan(math.radians(_h_fov_deg / 2)))  # ≈ 77.66 px
    camera_focal_length_x: float = _pixel_focal_length   # fx ≈ 77.66 px (low-res)
    camera_focal_length_y: float = _pixel_focal_length   # fy = fx (方形像素)
    camera_principal_point_x: float = camera_image_width / 2.0    # cx = 64.0
    camera_principal_point_y: float = camera_image_height / 2.0   # cy = 40.0
    
    # High-Res 相機內參 (同 FOV，解析度 5 倍)
    camera_high_image_height: int = 400
    camera_high_image_width: int = 640
    _pixel_focal_length_high: float = camera_high_image_width / (2 * math.tan(math.radians(_h_fov_deg / 2)))  # ≈ 388.3 px
    camera_high_focal_length_x: float = _pixel_focal_length_high
    camera_high_focal_length_y: float = _pixel_focal_length_high
    camera_high_principal_point_x: float = camera_high_image_width / 2.0   # cx = 320.0
    camera_high_principal_point_y: float = camera_high_image_height / 2.0  # cy = 200.0
    
    # ========== OBSERVATION SPACE DIMENSIONS ==========
    # Dual-Camera Vision Architecture:
    # - Low-Res: Stack past 4 frames of CNN features (128 dims per frame)
    #   * Total: 4 frames × 128 dims = 512 dims per step
    #   * Captures temporal dynamics, motion, trends
    # - High-Res: Single frame per episode (64 dims)
    #   * Provides static scene context (unchanged throughout episode)
    #   * Total vision: 512 + 64 = 576 dims
    
    # Policy (Actor) observation structure:
    # JPos(6) + JVel(6) + JErr(6) + Last4Actions(24) + VisionLow(512) + PointNet(512) + VisionHigh(64)
    # = 6 + 6 + 6 + 24 + 512 + 512 + 64 = 1130 dims
    observation_space = 1130
    
    # ========== SENSORS ==========
    # 左夾爪感測器
    left_finger_force: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_out_link",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # 右夾爪感測器
    right_finger_force: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_out_link",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )
    
    # Critic observation structure:
    # JPos(6) + JVel(6) + Last_4_Actions(24) + Obj_Pos(3) + Obj_BBox(24) + BasketPos(3) + ContactFriction(7)
    # = 6 + 6 + 24 + 3 + 24 + 3 + 7 = 73 dims
    critic_observation_space = 73
    
    # ========== HOME POSE CONFIGURATION (STRATEGY A) ==========
    # 設定固定起始姿態，確保 Step 0 能拍攝到全局。
    # 將 min 與 max 設定為相同數值，即可達到「固定角度」的效果，同時方便您之後微調。
    randomize_arm_init = True
    arm_init_offset_range = {
        "joint1": (0.0, 0.0),                           # Base rotation (yaw)
        "joint2": (-30 * math.pi/180, -30 * math.pi/180), # Shoulder (pitch) -30 deg
        "joint3": (45 * math.pi/180, 45 * math.pi/180),  # Elbow (pitch) 60 deg
        "joint4": (119 * math.pi/180, 119 * math.pi/180), # Wrist (pitch) 119 deg
        "joint5": (0.0, 0.0),                           # Wrist (roll)
    }
    
    def __post_init__(self):
        super().__post_init__()
        # Restore original viewer settings from 0203
        self.viewer.eye = [3.0, 3.0, 2.5]
        # Enable contact sensors for the robot fingertips
        self.robot_cfg.spawn.activate_contact_sensors = True
