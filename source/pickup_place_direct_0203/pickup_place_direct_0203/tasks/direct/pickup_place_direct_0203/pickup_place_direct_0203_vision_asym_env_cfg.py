# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from .pickup_place_direct_0203_vision_env_cfg import PickupPlaceDirect0203VisionEnvCfg


class PickupPlaceDirect0203VisionAsymEnvCfg(PickupPlaceDirect0203VisionEnvCfg):
    """
    Configuration for Asymmetric Vision Environment with Multi-Frame CNN Features and YOLO Detection.
    
    **OPTION A: Clean Observation Separation**
    
    This configuration implements a clear separation between:
    1. **Proprioceptive Information** (policy & critic): Joint positions and velocities
    2. **Privileged Information** (critic only): Object state, target, action history
    3. **Sensory Information** (both): Vision features and YOLO detection
    
    This eliminates information duplication and provides clear learning signals.
    
    Key Feature: Multi-Frame Vision Architecture
    - Low-Res Camera: Stacks 4 consecutive frames (4 × 128 dims = 512 dims)
      * Captures temporal dynamics and motion
      * Continuous feedback for RL agent
    
    - High-Res Camera: Single frame per episode (1 × 64 dims = 64 dims)
      * Provides static scene context
      * Remains the same throughout episode (no need for multi-frame)
      * Total vision: 512 + 64 = 576 dims
    
    Observation Spaces (Option A):
    - **Policy (Actor)**: 621 dimensions (vision-based only)
        - Proprioception (12): Joint Pos (6) + Joint Vel (6)
        - Target + Action (9): Goal position (3) + Previous action (6)
        - Vision Low-Res Multi-Frame (512): 4 stacked frames × 128 CNN features
        - Vision High-Res Single Frame (64): Static scene context
        - YOLO BBox Features (24): Detected object bounding box
        - Total: 12 + 9 + 512 + 64 + 24 = 621
        - Rationale: Forces agent to rely on vision for perception
    
    - **Critic**: 660 dimensions (with privileged environment access)
        - Proprioception (12): Joint Pos (6) + Joint Vel (6) - dynamics only
        - Vision Low-Res Multi-Frame (512): 4 stacked frames × 128 CNN features
        - Vision High-Res Single Frame (64): Static scene context
        - YOLO BBox Features (24): Detected object bounding box
        - Privileged GT Info (48): ObjPos(3) + ObjBBox(24) + Target(3) + Action(6) + Padding(12)
        - Total: 12 + 512 + 64 + 24 + 48 = 660
        - Rationale: Accurate value function with full state knowledge
    
    **Key Design Change from Previous Version:**
    - BEFORE: state_only(48) duplicated ObjPos + ObjBBox in gt_features → critic had redundant info
    - AFTER: state_only(12) is proprioceptive only, gt_features contains all non-proprioceptive state
             Clear separation, no duplication, cleaner learning signals
    
    YOLO Integration:
    - Real-time object detection from low-res camera using YOLOv8
    - Projected 2D -> 3D bounding box coordinates using depth map
    - Normalized bbox features for policy input
    - Actor learns from vision + YOLO, Critic has access to both YOLO and ground truth
    """
    
    # ========== YOLO DETECTION CONFIGURATION ==========
    # 性能指南 (RTX 4090):
    # - yolov8n: 最快, 記憶體最少, 精度中等 (~8MB, 10ms/幀)
    # - yolov8s: 平衡版, 推薦用於邊界條件 (~30MB, 15ms/幀)
    # - yolov8m: 推薦版本, 精度與速度兼衡 (~80MB, 25ms/幀) ⭐
    # - yolov8l: 高精度, 計算量大 (~200MB, 40ms/幀)
    # 
    # 環境數量建議:
    # - num_envs=20: 使用yolov8m, GPU利用率~50% (基準)
    # - num_envs=32: 使用yolov8m, GPU利用率~70% (推薦)
    # - num_envs=40: 使用yolov8s, GPU利用率~75% (激進)
    yolo_model_name: str = "yolov8m"          # YOLOv8 model size: "nano", "small", "medium", "large"
    yolo_conf_threshold: float = 0.5          # Detection confidence threshold (0.0-1.0)
    yolo_device: str = "cuda:0"               # Device for YOLO inference ("cuda:0" or "cpu")
    
    # YOLO 相機來源配置
    # "low"  → 使用 Low-Res (128×80)，快速但物件在低解析度下不易偵測
    # "high" → 使用 High-Res (640×400)，YOLO 偵測精度顯著提升
    yolo_camera_source: str = "high"           # "low" or "high"
    
    # ========== CAMERA INTRINSIC PARAMETERS FOR 3D PROJECTION ==========
    # Dabai DCW 完整規格:
    #   Depth:  640×400 @30fps, FOV H79° V62° (without D2C) / H79° V55° (with D2C)
    #   Color:  1920×1080 @30fps, FOV H86° V55° (16:9) / H64° V55° (4:3)
    #   工作範圍: 0.2m ~ 2.5m, 精度 6mm@1m, Baseline 40mm
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
    # - YOLO: 24 dims for detected object bounding box
    
    # Policy (Actor) observation structure (Option A):
    # [Proprio(12) | Target+Action(9) | Vision_Low4frames(512) | Vision_High1frame(64) | YOLO_BBox(24)]
    # = 12 + 9 + 512 + 64 + 24 = 621 dims
    observation_space = 621
    
    # Critic observation structure (Option A):
    # [Proprio(12) | Vision_Low4frames(512) | Vision_High1frame(64) | YOLO_BBox(24) | GT_Privileged(48)]
    # where GT_Privileged = ObjPos(3) + ObjBBox(24) + Target(3) + Action(6) + Padding(12)
    # = 12 + 512 + 64 + 24 + 48 = 660 dims
    critic_observation_space = 660
