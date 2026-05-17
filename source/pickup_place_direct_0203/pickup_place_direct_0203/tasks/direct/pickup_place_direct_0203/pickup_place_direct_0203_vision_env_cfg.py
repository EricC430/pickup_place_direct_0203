# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from .pickup_place_direct_0203_env_cfg import PickupPlaceDirect0203EnvCfg

from isaaclab.utils import configclass
@configclass
@configclass
class PickupPlaceDirect0203VisionEnvCfg(PickupPlaceDirect0203EnvCfg):
    """
    Direct workflow with vision (RGB + Depth) observations using dual cameras.
    
    Key improvements over direct state-based:
    - Uses **dual camera system**: Low-res for RL (continuous) + High-res for context (on-demand)
    - Observation space: 46 (state) + 128 (CNN features) = 174
    - Reduces massive image data to compact feature vectors
    - Ensures compatibility with PPO MLP policy network
    
    Camera Hardware: Dabai DCW (Orbbec binocular structured light, ASIC)
    - Depth:  640×400 @30fps, FOV H79° V62° (without D2C) / H79° V55° (with D2C)
    - Color:  1920×1080 @30fps, FOV H86° V55° (16:9) / H64° V55° (4:3)
    - 工作範圍: 0.2m ~ 2.5m, 精度 6mm@1m, Baseline 40mm
    - 延遲: 30~45ms
    
    Simulation Camera Notes:
    - 使用 PinholeCameraCfg (physical focal_length mm + horizontal_aperture mm)
      → HFOV = 2*atan(aperture / (2*focal_length)) = 79° (匹配深度相機)
    - PinholeCameraCfg 假設方形像素 → VFOV 由長寬比決定 ≈ 54.5° (真實 62°)
    - clipping_range 控制渲染裁切面，影響 RGB+Depth 兩者 → 設為寬範圍
      深度工作範圍 (0.2~2.5m) 在軟體層 _process_vision_data 中 torch.clamp 處理
    - RGB 解析度 (1920×1080) 遠高於 Depth (640×400)，訓練以 Depth 為主軸
      因 CNN 計算量隨解析度平方增長，且更高 RGB 無法提供更多深度資訊
    
    Camera System:
    1. Low-Res (128×80, 每步): 640×400 的 1/5 等比，RL 連續回饋
    2. High-Res (640×400, 手動): 原生深度解析度，episode 開始時上下文擷取
    """

    # ========== PHYSICAL PARAMETERS (Shared) ==========
    # Dabai DCW depth camera: HFOV = 79°, VFOV = 62°
    # 注意: PinholeCameraCfg 的 focal_length 是物理焦距 (mm)，用於控制 Isaac Sim 渲染 FOV
    #       與 asym_env_cfg 中用於 2D→3D 投影的像素焦距 (px) 不同
    target_hfov: float = 79.0                              # Horizontal field of view (degrees)
    aperture: float = 20.955                               # Sensor aperture (mm)
    # Computed physical focal length (mm): f = aperture / (2 * tan(FOV/2))
    focal_length: float = aperture / (2 * math.tan(math.radians(target_hfov / 2)))

    # ========== CNN FEATURE EXTRACTION ==========
    vision_encoder_type: str = "resnet"                    # "simple" or "resnet"
    vision_encoder_feature_dim: int = 128                  # Output features from CNN
    vision_encoder_device: str = "cuda:0"                  # Device for CNN processing
    
    # Updated observation space: state (48) + Low-Res CNN (128) + High-Res Context (64) = 240
    observation_space = 48 + 128 + 64
    
    # ========== DUAL CAMERA CONFIGURATION ==========
    
    # 1. Low-Resolution Camera (Real-time for RL Agent)
    # Purpose: Continuous feedback for immediate control decisions
    # Resolution: 128×80 (native 640×400 的 1/5 等比縮放，保持 16:10 比例)
    # NOTE: MUST have identical rotation and convention as camera_high to view same scene!
    camera_low_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/depth_cam_link/camera_mount_marker/Camera_Low",
        update_period=0.0,                                 # Sync with sim step
        height=80,
        width=128,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=None,                             # Will be set dynamically below
            horizontal_aperture=20.955,
            clipping_range=(0.01, 3.8),                    # 降低遠端裁切面，避免看到隔壁環境
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),                   # ROS optical frame convention
            convention="ros",
        ),
    )
    # Set computed focal length after initialization
    camera_low_cfg.spawn.focal_length = focal_length
    
    # Step index (relative to episode start) to trigger high-res capture
    high_res_capture_step: int = 0
    
    # 2. High-Resolution Camera (Context extraction, on-demand)
    # Purpose: Extract detailed static features at episode start
    # Resolution: 640×400 (匹配 Dabai DCW 原生深度解析度)
    camera_high_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/depth_cam_link/camera_mount_marker/Camera_High",
        update_period=0.0,                                 # Manual trigger (not continuous)
        height=400,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=None,                             # Will be set dynamically below
            horizontal_aperture=20.955,
            clipping_range=(0.01, 3.8),                    # 降低遠端裁切面，避免看到隔壁環境
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),                   # ROS optical frame convention
            convention="ros",
        ),
    )
    # Set computed focal length after initialization
    camera_high_cfg.spawn.focal_length = focal_length
    
    # ========== REWARD SCALES ==========
    rew_scale_vision_focus: float = 0.1                    # Reward for keeping object in camera view
    
    # ========== DEBUG VISUALIZATION ==========
    debug_vision_snapshots = False                          # Enable snapshots (but limited to specific envs)
    debug_vision_snapshot_interval = 3000                   # Save every N steps (low-res camera)
    debug_vision_snapshot_high_res = False                  # Also save high-res snapshots
    debug_vision_snapshot_high_res_interval = 6000         # Min steps between high-res snapshots
    debug_vision_snapshot_envs: list[int] = [0]            # Limit to Env 0 ONLY (saves resources)
    debug_vision_snapshot_dir = "/workspace/test_isaaclab/isaaclab_vision_debug"  # Persistent directory for snapshots
    debug_vision_enable_yolo_visualization = False          # Enable YOLO detection visualization in snapshots