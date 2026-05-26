# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Interactive Vision Debug Visualization Script
Demonstrates YOLO detection and 3D boundary box visualization in Isaac Sim.

Features:
- Single environment with interactive joint control via physics inspector
- Real-time YOLO object detection from camera
- 3D visualization of:
  - YOLO 2D bbox (projected as 2D box corners in 3D space)
  - YOLO 2D bbox + Depth -> 3D bbox (8 corner points)
  - Ground truth object 3D bbox (8 corner points)

Usage:
    python interactive_vision_debug.py
    
Requirements: Must be run within Isaac Lab environment
    export OMNI_KIT_ALLOW_ROOT=1
    export ISAACLAB_PATH=/workspace/isaaclab
"""

from __future__ import annotations

import argparse
import torch
import numpy as np
import math
import itertools
from pathlib import Path
from typing import Optional

# ============================================================================
# CRITICAL: Initialize Isaac Lab Application FIRST
# This MUST be done before importing any other Isaac Lab modules
# ============================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive Vision Debug with YOLO Detection")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force camera rendering
args_cli.enable_cameras = True

# Create app launcher and get simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# NOW import Isaac Lab modules (after app initialization)
# ============================================================================
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import FrameTransformer, Camera, CameraCfg, TiledCamera
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply

import omni.kit.viewport.utility
import carb.input
import omni.appwindow

# Application-specific imports
# 獨立腳本執行時需設定套件上下文，否則所有相對匯入
# (本檔案及依賴鏈: env_cfg → jetrover 等) 都會失敗
import sys as _sys
import os as _os
import importlib as _importlib

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PARENT_DIR = _os.path.dirname(_SCRIPT_DIR)
_PACKAGE_NAME = _os.path.basename(_SCRIPT_DIR)

if __package__ is None or __package__ == '':
    # 將父目錄加入 sys.path，讓 Python 找到套件
    if _PARENT_DIR not in _sys.path:
        _sys.path.insert(0, _PARENT_DIR)
    # 註冊套件，使相對匯入可用
    _importlib.import_module(_PACKAGE_NAME)
    __package__ = _PACKAGE_NAME

from .pickup_place_direct_0203_env_cfg import PickupPlaceDirect0203EnvCfg, SELECTED_OBJECT_IDS
from .pickup_place_direct_0203_vision_asym_env_cfg import PickupPlaceDirect0203VisionAsymEnvCfg
from .utils.yolo_detector import YOLODetector
from .utils.vision_encoder import get_vision_encoder


class InteractiveVisionDebug:
    """
    Interactive visualization of YOLO detection and 3D boundary boxes.
    """
    
    def __init__(self):
        """Initialize the interactive debug environment."""
        self.sim_context = None
        self.robot = None
        self.object = None
        self.ee_frame_transformer = None
        self.camera_low = None
        self.camera_high = None
        self.vision_encoder_low = None
        
        # Config
        self.cfg = None
        self.env_cfg = None
        
        # Object management
        self._current_object_idx = 0
        self._object_ids = SELECTED_OBJECT_IDS
        self._num_objects = len(self._object_ids)
        
        # YOLO detector
        self.yolo_detector = None
        
        # Visualization markers
        self.markers = None
        self.yolo_2d_corners_3d = None  # YOLO 2D bbox corners projected to 3D space
        self.yolo_3d_bbox = None         # YOLO 2D + Depth -> 3D bbox (8 points)
        self.gt_3d_bbox = None           # Ground truth 3D bbox (8 points)
        
        # Camera parameters (active camera - switches between low/high)
        self.camera_focal_length_x = None
        self.camera_focal_length_y = None
        self.camera_principal_point_x = None
        self.camera_principal_point_y = None
        
        # Camera source toggle (low-res vs high-res for YOLO visualization)
        self.use_high_res_camera = True
        
        # Snapshot control
        self._snapshot_requested = False
        self._snapshot_counter = 0
        
        print("[InteractiveVisionDebug] Initializing...")
    
    def setup(self):
        """Set up the simulation environment and visualization."""
        print("[InteractiveVisionDebug] Setting up simulation context...")
        
        # Create the asymmetric vision config
        self.cfg = PickupPlaceDirect0203VisionAsymEnvCfg()
        self.env_cfg = PickupPlaceDirect0203EnvCfg()
        
        try:
            # 使用 env_cfg 中已配置的 SimulationCfg（包含完整 PhysX 設定：重力、接觸參數等）
            # 僅覆寫 device="cpu" 以停用 GPU physics pipeline，讓 Physics Inspector 可用
            sim_cfg = self.cfg.sim
            sim_cfg.device = "cpu"
            self.sim_context = SimulationContext(sim_cfg)
            print("  ✓ Simulation context created")
        except Exception as e:
            print(f"  ✗ Failed to create simulation context: {e}")
            raise
        
        try:
            print("[InteractiveVisionDebug] Setting up scene...")
            self._setup_scene()
        except Exception as e:
            print(f"  ✗ Failed to setup scene: {e}")
            raise
        
        try:
            print("[InteractiveVisionDebug] Initializing vision components...")
            self._setup_vision()
        except Exception as e:
            print(f"  ✗ Failed to setup vision: {e}")
            raise
        
        try:
            print("[InteractiveVisionDebug] Setting up visualization markers...")
            self._setup_markers()
        except Exception as e:
            print(f"  ✗ Failed to setup markers: {e}")
            # Non-fatal error, continue without markers
        
        try:
            print("[InteractiveVisionDebug] Initializing YOLO detector...")
            self._setup_yolo()
        except Exception as e:
            print(f"  ✗ Warning: YOLO initialization failed: {e}")
            # Non-fatal error, continue without YOLO
        
        try:
            # Reset simulation
            print("[InteractiveVisionDebug] Resetting simulation...")
            self.sim_context.reset()
            self.sim_context.play()  # 確保物理引擎時間軸開始播放
            self.robot.reset()
            self.object.reset()
            
            # 手動將初始狀態寫入模擬器以喚醒 PhysX（這是獨立腳本中必要的步驟，否則物體可能進入休眠）
            self.robot.write_root_state_to_sim(self.robot.data.default_root_state)
            self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
            self.object.write_root_state_to_sim(self.object.data.default_root_state)
            
            print("  ✓ Simulation reset and play complete")
        except Exception as e:
            print(f"  ✗ Failed to reset simulation: {e}")
            raise
        
        try:
            # 初始化物件的 local bbox corners（需在 sim.reset() 之後，USD prim 已建立）
            print("[InteractiveVisionDebug] Computing object bounding box from USD...")
            self._init_object_local_corners()
        except Exception as e:
            print(f"  ✗ Warning: Failed to init object bbox: {e}")
        
        print("[InteractiveVisionDebug] Setup complete!\n")
    
    def _setup_scene(self):
        """Set up the simulation scene with robot, object, and cameras."""
        # Add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(size=(40.0, 40.0), color=(0.77, 0.77, 0.77)),
        )
        print("  ✓ Ground plane added")
        
        # ========== 建立環境 prim 結構 ==========
        # Isaac Lab 的 RL 環境管理器會自動建立 /World/envs/env_0, env_1, ...
        # 獨立腳本中需手動建立，否則 regex prim_path '/World/envs/env_.*' 找不到目標
        from pxr import UsdGeom
        stage = self.sim_context.stage
        UsdGeom.Xform.Define(stage, "/World/envs")
        UsdGeom.Xform.Define(stage, "/World/envs/env_0")
        print("  ✓ Environment prim /World/envs/env_0 created")
        
        # ========== 將 config 中的 regex prim_path 替換為具體路徑 ==========
        # env_cfg 中的 robot_cfg, object_cfg, ee_frame_cfg 使用 env_.*
        self.env_cfg.robot_cfg.prim_path = "/World/envs/env_0/Robot"
        self.env_cfg.object_cfg.prim_path = "/World/envs/env_0/Object"
        self.env_cfg.ee_frame_cfg.prim_path = "/World/envs/env_0/Robot/base_footprint"
        # ee_frame_cfg 的 target_frames 也需替換
        if hasattr(self.env_cfg.ee_frame_cfg, 'target_frames'):
            for tf in self.env_cfg.ee_frame_cfg.target_frames:
                if hasattr(tf, 'prim_path') and 'env_.*' in tf.prim_path:
                    tf.prim_path = tf.prim_path.replace("env_.*", "env_0")
        
        # vision cfg 中的 camera prim paths
        self.cfg.camera_low_cfg.prim_path = self.cfg.camera_low_cfg.prim_path.replace("env_.*", "env_0")
        self.cfg.camera_high_cfg.prim_path = self.cfg.camera_high_cfg.prim_path.replace("env_.*", "env_0")
        print("  ✓ Prim paths updated for single-env standalone mode")
        
        try:
            # Add robot
            print("  - Adding robot...")
            self.robot = Articulation(self.env_cfg.robot_cfg)
            print("  ✓ Robot added")
        except Exception as e:
            print(f"  ✗ Failed to add robot: {e}")
            raise
        
        try:
            # Add object
            print("  - Adding object...")
            self.object = RigidObject(self.env_cfg.object_cfg)
            print("  ✓ Object added")
        except Exception as e:
            print(f"  ✗ Failed to add object: {e}")
            raise
            
        try:
            # Add light sources to ensure YOLO can see the objects
            print("  - Adding lighting...")
            light_cfg = sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0)
            light_cfg.func("/World/Light", light_cfg)
            print("  ✓ Lighting added")
        except Exception as e:
            print(f"  ✗ Failed to add lighting: {e}")
        
        try:
            # Add end-effector frame transformer
            print("  - Adding EE frame transformer...")
            self.ee_frame_transformer = FrameTransformer(self.env_cfg.ee_frame_cfg)
            print("  ✓ EE frame transformer added")
        except Exception as e:
            print(f"  ✗ Failed to add EE frame transformer: {e}")
            # Non-fatal, continue without it
            self.ee_frame_transformer = None
        
        try:
            # Add cameras
            print("  - Adding cameras...")
            self.camera_low = TiledCamera(self.cfg.camera_low_cfg)
            self.camera_high = TiledCamera(self.cfg.camera_high_cfg)
            print("  ✓ Cameras added")
        except Exception as e:
            print(f"  ✗ Failed to add cameras: {e}")
            raise
    
    def _setup_vision(self):
        """Initialize vision encoders."""
        self.vision_encoder_low = get_vision_encoder(
            encoder_type=self.cfg.vision_encoder_type,
            image_height=self.cfg.camera_low_cfg.height,
            image_width=self.cfg.camera_low_cfg.width,
            feature_dim=self.cfg.vision_encoder_feature_dim,
            device=str(self.sim_context.device),
        )
        
        # 預計算兩組相機內參
        import math as _math
        h_fov = self.cfg.target_hfov
        
        # Low-res intrinsics (128×80)
        lw, lh = self.cfg.camera_low_cfg.width, self.cfg.camera_low_cfg.height
        _fl_low = lw / (2 * _math.tan(_math.radians(h_fov / 2)))
        self._intrinsics_low = {
            "fx": _fl_low, "fy": _fl_low,
            "cx": lw / 2.0, "cy": lh / 2.0,
            "w": lw, "h": lh,
        }
        
        # High-res intrinsics (640×400)
        hw, hh = self.cfg.camera_high_cfg.width, self.cfg.camera_high_cfg.height
        _fl_high = hw / (2 * _math.tan(_math.radians(h_fov / 2)))
        self._intrinsics_high = {
            "fx": _fl_high, "fy": _fl_high,
            "cx": hw / 2.0, "cy": hh / 2.0,
            "w": hw, "h": hh,
        }
        
        # 設定當前使用的內參（根據 use_high_res_camera 選擇）
        if self.use_high_res_camera:
            self._apply_intrinsics(self._intrinsics_high)
        else:
            self._apply_intrinsics(self._intrinsics_low)
        
        print(f"  ✓ Vision encoder initialized")
        print(f"    Low-Res:  {lw}×{lh}, fx=fy={_fl_low:.2f}px")
        print(f"    High-Res: {hw}×{hh}, fx=fy={_fl_high:.2f}px")
        print(f"    Active: {'High-Res' if self.use_high_res_camera else 'Low-Res'}")
    
    def _apply_intrinsics(self, intrinsics: dict):
        """套用相機內參到當前活動狀態。"""
        self.camera_focal_length_x = intrinsics["fx"]
        self.camera_focal_length_y = intrinsics["fy"]
        self.camera_principal_point_x = intrinsics["cx"]
        self.camera_principal_point_y = intrinsics["cy"]
    
    def _toggle_camera_source(self):
        """切換 YOLO 視覺化使用的相機來源 (Low-Res ↔ High-Res)。"""
        self.use_high_res_camera = not self.use_high_res_camera
        if self.use_high_res_camera:
            self._apply_intrinsics(self._intrinsics_high)
            src = "High-Res (640×400)"
        else:
            self._apply_intrinsics(self._intrinsics_low)
            src = "Low-Res (128×80)"
        print(f"[Camera Toggle] Switched YOLO input to: {src}")
        print(f"  fx=fy={self.camera_focal_length_x:.2f}, cx={self.camera_principal_point_x:.1f}, cy={self.camera_principal_point_y:.1f}")
    
    def _setup_yolo(self):
        """Initialize YOLO detector."""
        try:
            self.yolo_detector = YOLODetector(
                model_name=self.cfg.yolo_model_name,
                device=self.cfg.yolo_device,
                conf_threshold=0.25  # 降低置信度閾值以解決偵測不穩定的問題
            )
            print("  ✓ YOLO detector initialized explicitly with conf=0.25")
        except ImportError as e:
            print(f"  ✗ Warning: YOLO initialization failed: {e}")
            self.yolo_detector = None
    
    def _setup_markers(self):
        """Set up visualization markers for 3D bbox visualization."""
        # 使用 sim_utils.SphereCfg 作為 marker prototype
        # VisualizationMarkers 基於 UsdGeom.PointInstancer，
        # 透過 prototype index 切換不同顏色的 marker
        # Prototype 順序: 0=紅色(YOLO 2D), 1=綠色(YOLO 3D), 2=藍色(GT 3D)
        
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/bbox_markers",
            markers={
                "yolo_2d_corner": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),  # 紅色：YOLO 2D bbox 角點
                    ),
                ),
                "yolo_3d_bbox": sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),  # 綠色：YOLO 3D bbox（偵測結果）
                    ),
                ),
                "gt_3d_bbox": sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 1.0),  # 藍色：Ground truth 3D bbox
                    ),
                ),
            }
        )
        
        try:
            self.markers = VisualizationMarkers(marker_cfg)
            print("  ✓ Visualization markers initialized")
            print("    Color legend (prototype indices):")
            print("      [0] Red (1,0,0)   = YOLO 2D bbox corners (projected at fixed depth)")
            print("      [1] Green (0,1,0) = YOLO 2D bbox + Depth -> 3D bbox (detected)")
            print("      [2] Blue (0,0,1)  = Ground truth 3D bbox (from physics)")
        except Exception as e:
            print(f"  ✗ Warning: Could not initialize markers: {e}")
            self.markers = None
    
    def _get_camera_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get RGB and depth data from low-res camera.
        
        與訓練版本 _process_vision_data() 的前處理邏輯一致：
        - TiledCamera 輸出 rgb 為 (B, H, W, 4) RGBA，需取前 3 通道
        - 值域為 [0, 255] uint8，需轉換為 [0, 1] float
        
        Returns:
            (rgb, depth, rgb_uint8): RGB float [0,1] shape (H,W,3),
                                     depth in meters (H,W,1),
                                     RGB uint8 [0,255] shape (H,W,3)
        """
        try:
            # Ensure camera data is available
            if self.camera_low is None:
                print("[Camera] Low-res camera not initialized!")
                raise RuntimeError("Camera not initialized")
            
            # Get camera data (already in CUDA)
            if not hasattr(self.camera_low.data, 'output') or self.camera_low.data.output is None:
                print("[Camera] Camera output data not available!")
                raise RuntimeError("Camera output not available")
            
            # TiledCamera rgb output: (B, H, W, 4) 包含 alpha channel
            active_camera = self.camera_high if self.use_high_res_camera else self.camera_low
            rgb_data = active_camera.data.output["rgb"]    # Shape: (B, H, W, 4)
            depth_data = active_camera.data.output["depth"]  # Shape: (B, H, W, 1), in meters
            
            # Extract single environment (env 0)
            # 取前 3 通道去除 alpha，與訓練版本 rgb[..., :3] 一致
            rgb = rgb_data[0][..., :3]  # (H, W, 3)
            depth = depth_data[0]  # (H, W, 1)
            
            # Validate data
            if rgb.isnan().any() or depth.isnan().any():
                print("[Camera] Warning: NaN values detected in camera data")
            
            # 深度裁切到 Dabai DCW 工作範圍 (0.2~2.5m)
            # 與訓練版本 _process_vision_data 中 torch.clamp(depth, 0.2, 2.5) 一致
            # 注意: clipping_range 不能用於此目的，因為它也會裁切 RGB 渲染
            depth = torch.clamp(depth, 0.2, 2.5)
            
            # RGB 正規化到 [0, 1]，與訓練版本 _process_vision_data 一致
            rgb_float = rgb.float() / 255.0
            rgb_float = torch.clamp(rgb_float, 0.0, 1.0)
            
            # 保留 uint8 版本供 YOLO 使用
            rgb_uint8 = torch.clamp(rgb, 0, 255).to(torch.uint8)
            
            return rgb_float, depth, rgb_uint8
        
        except Exception as e:
            print(f"[Camera Error] Failed to get camera data: {e}")
            # Return dummy data to allow continued execution
            active_h = self._intrinsics_high["h"] if self.use_high_res_camera else self._intrinsics_low["h"]
            active_w = self._intrinsics_high["w"] if self.use_high_res_camera else self._intrinsics_low["w"]
            device = self.sim_context.device if self.sim_context else "cuda:0"
            return (
                torch.zeros((active_h, active_w, 3), device=device),
                torch.ones((active_h, active_w, 1), device=device) * 1.0,
                torch.zeros((active_h, active_w, 3), dtype=torch.uint8, device=device)
            )
    
    def _run_yolo_detection(self, rgb_uint8: torch.Tensor) -> Optional[dict]:
        """
        Run YOLO detection on RGB image.
        
        Args:
            rgb_uint8: RGB image as uint8 tensor, shape (H, W, 3)
        
        Returns:
            Detection results dict or None
        """
        if self.yolo_detector is None:
            return None
        
        try:
            # 驗證影像尺寸有效（前幾幀相機可能尚未渲染）
            if rgb_uint8.numel() == 0 or rgb_uint8.shape[0] == 0 or rgb_uint8.shape[1] == 0:
                return None
            
            # 使用 detect_batch()（與訓練環境相同的推理路徑）
            # detect_batch 使用 model() 直接推理 GPU tensor，
            # 而 detect() 使用 model.predict() 會觸發 YOLO 內部的 cv2.resize 導致小圖出錯
            img = rgb_uint8.unsqueeze(0)  # (H,W,3) → (1,H,W,3) batch format
            if img.device.type != 'cuda':
                img = img.to('cuda:0')
            
            batch_results = self.yolo_detector.detect_batch(img)
            
            if batch_results and len(batch_results) > 0 and batch_results[0] is not None:
                res = batch_results[0]
                if getattr(self, '_yolo_log_counter', 0) % 30 == 0:  # 印出偶爾的診斷
                    print(f"  [YOLO] Detected {res['num_detections']} objects. Confs: {res['confidences']}")
                self._yolo_log_counter = getattr(self, '_yolo_log_counter', 0) + 1
                return res
            
            if getattr(self, '_yolo_log_counter', 0) % 30 == 0:
                print("  [YOLO] No objects detected in this frame.")
            self._yolo_log_counter = getattr(self, '_yolo_log_counter', 0) + 1
            return None
        except Exception as e:
            print(f"[YOLO Detection Error] {e}")
            return None
    
    def _init_object_local_corners(self):
        """
        使用 USD BBoxCache 計算物件的局部邊界框角點。
        與訓練版本 env.py _update_object_local_corners() 完全一致。
        """
        import isaaclab.sim as sim_utils_local
        from pxr import UsdGeom, Usd
        
        try:
            stage = sim_utils_local.get_current_stage()
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            
            prim_path = "/World/envs/env_0/Object"
            prim = stage.GetPrimAtPath(prim_path)
            
            half_extents = torch.ones(3, device=self.sim_context.device) * 0.05
            
            if prim.IsValid():
                bound = bbox_cache.ComputeUntransformedBound(prim)
                box_range = bound.GetRange()
                min_pt = box_range.GetMin()
                max_pt = box_range.GetMax()
                
                hx = (max_pt[0] - min_pt[0]) / 2.0
                hy = (max_pt[1] - min_pt[1]) / 2.0
                hz = (max_pt[2] - min_pt[2]) / 2.0
                half_extents = torch.tensor([hx, hy, hz], dtype=torch.float32,
                                           device=self.sim_context.device)
                print(f"  ✓ Object USD bbox half_extents: ({hx:.4f}, {hy:.4f}, {hz:.4f})")
            else:
                print(f"  ⚠ Object prim not found at {prim_path}, using default 0.05m")
            
            # 套用物件縮放 (與 env_cfg.object_scale 一致)
            config_scale = torch.tensor(
                self.env_cfg.object_scale, device=self.sim_context.device
            )
            half_extents = half_extents * config_scale
            
            # 產生 8 個角點的局部座標 (與 env.py 一致)
            base_corners = torch.tensor(
                list(itertools.product([1, -1], repeat=3)),
                dtype=torch.float32, device=self.sim_context.device
            )  # (8, 3)
            self.object_local_corners = base_corners * half_extents  # (8, 3)
            
            print(f"  ✓ Object local corners initialized, scaled half_extents: "
                  f"({half_extents[0]:.4f}, {half_extents[1]:.4f}, {half_extents[2]:.4f})")
            
        except Exception as e:
            print(f"  ⚠ Failed to compute object bbox from USD: {e}")
            print(f"    Falling back to default 0.05m cube")
            half_extents = torch.ones(3, device=self.sim_context.device) * 0.05 * 0.6
            base_corners = torch.tensor(
                list(itertools.product([1, -1], repeat=3)),
                dtype=torch.float32, device=self.sim_context.device
            )
            self.object_local_corners = base_corners * half_extents
    
    def _get_object_bbox_3d(self) -> torch.Tensor:
        """
        取得物件的 ground truth 3D 邊界框角點 (世界座標)。
        使用 isaaclab.utils.math.quat_apply 進行旋轉，
        與訓練版本 observations.py object_bbox_corners() 完全一致。
        
        Returns:
            (8, 3) tensor of 3D corner positions in world frame
        """
        # 物件姿態
        obj_pos = self.object.data.root_com_pose_w[0, :3]  # (3,)
        obj_quat = self.object.data.root_com_pose_w[0, 3:7]  # (4,) [w, x, y, z]
        
        # 旋轉角點到世界座標 (與 observations.py 一致)
        # quat_apply 接受 (N, 4) quat 和 (N, 3) points
        quat_expanded = obj_quat.unsqueeze(0).expand(8, -1)  # (8, 4)
        rotated_corners = quat_apply(quat_expanded, self.object_local_corners)  # (8, 3)
        
        # 加上位置偏移
        corners_world = rotated_corners + obj_pos.unsqueeze(0)  # (8, 3)
        
        return corners_world
    
    @staticmethod
    def _quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
        """
        將四元數 [w, x, y, z] 轉換為 3×3 旋轉矩陣。
        純 numpy 實作，不依賴 scipy。
        
        Args:
            quat: (4,) quaternion in [w, x, y, z] format
        
        Returns:
            (3, 3) rotation matrix
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Rotation matrix from quaternion
        rot = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        return rot
    
    def _get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        從 TiledCamera 直接取得攝影機在世界座標中的姿態。
        使用當前 active 的相機（根據 use_high_res_camera 切換）。
        
        Returns:
            (cam_pos, cam_rot_matrix): Camera position (3,) and rotation matrix (3, 3)
        """
        try:
            # 根據當前設定選擇正確的相機
            active_camera = self.camera_high if self.use_high_res_camera else self.camera_low
            if active_camera is None:
                active_camera = self.camera_low  # fallback
            
            cam_pos = active_camera.data.pos_w[0].cpu().numpy()  # (3,)
            cam_quat = active_camera.data.quat_w_world[0].cpu().numpy()  # (4,) [w, x, y, z]
            
            # 將四元數轉換為旋轉矩陣
            cam_rot_matrix = self._quat_to_rot_matrix(cam_quat)
            
            return cam_pos, cam_rot_matrix
            
        except Exception as e:
            print(f"[Camera Pose] Error getting TiledCamera pose: {e}")
            print(f"  Falling back to identity pose at origin")
            return np.zeros(3), np.eye(3)
    
    def _hide_markers_from_camera(self):
        """
        將所有 marker 移到地底下方很遠的地方，讓相機拍不到。
        不使用 USD visibility（會永久破壞 PointInstancer），
        而是改用位置移動的方式。
        之後由 _update_visualization() 會重新設定正確位置。
        """
        if self.markers is None:
            return
        try:
            # 總共有 4 (YOLO 2D) + 8 (YOLO 3D) + 8 (GT 3D) = 20 個 marker
            device = self.sim_context.device
            far_away = torch.tensor([[0.0, 0.0, -100.0]], device=device).expand(20, -1)
            self.markers.visualize(
                translations=far_away,
                marker_indices=torch.zeros(20, dtype=torch.int32, device=device)  # 所有都用 index 0
            )
        except Exception:
            pass
    
    def _save_yolo_snapshot(self, rgb_uint8: torch.Tensor, depth: torch.Tensor):
        """
        手動拍攝快照：將相機畫面+YOLO偵測結果存為 PNG 圖片。
        
        Args:
            rgb_uint8: 相機 RGB 影像 (H, W, 3) uint8
            depth: 深度圖 (H, W, 1) float meters
        """
        import cv2
        import os
        
        self._snapshot_counter += 1
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_snapshots")
        os.makedirs(save_dir, exist_ok=True)
        
        # 將 tensor 轉為 numpy
        img_np = rgb_uint8.cpu().numpy().copy()  # (H, W, 3) RGB
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV 使用 BGR
        
        # 執行 YOLO 偵測
        detections = self._run_yolo_detection(rgb_uint8)
        
        # 在圖上畫出所有偵測框
        if detections and detections["num_detections"] > 0:
            boxes = detections["boxes"]
            confs = detections["confidences"]
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # 紅色邊框
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 信心分數標籤
                label = f"obj_{i}: {conf:.3f}"
                cv2.putText(img_bgr, label, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            info_text = f"Detected: {detections['num_detections']} objects"
        else:
            info_text = "No objects detected"
        
        # 加上快照資訊
        h, w = img_bgr.shape[:2]
        cv2.putText(img_bgr, f"Snapshot #{self._snapshot_counter} ({w}x{h})", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_bgr, info_text, (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 也儲存一份原始影像（無框）
        raw_path = os.path.join(save_dir, f"snapshot_{self._snapshot_counter:04d}_raw.png")
        yolo_path = os.path.join(save_dir, f"snapshot_{self._snapshot_counter:04d}_yolo.png")
        
        raw_bgr = cv2.cvtColor(rgb_uint8.cpu().numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(raw_path, raw_bgr)
        cv2.imwrite(yolo_path, img_bgr)
        
        # 印出 YOLO 偵測的詳細資料
        print(f"\n{'='*60}")
        print(f"📸 Snapshot #{self._snapshot_counter} saved!")
        print(f"  Raw image:  {raw_path}")
        print(f"  YOLO image: {yolo_path}")
        print(f"  Image shape: {rgb_uint8.shape}, range: [{rgb_uint8.min()}, {rgb_uint8.max()}]")
        if detections and detections["num_detections"] > 0:
            for i, (box, conf) in enumerate(zip(detections["boxes"], detections["confidences"])):
                print(f"  Box[{i}]: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}, conf={conf:.4f}")
        else:
            print(f"  YOLO result: No objects detected")
        print(f"{'='*60}\n")
    
    def _camera_to_world_transform(self, points_cam: np.ndarray) -> np.ndarray:
        """
        Transform points from camera frame (OpenCV convention) to world frame.
        
        OpenCV camera convention: +X right, +Y down,  +Z forward
        USD/Isaac Sim convention: +X right, +Y up,    -Z forward
        
        Args:
            points_cam: (N, 3) points in camera frame (OpenCV convention)
        
        Returns:
            (N, 3) points in world frame
        """
        cam_pos, cam_rot_matrix = self._get_camera_pose()
        
        # 先將 OpenCV 座標轉換為 USD 座標
        # OpenCV: (x, y, z) -> USD: (x, -y, -z)
        points_usd_cam = points_cam.copy()
        points_usd_cam[:, 1] *= -1  # Y 翻轉（OpenCV +Y down -> USD +Y up）
        points_usd_cam[:, 2] *= -1  # Z 翻轉（OpenCV +Z forward -> USD -Z forward）
        
        # 用 USD 相機的旋轉矩陣轉到世界座標
        # p_world = R_cam_to_world @ p_usd_cam + t_cam_to_world
        points_world = (cam_rot_matrix @ points_usd_cam.T).T + cam_pos
        
        return points_world
    
    def _update_visualization(self, rgb_uint8: torch.Tensor, depth: torch.Tensor):
        """
        Update the visualization with YOLO detection results and 3D boxes.
        
        注意：相機影像已在 marker 被隱藏時拍攝完畢，因此 marker 不會污染 YOLO 輸入。
        
        Args:
            rgb_uint8: RGB image from camera (captured BEFORE markers are shown)
            depth: Depth image from camera
        """
        if self.markers is None:
            return
        
        # Reset marker positions to NaN (invisible)
        device = self.sim_context.device
        yolo_2d_pos = torch.full((4, 3), float('nan'), device=device)
        yolo_3d_pos = torch.full((8, 3), float('nan'), device=device)
        gt_3d_pos = torch.full((8, 3), float('nan'), device=device)
        
        # Get camera pose information
        cam_pos, cam_rot_matrix = self._get_camera_pose()
        
        try:
            # Run YOLO detection
            detections = self._run_yolo_detection(rgb_uint8)
            
            if detections and detections["num_detections"] > 0:
                # Get center object (closest to image center)
                result = self.yolo_detector.get_center_object(
                    detections,
                    rgb_uint8.shape[0],
                    rgb_uint8.shape[1]
                )
                
                if result is not None:
                    center_bbox, idx = result
                    x1, y1, x2, y2 = center_bbox
                    
                    # ================== 從深度圖取得實際深度 ==================
                    # 取 bbox 區域的中值深度，比 fixed_depth=1.0 準確很多
                    ix1, iy1 = int(np.clip(x1, 0, depth.shape[1]-1)), int(np.clip(y1, 0, depth.shape[0]-1))
                    ix2, iy2 = int(np.clip(x2, 0, depth.shape[1]-1)), int(np.clip(y2, 0, depth.shape[0]-1))
                    depth_region = depth[iy1:iy2+1, ix1:ix2+1, 0]
                    valid_depths = depth_region[~torch.isnan(depth_region) & ~torch.isinf(depth_region) & (depth_region > 0.01)]
                    if len(valid_depths) > 0:
                        actual_depth = torch.median(valid_depths).item()
                    else:
                        actual_depth = 0.5  # fallback
                    
                    # ================== YOLO 2D Bbox Corners (projected to 3D) ==================
                    corners_2d = np.array([
                        [x1, y1], [x2, y1],
                        [x1, y2], [x2, y2],
                    ])
                    
                    corners_3d_cam = []
                    for u, v in corners_2d:
                        x = (u - self.camera_principal_point_x) * actual_depth / self.camera_focal_length_x
                        y = (v - self.camera_principal_point_y) * actual_depth / self.camera_focal_length_y
                        z = actual_depth
                        corners_3d_cam.append([x, y, z])
                    corners_3d_cam = np.array(corners_3d_cam)
                    
                    # Transform to world frame
                    corners_3d_world = self._camera_to_world_transform(corners_3d_cam)
                    yolo_2d_pos = torch.from_numpy(corners_3d_world).to(device).float()
                    
                    # === 前 5 次偵測：測試 4 種旋轉組合找出正確的 ===
                    det_counter = getattr(self, '_det_trace_counter', 0)
                    if det_counter < 5:
                        # 取 bbox 中心的相機座標 (OpenCV convention)
                        cu = (x1 + x2) / 2.0
                        cv = (y1 + y2) / 2.0
                        p_cv = np.array([[
                            (cu - self.camera_principal_point_x) * actual_depth / self.camera_focal_length_x,
                            (cv - self.camera_principal_point_y) * actual_depth / self.camera_focal_length_y,
                            actual_depth
                        ]])
                        # OpenCV→USD flipped version
                        p_usd = p_cv.copy()
                        p_usd[:, 1] *= -1
                        p_usd[:, 2] *= -1
                        
                        R = cam_rot_matrix
                        Rt = R.T
                        
                        combos = {
                            "R@cv":       (R @ p_cv.T).T[0] + cam_pos,
                            "R@usd":      (R @ p_usd.T).T[0] + cam_pos,
                            "R^T@cv":     (Rt @ p_cv.T).T[0] + cam_pos,
                            "R^T@usd":    (Rt @ p_usd.T).T[0] + cam_pos,
                        }
                        
                        # GT 參考
                        try:
                            gt_ref = self._get_object_bbox_3d().mean(dim=0).cpu().numpy()
                        except:
                            gt_ref = np.array([0, 0, 0])
                        
                        print(f"  [COMBO #{det_counter}] bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), depth={actual_depth:.3f}m")
                        print(f"    GT ref = ({gt_ref[0]:.3f}, {gt_ref[1]:.3f}, {gt_ref[2]:.3f})")
                        for name, pos in combos.items():
                            dist = np.linalg.norm(pos - gt_ref)
                            marker = " <<<" if dist < 0.15 else ""
                            print(f"    {name:10s} = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  err={dist:.3f}{marker}")
                    self._det_trace_counter = det_counter + 1
                    
                    # ================== YOLO 2D + Depth -> 3D Bbox ==================
                    if self.yolo_detector and hasattr(self.yolo_detector, 'project_2d_to_3d'):
                        projected_3d = self.yolo_detector.project_2d_to_3d(
                            center_bbox,
                            depth.cpu(),
                            self.camera_focal_length_x,
                            self.camera_focal_length_y,
                            self.camera_principal_point_x,
                            self.camera_principal_point_y
                        )
                        
                        if projected_3d is not None and len(projected_3d) == 8:
                            projected_3d_world = self._camera_to_world_transform(projected_3d)
                            yolo_3d_pos = torch.from_numpy(projected_3d_world).to(device).float()
        
        except Exception as e:
            print(f"[Visualization Error] {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # ================== Ground Truth 3D Bbox ==================
            gt_3d_pos = self._get_object_bbox_3d().to(device).float()  # (8, 3)
        except Exception as e:
            print(f"[GT Bbox Error] {e}")
        
        # Update markers 使用 VisualizationMarkers.visualize() API
        # 將所有 marker 位置合併為一個陣列，並指定對應的 prototype index
        # Prototype indices: 0=紅色(YOLO 2D), 1=綠色(YOLO 3D), 2=藍色(GT 3D)
        try:
            if self.markers is not None:
                # 收集有效的 marker 位置和對應的 prototype indices
                all_positions = []
                all_indices = []
                
                # YOLO 2D corners (prototype 0 = 紅色)
                if not torch.isnan(yolo_2d_pos).all():
                    all_positions.append(yolo_2d_pos)
                    all_indices.extend([0] * yolo_2d_pos.shape[0])
                
                # YOLO 3D bbox (prototype 1 = 綠色)
                if not torch.isnan(yolo_3d_pos).all():
                    all_positions.append(yolo_3d_pos)
                    all_indices.extend([1] * yolo_3d_pos.shape[0])
                
                # GT 3D bbox (prototype 2 = 藍色)
                if not torch.isnan(gt_3d_pos).all():
                    all_positions.append(gt_3d_pos)
                    all_indices.extend([2] * gt_3d_pos.shape[0])
                
                if len(all_positions) > 0:
                    translations = torch.cat(all_positions, dim=0)  # (M, 3)
                    # IsaacLab VisualizationMarkers expects marker_indices as a PyTorch Tensor
                    indices_tensor = torch.tensor(all_indices, dtype=torch.int32, device=device)
                    self.markers.visualize(
                        translations=translations,
                        marker_indices=indices_tensor
                    )
                    
                    # 每 100 次 YOLO 偵測印出一次 marker 位置診斷
                    log_counter = getattr(self, '_marker_log_counter', 0)
                    if log_counter % 100 == 0:
                        has_yolo_2d = not torch.isnan(yolo_2d_pos).all()
                        has_yolo_3d = not torch.isnan(yolo_3d_pos).all()
                        has_gt = not torch.isnan(gt_3d_pos).all()
                        print(f"  [Markers] Total={translations.shape[0]} points, "
                              f"YOLO_2D(red)={'✓' if has_yolo_2d else '✗'}, "
                              f"YOLO_3D(green)={'✓' if has_yolo_3d else '✗'}, "
                              f"GT(blue)={'✓' if has_gt else '✗'}")
                        if has_gt:
                            gt_center = gt_3d_pos.mean(dim=0)
                            print(f"           GT center: ({gt_center[0]:.3f}, {gt_center[1]:.3f}, {gt_center[2]:.3f})")
                        if has_yolo_2d:
                            y2d_center = yolo_2d_pos.mean(dim=0)
                            print(f"           YOLO 2D center: ({y2d_center[0]:.3f}, {y2d_center[1]:.3f}, {y2d_center[2]:.3f})")
                    self._marker_log_counter = log_counter + 1
                else:
                    # 沒有任何有效 marker，也印一次
                    log_counter = getattr(self, '_marker_log_counter', 0)
                    if log_counter % 100 == 0:
                        print(f"  [Markers] No valid positions to display (all NaN)")
                    self._marker_log_counter = log_counter + 1
        except Exception as e:
            print(f"[Marker Update Error] {e}")
    
    def _reset_environment(self):
        """Reset the environment to initial state."""
        print("[Reset] Resetting environment...")
        
        # Reset robot to default pose
        self.robot.set_joint_position_target(
            torch.tensor(
                [0.0, 0.61086472, 0.7853975, 0.95993027, 0.0, 1.569993],
                device=self.sim_context.device
            ),
            joint_indices=torch.arange(6, device=self.sim_context.device)
        )
        
        # Reset object position
        obj_pos = torch.tensor([0.14875, 0.0, 0.15], device=self.sim_context.device)
        obj_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.sim_context.device)
        self.object.write_root_pose_to_sim(
            torch.cat([obj_pos.unsqueeze(0), obj_rot.unsqueeze(0)], dim=-1),
            env_indices=torch.tensor([0], device=self.sim_context.device)
        )
        
        self.sim_context.step()
        print("[Reset] Environment reset complete")
    
    def _switch_object(self, direction: int = 1):
        """
        Switch to next or previous object.
        
        Args:
            direction: 1 for next, -1 for previous
        """
        self._current_object_idx = (self._current_object_idx + direction) % self._num_objects
        current_obj_id = self._object_ids[self._current_object_idx]
        print(f"[Object Switch] Switched to object {self._current_object_idx}: ID={current_obj_id}")
        
        # In a full implementation, you would reload the object USD file here
        # For now, just update the visualization
    
    def run(self):
        """Main loop for interactive visualization."""
        print("[InteractiveVisionDebug] Starting main loop...")
        print("\n" + "="*60)
        print("INTERACTIVE VISION DEBUG - CONTROLS")
        print("="*60)
        print("Right-click + drag  : Rotate camera view")
        print("Mouse wheel         : Zoom in/out")
        print("Middle-click        : Pan camera")
        print("\nScene Visualization:")
        print("  Red (1,0,0)   = YOLO 2D bbox corners at fixed depth (1.0m)")
        print("  Green (0,1,0) = YOLO 2D bbox + Depth -> Full 3D bbox (8 corners)")
        print("  Blue (0,0,1)  = Ground truth 3D bbox (from object physics)")
        print(f"\nActive YOLO Camera: {'High-Res (640×400)' if self.use_high_res_camera else 'Low-Res (128×80)'}")
        print("  Toggle with _toggle_camera_source() in console")
        print("\n📸 YOLO Snapshot:")
        print("  Auto-taken at step 50 for initial diagnosis")
        print("  Manual: Open another terminal and run:")
        print("    touch debug_snapshots/TAKE_SNAPSHOT")
        print("  Images saved to: debug_snapshots/ folder")
        print("="*60)
        print("\nPhysics Inspector:")
        print("  Window menu -> Omniverse -> Physics Inspector")
        print("  Use it to manually adjust robot joint angles in real-time")
        print("="*60)
        print("\nStarting simulation loop...\n")
        
        step = 0
        yolo_detections_count = 0
        last_print_step = 0
        yolo_interval = 10  # 每 N 幀執行一次 YOLO（減少 marker 閃爍）
        
        # 快照觸發檔案路徑
        import os
        snapshot_trigger_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_snapshots")
        os.makedirs(snapshot_trigger_dir, exist_ok=True)
        snapshot_trigger_file = os.path.join(snapshot_trigger_dir, "TAKE_SNAPSHOT")
        
        try:
            # 取得 physics dt 用於 asset update
            sim_dt = self.sim_context.get_physics_dt()
            
            while simulation_app.is_running():
                is_yolo_frame = (step % yolo_interval == 0)
                
                # ============================================================
                # YOLO 幀：隱藏 marker → 渲染（乾淨畫面）→ 拍攝 → YOLO → 顯示 marker
                # 非 YOLO 幀：直接渲染（marker 保持顯示）
                # ============================================================
                if is_yolo_frame:
                    # 先把 marker 移到地底下，讓相機拍不到
                    self._hide_markers_from_camera()
                
                self.sim_context.step(render=True)
                
                # 更新所有資產的資料緩衝區
                self.robot.update(sim_dt)
                self.object.update(sim_dt)
                if self.camera_low is not None:
                    self.camera_low.update(sim_dt)
                if self.camera_high is not None:
                    self.camera_high.update(sim_dt)
                if self.ee_frame_transformer is not None:
                    self.ee_frame_transformer.update(sim_dt)
                
                if is_yolo_frame:
                    try:
                        # 取得「乾淨的」相機資料（不含 marker）
                        rgb, depth, rgb_uint8 = self._get_camera_data()
                        
                        # 前 10 步診斷輸出
                        if step < 10:
                            print(f"[Step {step}] Camera: rgb_uint8 shape={rgb_uint8.shape}, "
                                  f"dtype={rgb_uint8.dtype}, range=[{rgb_uint8.min()}, {rgb_uint8.max()}], "
                                  f"depth range=[{depth.min():.3f}, {depth.max():.3f}]")
                        
                        # 檢查快照觸發
                        take_snapshot = (step == 1000)
                        if os.path.exists(snapshot_trigger_file):
                            take_snapshot = True
                            os.remove(snapshot_trigger_file)
                        if take_snapshot:
                            self._save_yolo_snapshot(rgb_uint8, depth)
                        
                        # 更新 YOLO 及 3D Marker 位置（此步驟會把 marker 放回正確位置）
                        self._update_visualization(rgb_uint8, depth)
                    except Exception as e:
                        if step < 10 or step % 200 == 0:
                            print(f"[Step {step}] Error in visualization update: {e}")
                
                step += 1
                
                # Print status every 200 steps
                if step - last_print_step >= 200:
                    try:
                        # Get robot joint info
                        jpos = self.robot.data.joint_pos[0][:5].cpu().numpy()
                        print(f"[Step {step:6d}] Robot joints: {jpos}")
                        
                        # Get object position
                        obj_pos = self.object.data.root_pos_w[0].cpu().numpy()
                        print(f"               Object pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
                    except Exception as e:
                        print(f"[Step {step}] Status update error: {e}")
                    
                    last_print_step = step
        
        except KeyboardInterrupt:
            print("\n[InteractiveVisionDebug] Keyboard interrupt received...")
        except Exception as e:
            print(f"\n[InteractiveVisionDebug] Simulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n[InteractiveVisionDebug] Closing simulation context...")
            self.sim_context.close()
            print("[InteractiveVisionDebug] Done!")



def main():
    """Entry point for the interactive vision debug script."""
    global debug  # 讓使用者可從 Isaac Sim Console 存取 debug 物件
    try:
        debug = InteractiveVisionDebug()
        debug.setup()
        debug.run()
    finally:
        # Ensure Isaac Sim app is properly closed
        if simulation_app.is_running():
            print("[main] Closing Isaac Sim application...")
            simulation_app.close()
        print("[main] Script completed!")


if __name__ == "__main__":
    main()
