# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections import deque
from typing import Dict

from .pickup_place_direct_0203_vision_env import PickupPlaceDirect0203VisionEnv
from .pickup_place_direct_0203_vision_asym_env_cfg import PickupPlaceDirect0203VisionAsymEnvCfg
from .utils.yolo_detector import YOLODetector
from .utils.performance_monitor import get_perf_monitor


class PickupPlaceDirect0203VisionAsymEnv(PickupPlaceDirect0203VisionEnv):
    """
    Asymmetric Direct RL environment with dual-camera vision system and YOLO object detection.
    
    **Observation Architecture (Option A: Clean Separation)**
    
    1. **Policy (Actor)**: Vision-based observations only (no ground truth)
       - Forced to rely on CNN features + YOLO detection for object localization
       - Structure: Proprio(12) + Cmd+Target(9) + Vision_4L+1H(576) + YOLO_BBox(24)
       - Total Dimension: 621 dims
       - Rationale: Trains agent with realistic vision-based perception
    
    2. **Critic**: Full access to privileged environment information
       - Maintains accurate value function with ground truth state
       - Structure: Proprio(12) + Vision_4L+1H(576) + YOLO_BBox(24) + GT_Privileged(48)
       - Where GT_Privileged = ObjPos(3) + ObjBBox(24) + Target(3) + Action(6) + Padding(12)
       - Total Dimension: 660 dims
       - Rationale: Accurate value estimation improves learning stability
    
    **Observation Duplication: RESOLVED**
    - Previous design: gt_obj_pos + gt_bbox appeared in state_only AND gt_features (100% redundancy)
    - This design: Clear separation - state_only is proprioceptive, gt_features is all privileged info
    - No more redundant information across observation components
    
    **Multi-Frame CNN Features:**
    - Low-Res: 4 frames × 128 dims = 512 dims (captures temporal dynamics)
    - High-Res: 1 frame × 64 dims = 64 dims (static scene context)
    - Total Vision: 576 dims (included in both policy and critic)
    """

    cfg: PickupPlaceDirect0203VisionAsymEnvCfg

    def __init__(self, cfg: PickupPlaceDirect0203VisionAsymEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize asymmetric vision environment with YOLO detection.
        
        Args:
            cfg: Configuration for the asymmetric vision environment
            render_mode: Optional render mode for visualization
            **kwargs: Additional arguments for parent class
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # ========== YOLO DETECTOR INITIALIZATION ==========
        try:
            self.yolo_detector = YOLODetector(
                model_name=self.cfg.yolo_model_name,
                device=self.cfg.yolo_device,
                conf_threshold=self.cfg.yolo_conf_threshold
            )
            print(f"[PickupPlaceDirect0203VisionAsymEnv] YOLO Detector initialized with {self.cfg.yolo_model_name}")
            print(f"  YOLO camera source: {self.cfg.yolo_camera_source}")
        except ImportError as e:
            print(f"[PickupPlaceDirect0203VisionAsymEnv] Warning: YOLO initialization failed: {e}")
            print("Continuing without YOLO detection...")
            self.yolo_detector = None
        
        # 預計算 YOLO 使用的相機內參（依據 yolo_camera_source 選擇）
        if self.cfg.yolo_camera_source == "high":
            self._yolo_cam_key = "camera_high"
            self._yolo_img_h = self.cfg.camera_high_image_height
            self._yolo_img_w = self.cfg.camera_high_image_width
            self._yolo_fx = self.cfg.camera_high_focal_length_x
            self._yolo_fy = self.cfg.camera_high_focal_length_y
            self._yolo_cx = self.cfg.camera_high_principal_point_x
            self._yolo_cy = self.cfg.camera_high_principal_point_y
        else:
            self._yolo_cam_key = "camera_low"
            self._yolo_img_h = self.cfg.camera_image_height
            self._yolo_img_w = self.cfg.camera_image_width
            self._yolo_fx = self.cfg.camera_focal_length_x
            self._yolo_fy = self.cfg.camera_focal_length_y
            self._yolo_cx = self.cfg.camera_principal_point_x
            self._yolo_cy = self.cfg.camera_principal_point_y
        
        # ========== MULTI-FRAME CNN FEATURE BUFFER ==========
        # Low-Res Camera: Stack 4 frames for temporal dynamics
        # High-Res Camera: Single frame (fixed context for entire episode)
        # 
        # Reasoning:
        # - Low-res: 4 frames × 128 dims = 512 dims (captures motion & dynamics)
        # - High-res: 1 frame × 64 dims = 64 dims (static scene context, same throughout episode)
        # Total vision: 512 + 64 = 576 dims
        # Policy obs: 10 (proprio) + 9 (target+action) + 576 (vision) + 24 (yolo) = 619 dims
        
        self.num_vision_frames = 4  # Number of low-res frames to stack
        self.cnn_feature_history = deque(maxlen=self.num_vision_frames)
        
        # High-res context cache (single frame per episode, not multi-frame)
        self.high_res_context_single = None
        self.high_res_context_valid = False
        
        # Pre-allocate buffer: (num_envs, num_frames, feature_dim)
        # Will be filled on first call to _get_observations()
        self.cnn_features_stacked = None
        
        # ========== JOINT POSITION HISTORY FOR VELOCITY ESTIMATION ==========
        # Store joint positions for numerical differentiation of velocity
        # Using deque with maxlen=2 to keep only current and previous positions
        self.joint_pos_history = deque(maxlen=2)
        
        # ========== VELOCITY ESTIMATION CACHE ==========
        # Cache for simulated joint velocity (computed via numerical differentiation)
        self.jvel_simulated = torch.zeros(
            (self.num_envs, 5),
            device=self.device,
            dtype=torch.float32
        )
        self.use_simulated_jvel = False  # Toggle flag to switch between simulator and computed velocity
        
        # Initialize performance monitor for tracking bottlenecks
        self.perf_monitor = get_perf_monitor()
        self.perf_monitor.set_device(self.device)
        self.perf_log_interval = 100  # Log performance every N steps
        
        print(f"[PickupPlaceDirect0203VisionAsymEnv] Asymmetric environment initialized (Option A)")
        print(f"  - Policy observation: 621 dims (12 Proprio + 9 Cmd+Target + 576 Vision_4L+1H + 24 YOLO_BBox)")
        print(f"    • Vision breakdown: 4× low-res (512) + 1× high-res (64) = 576 dims")
        print(f"  - Critic observation: 660 dims (12 Proprio + 576 Vision_4L+1H + 24 YOLO_BBox + 48 GT_Privileged)")
        print(f"    • GT_Privileged: ObjPos(3) + ObjBBox(24) + Target(3) + Action(6) + Padding(12) = 48 dims")
        print(f"  - Observation Architecture: **OPTION A** - Clean separation of proprioceptive vs privileged info")
        print(f"  - YOLO Confidence Encoding: EXPLICIT (confidence dimension [4] = 0.0 means 'object not visible')")
        print(f"  - Performance monitoring: ENABLED (bottleneck tracking active)")

    def _get_observations(self) -> dict:
        """
        Collect asymmetric observations including multi-frame CNN features and YOLO detection.
        
        Pipeline:
        1. Get base observations from parent VisionEnv (240 dims)
        2. Extract current CNN features and add to history buffer (multi-frame stacking)
        3. Run YOLO detection on current RGB frame
        4. Estimate JVel via numerical differentiation
        5. Construct actor obs: exclude GT object state, include multi-frame vision + YOLO
        6. Construct critic obs: include multi-frame vision + YOLO + GT object state (privileged)
        
        **Option A: Clean Observation Separation**
        - **Policy (Actor)**: Vision-only, no ground truth
          Structure: Proprio(12) + Cmd+Target(9) + Vision(576) + YOLO(24) = 621 dims
          Role: Learns from vision and velocity feedback alone
          
        - **Critic**: Can access privileged environment information
          Structure: Proprio(12) + Vision(576) + YOLO(24) + GT_Privileged(48) = 660 dims
          where GT_Privileged = ObjPos(3) + ObjBBox(24) + Target(3) + Action(6) + Padding(12)
          Role: Accurate value estimation with full state knowledge
        
        Observation Duplication: **RESOLVED**
        - Before: gt_obj_pos + gt_bbox appeared in both state_only and gt_features (100% redundancy)
        - After: state_only contains only proprioceptive info (JPos+JVel)
                gt_features contains all privileged environment info (ObjPos+ObjBBox+Target+Action)
                Clear separation without duplication
        
        Returns:
            dict: {
                "policy": Actor observations (621 dims) - Vision-dependent only
                "critic": Critic observations (660 dims) - Vision + Privileged GT
            }
        """
        # Start performance tracking
        self.perf_monitor.start_timer("_get_observations_total")
        self.perf_monitor.start_timer("parent_observations")
        # Get the full observations from the parent class (VisionEnv)
        # Parent returns: {"policy": combined_obs} where combined_obs is (B, 238)
        # Structure: [State(48) | Vision_Low(128) | Vision_High(64)]
        # State(48): JPos(6), JVel(6), ObjPos(3), ObjBBox(24), Target(3), Action(6)
        full_obs_dict = super()._get_observations()
        full_obs = full_obs_dict["policy"]  # Shape: (B, 240)
        self.perf_monitor.end_timer("parent_observations")
        
        # ========== EXTRACT AND BUFFER MULTI-FRAME CNN FEATURES ==========
        self.perf_monitor.start_timer("cnn_feature_extraction")
        
        # Separate low-res (128 dims) and high-res (64 dims) from parent observation
        # Parent obs structure: [State(48) | LowRes(128) | HighRes(64)]
        current_cnn_features_low = full_obs[:, 48:176]    # 128 dims (low-res continuous)
        current_cnn_features_high = full_obs[:, 176:240]  # 64 dims (high-res context)
        
        # ===== LOW-RES: Multi-frame stacking (4 frames for temporal dynamics) =====
        # Initialize buffer on first call
        if self.cnn_features_stacked is None:
            # Pre-fill with current frame repeated (no history yet)
            for _ in range(self.num_vision_frames):
                self.cnn_feature_history.append(current_cnn_features_low.clone())
            # Stack: (B, num_frames, 128)
            self.cnn_features_stacked = torch.stack(
                list(self.cnn_feature_history), dim=1
            )  # Shape: (B, 4, 128)
        else:
            # Add current frame to history
            self.cnn_feature_history.append(current_cnn_features_low.clone())
            # Stack all frames: (B, num_frames, 128)
            self.cnn_features_stacked = torch.stack(
                list(self.cnn_feature_history), dim=1
            )  # Shape: (B, 4, 128)
        
        # Flatten multi-frame low-res features: (B, 4*128) = (B, 512)
        vision_low_multiframe = self.cnn_features_stacked.reshape(
            self.num_envs, -1
        )  # Shape: (B, 512) [4 frames * 128 dims/frame]
        
        # ===== HIGH-RES: Single frame per episode (static context) =====
        # Store high-res context only once and reuse throughout episode
        if not self.high_res_context_valid:
            # First time: cache the high-res context
            self.high_res_context_single = current_cnn_features_high.clone()
            self.high_res_context_valid = True
        
        # Use cached high-res context (same throughout episode)
        vision_high_single = self.high_res_context_single.clone()
        
        self.perf_monitor.end_timer("cnn_feature_extraction")
        
        # ========== YOLO DETECTION ==========
        self.perf_monitor.start_timer("yolo_detection")
        # Extract RGB and Depth from low-res camera and run YOLO detection
        # Returns normalized or 3D bounding box features
        yolo_bbox_features = self._get_yolo_detection()  # Shape: (B, 24)
        self.perf_monitor.end_timer("yolo_detection")
        
        # ========== JOINT VELOCITY ESTIMATION ==========
        # Extract current joint positions for numerical differentiation
        jpos_current = full_obs[:, 0:6]  # Shape: (B, 6)
        
        # Store in history for differentiation
        self.joint_pos_history.append(jpos_current)
        
        # Compute JVel via numerical differentiation if we have history
        if len(self.joint_pos_history) == 2:
            jpos_prev = self.joint_pos_history[0]
            jpos_curr = self.joint_pos_history[1]
            # Numerical differentiation: dq/dt
            # dt = 0.01s (100Hz simulator, decimation=2 -> 50Hz control)
            self.jvel_simulated = (jpos_curr - jpos_prev) / 0.01
        
        # Get JVel from full observation (simulator's velocity)
        jvel_from_sim = full_obs[:, 6:12]  # Shape: (B, 6)
        
        # Choose which JVel to use (option: can add option to toggle in config)
        jvel_to_use = self.jvel_simulated if self.use_simulated_jvel else jvel_from_sim
        
        # ========== BUILD POLICY OBSERVATION ==========
        self.perf_monitor.start_timer("construct_policy_obs")
        # Structure: [Proprio(12) | Target+Action(9) | Vision_LowRes_4frames(512) | Vision_HighRes_1frame(64) | YOLO_BBox(24)]
        # = 12 + 9 + 512 + 64 + 24 = 621 dims
        # Note: EXCLUDES ground truth object state
        
        # 1. Proprioception: Joint positions and velocities
        jpos = full_obs[:, 0:6]           # Joint positions (B, 6)
        jvel = jvel_to_use                # Joint velocities (B, 6)
        proprio = torch.cat([jpos, jvel], dim=-1)  # (B, 12)
        
        # 2. Command and target
        cmd_action = full_obs[:, 39:48]   # Target(39-41) + Action(42-47) (B, 9)
        
        # 3. Vision: Low-res 4-frame stack + High-res single frame
        vision_combined = torch.cat([
            vision_low_multiframe,  # (B, 512) ← 4 frames × 128 dims
            vision_high_single      # (B, 64)  ← 1 frame × 64 dims
        ], dim=-1)  # → (B, 576)
        
        # Concatenate all components
        policy_obs = torch.cat([
            proprio,                # (B, 12)
            cmd_action,             # (B, 9)
            vision_combined,        # (B, 576) ← Low-res 4frames + High-res 1frame
            yolo_bbox_features      # (B, 24)
        ], dim=-1)  # → (B, 621)
        self.perf_monitor.end_timer("construct_policy_obs")
        
        # ========== BUILD CRITIC OBSERVATION ==========
        self.perf_monitor.start_timer("construct_critic_obs")
        # Structure (Option A): [State_Proprio(12) | Vision_LowRes_4frames(512) | Vision_HighRes_1frame(64) | YOLO_BBox(24) | GT_Privileged(48)]
        # = 12 + 512 + 64 + 24 + 48 = 660 dims
        # Note: Clear separation of dynamics (proprioceptive) vs environment privileged information
        
        # ===== OPTION A: Clean Separation =====
        # state_only = proprioceptive only (JPos + JVel)
        # gt_features = privileged environment info (ObjPos + ObjBBox + Target + Action + padding)
        
        # Extract proprioceptive state only: JPos(6) + JVel(6) = 12 dims
        state_only = full_obs[:, :12]  # JPos(0:6) + JVel(6:12) → (B, 12)
        
        # Extract privileged ground truth information: ObjPos(3) + ObjBBox(24) + Target(3) + Action(6) = 36 dims
        # From full_obs indices [12:48] which contains: ObjPos(12:15) + ObjBBox(15:39) + Target(39:42) + Action(42:48)
        gt_privileged_base = full_obs[:, 12:48]  # All object/task info from state: (B, 36)
        
        # Pad to 48 dims for network symmetry and future extensions
        padding_dim = 48 - 36  # 12 dimensions for padding
        gt_features = torch.cat([
            gt_privileged_base,                   # (B, 36) = ObjPos(3) + ObjBBox(24) + Target(3) + Action(6)
            torch.zeros(
                (self.num_envs, padding_dim),
                device=self.device,
                dtype=torch.float32
            )                                     # (B, 12) for future extensions
        ], dim=-1)  # → (B, 48)
        
        # Concatenate all components for critic
        critic_obs = torch.cat([
            state_only,                   # (B, 12)  ← Proprioceptive only
            vision_combined,              # (B, 576) ← Low-res 4frames + High-res 1frame
            yolo_bbox_features,          # (B, 24)  ← Detected object bbox
            gt_features                  # (B, 48)  ← Privileged environment info
        ], dim=-1)  # → (B, 660) [12 + 576 + 24 + 48]
        self.perf_monitor.end_timer("construct_critic_obs")
        
        # End total timing and log if needed
        self.perf_monitor.end_timer("_get_observations_total")
        
        # Log performance summary periodically
        if hasattr(self, 'common_step_counter') and self.common_step_counter % self.perf_log_interval == 0:
            self.perf_monitor.log_summary(
                step=self.common_step_counter,
                num_envs=self.num_envs,
                prefix="[AsymEnv] "
            )
        
        return {
            "policy": policy_obs,
            "critic": critic_obs
        }
    
    def _get_yolo_detection(self) -> torch.Tensor:
        """
        Run YOLO detection on camera RGB stream and extract bbox features.
        
        Camera source is determined by cfg.yolo_camera_source ("low" or "high").
        
        YOLO Feature Format (24 dims):
        - dims 0-3: Normalized 2D bounding box [x1_norm, y1_norm, x2_norm, y2_norm]
        - dim 4: Detection confidence [0, 1] - EXPLICIT SIGNAL for "object visible"
        - dims 5-23: Optional 3D corners or padding (currently padding)
        
        Returns:
            torch.Tensor: Shape (B, 24) containing bounding box features with confidence indicator
        """
        # Initialize output tensor
        batch_size = self.num_envs
        yolo_features = torch.zeros(
            (batch_size, 24),
            device=self.device,
            dtype=torch.float32
        )
        
        if self.yolo_detector is None:
            return yolo_features
        
        # 依據 yolo_camera_source 配置選擇相機
        try:
            camera_data = self.scene[self._yolo_cam_key]
            rgb_images = camera_data.data.output["rgb"]      # (B, H, W, 4)
            depth_images = camera_data.data.output["depth"]  # (B, H, W, 1)
            
            # 去除 alpha channel
            if rgb_images.shape[-1] == 4:
                rgb_images_rgb = rgb_images[..., :3]
            else:
                rgb_images_rgb = rgb_images
                
            # Batched YOLO detection
            try:
                batched_detections = self.yolo_detector.detect_batch(rgb_images_rgb)
            except Exception as e:
                print(f"[YOLODetection] 批次預測失敗: {e}")
                batched_detections = [None] * batch_size
                
            # Process each environment independently
            for env_idx in range(batch_size):
                try:
                    depth = depth_images[env_idx]  # (H, W, 1)
                    
                    detections = batched_detections[env_idx]
                    
                    if detections is not None and detections["num_detections"] > 0:
                        center_result = self.yolo_detector.get_center_object(
                            detections,
                            image_height=self._yolo_img_h,
                            image_width=self._yolo_img_w
                        )
                        
                        if center_result is not None:
                            bbox_2d, bbox_idx = center_result
                            conf = float(detections["confidences"][bbox_idx])
                            
                            # Project 2D bbox to 3D
                            bbox_3d = self.yolo_detector.project_2d_to_3d(
                                bbox_2d,
                                depth,
                                fx=self._yolo_fx,
                                fy=self._yolo_fy,
                                cx=self._yolo_cx,
                                cy=self._yolo_cy
                            )
                            
                            if bbox_3d is not None and bbox_3d.shape[0] >= 8:
                                bbox_3d_partial = bbox_3d[:4, :].flatten()
                                yolo_features[env_idx, 0:12] = torch.from_numpy(bbox_3d_partial).float().to(self.device)
                                yolo_features[env_idx, 4] = conf
                            else:
                                x1, y1, x2, y2 = bbox_2d
                                x1_norm = float(x1) / self._yolo_img_w
                                y1_norm = float(y1) / self._yolo_img_h
                                x2_norm = float(x2) / self._yolo_img_w
                                y2_norm = float(y2) / self._yolo_img_h
                                
                                yolo_features[env_idx, 0] = x1_norm
                                yolo_features[env_idx, 1] = y1_norm
                                yolo_features[env_idx, 2] = x2_norm
                                yolo_features[env_idx, 3] = y2_norm
                                yolo_features[env_idx, 4] = conf
                
                except Exception as e:
                    print(f"[YOLODetection] Error processing env {env_idx}: {e}")
        
        except Exception as e:
            print(f"[YOLODetection] Failed to access camera data ({self._yolo_cam_key}): {e}")
        
        return yolo_features
