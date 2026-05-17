# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os
from pathlib import Path
from typing import Optional

from isaaclab.sensors import TiledCamera
from isaaclab.sim import SimulationContext
import numpy as np

from .pickup_place_direct_0203_env import PickupPlaceDirect0203Env
from .pickup_place_direct_0203_vision_env_cfg import PickupPlaceDirect0203VisionEnvCfg
from .vision_encoder import get_vision_encoder
from .mdp import rewards as mdp_rewards
from isaaclab.managers import SceneEntityCfg

class PickupPlaceDirect0203VisionEnv(PickupPlaceDirect0203Env):
    """
    Direct RL environment with dual-camera vision system.
    
    Key Features:
    - **Dual Camera Architecture**:
      1. Low-Res (80×128, 30Hz): Continuous proprioceptive feedback for RL agent
         - Provides immediate perception for control decisions
         - Uses RGB + Depth for robust feature extraction
      
      2. High-Res (400×640, Manual): Context extraction at episode start
         - Provides detailed static features (texture, layout)
         - Triggered manually to minimize computational overhead
         - Uses RGB only
    
    - Inherits all non-vision RL logic from base PickupPlaceDirect0203Env
    - Combines state observations (46 dims) with multi-level vision features
    
    Observation structure:
    - State: joint positions, velocities, object info, target (46 dims)
    - Vision (Low-Res): CNN features from 80×128 RGB+Depth (128 dims)
    - Vision (High-Res Context): Extracted at episode start, cached (64 dims optional)
    - Total: 174+ dims
    """

    cfg: PickupPlaceDirect0203VisionEnvCfg

    def __init__(self, cfg: PickupPlaceDirect0203VisionEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize vision environment with dual-camera CNN encoders."""
        # Call parent init (will call _setup_scene which now adds cameras)
        super().__init__(cfg, render_mode, **kwargs)
        
        # ==================== DUAL ENCODER INITIALIZATION ====================
        
        # 1. Low-Res Encoder (Continuous, 30Hz)
        # Used for real-time proprioceptive feedback
        self.vision_encoder_low = get_vision_encoder(
            encoder_type=self.cfg.vision_encoder_type,
            image_height=80,                          # Low-res height
            image_width=128,                          # Low-res width
            feature_dim=self.cfg.vision_encoder_feature_dim,
            device=str(self.device),
        )
        
        # 2. High-Res Encoder (Manual trigger, 400x640)
        # Used for context extraction at episode start
        # Can share same architecture or use lighter encoder
        self.vision_encoder_high = get_vision_encoder(
            encoder_type="resnet",                    # Use ResNet for high-res
            image_height=400,
            image_width=640,
            feature_dim=64,                           # Smaller feature space for context
            device=str(self.device),
        )
        
        # ==================== CONTEXT CACHE ====================
        # Cache for high-res context features (extracted once per episode)
        self.context_features_cache = torch.zeros(
            (self.num_envs, 64),
            device=self.device,
            dtype=torch.float32
        )
        self.context_features_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Setup debug snapshot directory
        if self.cfg.debug_vision_snapshots:
            self.debug_snapshot_dir = Path(self.cfg.debug_vision_snapshot_dir)
            self.debug_snapshot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VisionEnv] Debug snapshots enabled. Saving to {self.debug_snapshot_dir}")
        
        # ==================== YOLO DETECTOR INITIALIZATION ====================
        # Initialize YOLO detector for debug visualization of detected objects
        self.yolo_detector = None
        if self.cfg.debug_vision_snapshots and getattr(self.cfg, "debug_vision_enable_yolo_visualization", True):
            try:
                from .yolo_detector import YOLODetector
                self.yolo_detector = YOLODetector(
                    model_name=getattr(self.cfg, "yolo_model_name", "yolov8m"),
                    device=getattr(self.cfg, "yolo_device", str(self.device)),
                    conf_threshold=getattr(self.cfg, "yolo_conf_threshold", 0.5)
                )
                print(f"[VisionEnv] YOLO detector initialized for debug visualization")
            except ImportError as e:
                print(f"[VisionEnv] Warning: YOLO initialization failed: {e}")
                self.yolo_detector = None
        
        # Vision statistics tracking
        self._vision_stats = {
            "rgb_low_mean": torch.zeros(self.num_envs, device=self.device),
            "rgb_low_std": torch.zeros(self.num_envs, device=self.device),
            "depth_low_mean": torch.zeros(self.num_envs, device=self.device),
            "depth_low_std": torch.zeros(self.num_envs, device=self.device),
            "rgb_high_mean": torch.zeros(self.num_envs, device=self.device),
            "rgb_high_std": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Track last snapshot step to enforce interval
        if self.cfg.debug_vision_snapshots and getattr(self.cfg, "debug_vision_snapshot_high_res_interval", 0) > 0:
             # Initialize so the first snapshot (Step 0) is valid
            self._last_high_res_snapshot_step = -self.cfg.debug_vision_snapshot_high_res_interval
        else:
            self._last_high_res_snapshot_step = 0
            
        # Initialize episode sum for new visual reward
        self.episode_sums["reward_vision_focus"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        print(f"[VisionEnv] Dual-camera system initialized:")
        print(f"  - Low-Res Encoder: CNN (type={self.cfg.vision_encoder_type}, features={self.cfg.vision_encoder_feature_dim})")
        print(f"  - High-Res Encoder: CNN (type=simple, features=64, manual trigger)")

    def _setup_scene(self):
        """Setup scene with dual cameras."""
        # Call parent setup first
        super()._setup_scene()
        
        # Add low-resolution camera (continuous)
        self.camera_low = TiledCamera(self.cfg.camera_low_cfg)
        self.scene.sensors["camera_low"] = self.camera_low
        print(f"[VisionEnv] Low-Res Camera added: {self.cfg.camera_low_cfg.prim_path} (Tiled)")
        
        # Add high-resolution camera (manual)
        self.camera_high = TiledCamera(self.cfg.camera_high_cfg)
        self.scene.sensors["camera_high"] = self.camera_high
        print(f"[VisionEnv] High-Res Camera added: {self.cfg.camera_high_cfg.prim_path} (Tiled)")

    def _trigger_high_res_capture(self, env_ids: Optional[torch.Tensor] = None):
        """
        Manually trigger high-resolution camera capture and context extraction.
        
        Args:
            env_ids: Environments to capture (None = all)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        if (self.episode_length_buf[env_ids] == 0).any():
            # Note: Physics poses are already set in _reset_idx, but camera buffers are stale.
            self.sim.render()
        
        try:
            # Get high-res RGB and Depth data
            # Use whole batch for snapshots, but slice for processing to ensure correctness/efficiency
            full_rgb_data = self.camera_high.data.output["rgb"]  # (N, 400, 640, 4)
            
            # Slice only requested environments for high-res vision processing
            rgb_data_slice = full_rgb_data[env_ids]
            
            # Check for depth data (required for GraspNet)
            depth_key = "distance_to_image_plane" if "distance_to_image_plane" in self.camera_high.data.output else "depth"
            if depth_key in self.camera_high.data.output:
                full_depth_data = self.camera_high.data.output[depth_key] # (B, 400, 640, 1)
                depth_data_slice = full_depth_data[env_ids]
            else:
                depth_data_slice = torch.zeros(
                    (len(env_ids), rgb_data_slice.shape[1], rgb_data_slice.shape[2], 1),
                    device=rgb_data_slice.device, dtype=torch.float32
                )
                full_depth_data = None
                print("[VisionEnv] Warning: High-res depth not found! Using zeros.")
            
            if rgb_data_slice is None:
                return
            
            # Process high-res image to extract context (ONLY for requested envs)
            context_features = self._process_high_res_vision(rgb_data_slice.clone(), depth_data_slice.clone())
            
            # Store in cache
            self.context_features_cache[env_ids] = context_features
            self.context_features_valid[env_ids] = True
            
            # ===== SAVE SNAPSHOTS FOR CONTACT GRASPNET =====
            # We save immediately here so we capture exactly what was just rendered for the high-res step.
            # This call respects self.cfg.debug_vision_snapshots AND self.cfg.debug_vision_snapshot_high_res
            if self.cfg.debug_vision_snapshots and self.cfg.debug_vision_snapshot_high_res:
                step = self.common_step_counter
                
                # Rate Limiting: Check if enough steps have passed since last snapshot
                interval = getattr(self.cfg, "debug_vision_snapshot_high_res_interval", 0)
                if interval > 0 and (step - self._last_high_res_snapshot_step) < interval:
                    # Skip saving if interval hasn't passed
                    pass
                else:
                    # Proceed to save
                    saved_any = False
                    # Iterate through configured debug environments
                    for env_id in self.cfg.debug_vision_snapshot_envs:
                        # Check if updated env_id is valid and in current batch
                        if env_id < self.num_envs and env_id in env_ids:
                            # Pass current data to avoid re-fetching (use full batch data for global indexing)
                            self._save_debug_snapshots(step, camera_type="high", env_id=env_id, 
                                                       rgb_data=full_rgb_data, depth_data=full_depth_data)
                            # Also save low-res snapshot at the same time
                            self._save_debug_snapshots(step, camera_type="low", env_id=env_id)
                            saved_any = True
                    
                    if saved_any:
                        self._last_high_res_snapshot_step = step

        except Exception as e:
            print(f"[VisionEnv] Warning: High-res capture failed: {e}")


    def _process_vision_data(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Process low-res RGB and Depth data (80×128) into CNN features.
        
        This is called every step for continuous proprioceptive feedback.
        
        Args:
            rgb: RGB image tensor (B, 80, 128, 3) with values in [0, 255]
            depth: Depth image tensor (B, 80, 128, 1) with distance values
        
        Returns:
            CNN features (B, 128)
        """
        # 效能優化：如果環境設定為輸出 Raw Data 交給 Policy 訓練，
        # 則 Env 端不需要執行 CNN Inference，直接回傳零向量以節省算力。
        if getattr(self.cfg, "use_raw_observations", False):
            return torch.zeros((rgb.shape[0], self.cfg.vision_encoder_feature_dim), 
                               device=rgb.device, dtype=torch.float32)

        # ===== RGB Normalization =====
        rgb_norm = rgb[..., :3] / 255.0
        rgb_norm = torch.clamp(rgb_norm, 0.0, 1.0)
        
        # ===== Depth Normalization =====
        # Clip depth to desired range (0.2 - 2.5m) because camera render range is wider (0.1 - 100.0m)
        depth_clipped = torch.clamp(depth, 0.2, 2.5)
        depth_norm = depth_clipped / 2.3  # Max clipping distance is 2.5m
        depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
        
        # ===== Permute to CNN format: (B, H, W, C) -> (B, C, H, W) =====
        rgb_chw = rgb_norm.permute(0, 3, 1, 2).float()
        
        # ===== Extract features using CNN =====
        with torch.no_grad():
            features = self.vision_encoder_low(rgb_chw)  # (B, 128)
        
        return features

    def _process_high_res_vision(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Process high-res RGB and Depth data (400×640) into context features.
        
        This provides detailed static features for scene understanding.
        
        Args:
            rgb: RGB image tensor (B, 400, 640, 3) with values in [0, 255]
            depth: Depth image tensor (B, 400, 640, 1) with distance values
        
        Returns:
            Context features (B, 64)
        """
        # ===== RGB Normalization =====
        rgb_norm = rgb[..., :3] / 255.0
        rgb_norm = torch.clamp(rgb_norm, 0.0, 1.0)
        
        # ===== Depth Normalization =====
        # Normalize depth to [0, 1] range (same as low-res for consistency)
        depth_clipped = torch.clamp(depth, 0.2, 2.5)
        depth_norm = depth_clipped / 2.3  # Max clipping distance is 2.5m
        depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
        
        # ===== Permute to CNN format =====
        rgb_chw = rgb_norm.permute(0, 3, 1, 2).contiguous().float()     # (B, 3, 400, 640)
        
        # ===== Extract features using CNN =====
        with torch.no_grad():
            features = self.vision_encoder_high(rgb_chw)  # (B, 64)

        
        return features

    def _draw_yolo_detections(self, rgb_image: torch.Tensor, rgb_h: int, rgb_w: int) -> torch.Tensor:
        """
        Draw YOLO detections on RGB image for visualization.
        
        Args:
            rgb_image: (H, W, 3) RGB image tensor in range [0, 1]
            rgb_h: Image height
            rgb_w: Image width
        
        Returns:
            (H, W, 3) RGB image with drawn bounding boxes and keypoints
        """
        if self.yolo_detector is None:
            return rgb_image
        
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("[VisionEnv] Warning: OpenCV not available for YOLO visualization")
            return rgb_image
        
        # Convert to numpy uint8 for OpenCV
        rgb_np = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
        
        # Ensure proper shape (H, W, 3)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        if rgb_np.ndim != 3 or rgb_np.shape[-1] != 3:
            return rgb_image
        
        # Run YOLO detection
        try:
            # Add batch dimension for detection
            rgb_batch = torch.from_numpy(rgb_np).unsqueeze(0).to(self.device)
            detections_list = self.yolo_detector.detect_batch(rgb_batch)
            detections = detections_list[0]
        except Exception as e:
            print(f"[VisionEnv] YOLO detection failed: {e}")
            return rgb_image
        
        if detections is None or detections["num_detections"] == 0:
            # No detections, add text to image
            cv2.putText(rgb_np, "No objects detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return torch.from_numpy(rgb_np).float() / 255.0
        
        # Draw bounding boxes
        boxes = detections["boxes"]  # (N, 4) format [x1, y1, x2, y2]
        confidences = detections["confidences"]  # (N,)
        
        # Colors for different confidences (green = high, red = low)
        for box_idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, rgb_w - 1))
            y1 = max(0, min(y1, rgb_h - 1))
            x2 = max(0, min(x2, rgb_w - 1))
            y2 = max(0, min(y2, rgb_h - 1))
            
            # Color based on confidence (gradient from red to green)
            confidence_normalized = float(conf)
            b = int(255 * (1 - confidence_normalized))
            g = int(255 * confidence_normalized)
            r = 0
            color = (b, g, r)
            
            # Draw bounding box rectangle
            cv2.rectangle(rgb_np, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence text
            label = f"Conf: {confidence_normalized:.2f}"
            cv2.putText(rgb_np, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(rgb_np, (cx, cy), 3, color, -1)
            
            # Draw corner points
            corner_size = 4
            for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.circle(rgb_np, (px, py), corner_size, color, -1)
        
        # Add detection count
        det_text = f"Detections: {detections['num_detections']}"
        cv2.putText(rgb_np, det_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return torch.from_numpy(rgb_np).float() / 255.0

    def _save_debug_snapshots(self, step: int, camera_type: str = "low", env_id: int = 0, 
                              rgb_data: Optional[torch.Tensor] = None, depth_data: Optional[torch.Tensor] = None):
        """
        Save RGB and Depth snapshots for debugging.
        
        Args:
            step: Current step number
            camera_type: "low" or "high" camera
            env_id: Which environment to snapshot
            rgb_data: Optional pre-fetched RGB tensor (N, H, W, C)
            depth_data: Optional pre-fetched Depth tensor (N, H, W, 1)
        """
        if not self.cfg.debug_vision_snapshots:
            return
        
        # Skip high-res snapshots if disabled
        if camera_type == "high" and not self.cfg.debug_vision_snapshot_high_res:
            return
        
        try:
            # Create subdirectory for this step
            step_dir = self.debug_snapshot_dir / f"step_{step:08d}" / camera_type
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Select camera
            if camera_type == "low":
                camera = self.camera_low
            else:
                camera = self.camera_high
            
            # 1. Resolve RGB Batch Data
            if rgb_data is None:
                rgb_data = camera.data.output["rgb"] # (N, H, W, C)
            
            # 2. Resolve Depth Batch Data
            if depth_data is None:
                if "distance_to_image_plane" in camera.data.output:
                    depth_data = camera.data.output["distance_to_image_plane"]
                elif "depth" in camera.data.output:
                    depth_data = camera.data.output["depth"]
            
            # 3. Extract Env-Specific RGB
            env_rgb = rgb_data[env_id].clone()
            
            # Remove alpha channel if present
            if env_rgb.shape[-1] == 4:
                env_rgb = env_rgb[..., :3]
            
            # Normalize to [0, 1]
            rgb_normalized = env_rgb.float() / 255.0
            rgb_normalized = torch.clamp(rgb_normalized, 0.0, 1.0)
            
            # ===== DRAW YOLO DETECTIONS ON RGB =====
            rgb_with_yolo = rgb_normalized.clone()
            if self.yolo_detector is not None:
                try:
                    rgb_with_yolo = self._draw_yolo_detections(
                        rgb_normalized, 
                        rgb_h=env_rgb.shape[0], 
                        rgb_w=env_rgb.shape[1]
                    )
                except Exception as e:
                    print(f"[VisionEnv] Warning: Failed to draw YOLO detections: {e}")
            
            # Save RGB
            try:
                from PIL import Image
                import numpy as np
                
                # Save original RGB
                rgb_np = (rgb_normalized.cpu().numpy() * 255).astype(np.uint8)
                rgb_img = Image.fromarray(rgb_np)
                rgb_path = step_dir / "rgb.png"
                rgb_img.save(str(rgb_path))
                
                # Save RGB with YOLO detections
                if self.yolo_detector is not None:
                    rgb_yolo_np = (rgb_with_yolo.cpu().numpy() * 255).astype(np.uint8)
                    rgb_yolo_img = Image.fromarray(rgb_yolo_np)
                    rgb_yolo_path = step_dir / "rgb_with_yolo.png"
                    rgb_yolo_img.save(str(rgb_yolo_path))
            except ImportError:
                rgb_path = step_dir / "rgb.pt"
                torch.save(rgb_normalized, str(rgb_path))
                if self.yolo_detector is not None:
                    rgb_yolo_path = step_dir / "rgb_with_yolo.pt"
                    torch.save(rgb_with_yolo, str(rgb_yolo_path))
            
            # 4. Extract and Save Env-Specific Depth (if available)
            if depth_data is not None:
                env_depth = depth_data[env_id].clone()
                
                # ===== Save Raw Depth (Metric Float32) for Contact GraspNet =====
                try:
                    # Handle Infinity (Sky/Far clip) -> 0.0
                    precise_depth = env_depth.clone()
                    precise_depth[torch.isinf(precise_depth)] = 0.0
                    
                    # Convert to Numpy Float32 (Meters)
                    precise_depth_np = precise_depth.cpu().numpy().astype(np.float32)
                    
                    # Squeeze dimensions: (H, W, 1) -> (H, W)
                    if precise_depth_np.ndim == 3:
                        precise_depth_np = precise_depth_np.squeeze(-1)
                        
                    # Save as .npy
                    npy_path = step_dir / "depth_raw.npy"
                    np.save(str(npy_path), precise_depth_np)
                    
                except Exception as e:
                    print(f"[VisionEnv] Warning: Failed to save raw depth .npy: {e}")

                # ===== Save Normalized Depth (Visualization) =====
                # Handle Inf/NaN
                depth_valid = env_depth.clone()
                depth_valid[torch.isinf(depth_valid)] = 0.0
                depth_valid[torch.isnan(depth_valid)] = 0.0

                depth_min = depth_valid.min()
                depth_max = depth_valid.max()
                if depth_max > depth_min:
                    depth_norm = (depth_valid - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = torch.zeros_like(depth_valid)
                
                depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
                
                try:
                    from PIL import Image
                    import numpy as np
                    
                    depth_np = (depth_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    depth_img = Image.fromarray(depth_np, mode='L')
                    depth_path = step_dir / "depth.png"
                    depth_img.save(str(depth_path))
                except ImportError:
                    depth_path = step_dir / "depth.pt"
                    torch.save(depth_norm, str(depth_path))
            
            # Save metadata
            metadata_path = step_dir / "metadata.txt"
            with open(str(metadata_path), 'w') as f:
                f.write(f"Step: {step}\n")
                f.write(f"Environment ID: {env_id}\n")
                f.write(f"Camera: {camera_type}\n")
                f.write(f"RGB Shape: {env_rgb.shape}\n")
                f.write(f"RGB Range: [{rgb_normalized.min():.4f}, {rgb_normalized.max():.4f}]\n")
            
        except Exception as e:
            print(f"[VisionEnv] Warning: Failed to save {camera_type} snapshots at step {step}: {e}")

    def _get_observations(self) -> dict:
        """
        Collect observations combining state and dual-camera vision features.
        
        Returns:
        - State observations (46 dims)
        - Low-Res vision features (128 dims from 80×128 RGB+Depth)
        - Context features cached from high-res camera (64 dims)
        -------
        Total: 174 + 64 = 238 dims (configurable)
        """
        # Get base state observations
        base_obs = super()._get_observations()
        policy_obs = base_obs["policy"]  # (B, 46)
        
        # ===== LOW-RES VISION (Continuous, 30Hz) =====
        try:
            # Get low-res RGB and Depth
            # TiledCamera: (N, H, W, C)
            rgb_low = self.camera_low.data.output["rgb"]      # (N, 80, 128, 4)
            depth_low = self.camera_low.data.output["depth"]  # (N, 80, 128, 1)
            
            # Process through CNN
            vision_features_low = self._process_vision_data(rgb_low, depth_low)  # (B, 128)
            
        except Exception as e:
            print(f"[VisionEnv] Warning: Low-res camera unavailable: {e}")
            vision_features_low = torch.zeros(
                (self.num_envs, self.cfg.vision_encoder_feature_dim),
                device=self.device,
                dtype=policy_obs.dtype
            )
        
        # ===== HIGH-RES CONTEXT (Manual/Episode-start) =====
        # Context features are cached and only updated manually
        vision_features_high = self.context_features_cache.clone()
        
        # ===== DEBUG SNAPSHOTS =====
        # Low-res snapshots every N steps
        if (self.cfg.debug_vision_snapshots and 
            self.common_step_counter % self.cfg.debug_vision_snapshot_interval == 0):
            for env_id in self.cfg.debug_vision_snapshot_envs:
                if env_id < self.num_envs:
                    self._save_debug_snapshots(self.common_step_counter, camera_type="low", env_id=env_id)
        

        
        # ===== COMBINE OBSERVATIONS =====
        # Concatenate: state (46) + low-res features (128) + high-res context (64)
        combined_obs = torch.cat([
            policy_obs,              # (B, 46)
            vision_features_low,     # (B, 128)
            vision_features_high     # (B, 64)
        ], dim=-1)  # → (B, 238)
        
        return {"policy": combined_obs}



    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards, extending base rewards with visual focus.
        """
        # Get base rewards (state-based)
        total_reward = super()._get_rewards()
        
        # Add Visual Focus Reward
        scale = getattr(self.cfg, "rew_scale_vision_focus", 0.0)
        
        if scale != 0.0:
            focus_reward = mdp_rewards.object_in_view_reward(
                self, 
                camera_sensor_name="camera_low", 
                object_cfg=SceneEntityCfg("object"),
                focus_exponent=2.0
            )
            
            # Add to total
            total_reward += scale * focus_reward
            
            # Log Episode Sums
            if "reward_vision_focus" in self.episode_sums:
                self.episode_sums["reward_vision_focus"] += scale * focus_reward
            
            # Log current step value to extras
            # OPTIMIZATION: Removed .item() to avoid CPU sync
            if hasattr(self, "extras") and "episode" in self.extras:
                self.extras["episode"]["last_vision_focus"] = torch.mean(focus_reward)
            
        return total_reward

    def _reset_idx(self, env_ids):
        """
        Reset selected environments and trigger high-res context capture.
        
        Overrides parent method to add high-resolution camera trigger
        for context extraction at episode start.
        """
        # Call parent reset
        super()._reset_idx(env_ids)
        
        # Trigger high-res camera if configured to capture at Step 0 (Reset)
        if self.cfg.high_res_capture_step == 0:
            try:
                self._trigger_high_res_capture(env_ids)
            except Exception as e:
                print(f"[VisionEnv] Note: High-res capture at reset: {e}")

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment and trigger high-res capture if needed.
        """
        # Trigger high-res camera for environments reaching the configured step
        # We check this before the step (using current buf) or after?
        # episode_length_buf increments in super().step() usually (or post_physics).
        # Let's check before calling super().step() to match the logic of "at step N".
        # But wait, step() is called for all envs. We need to find which envs match.
        
        # NOTE: This runs for all environments, but we only trigger for specific ones.
        if self.cfg.high_res_capture_step > 0:
            # Find envs that are exactly at the capture step
            # Note: We use the buffer directly. 
            # If high_res_capture_step is 10, we want to capture when episode_length_buf is 10.
            # This happens once per episode.
            trigger_ids = torch.nonzero(self.episode_length_buf == self.cfg.high_res_capture_step, as_tuple=True)[0]
            
            if len(trigger_ids) > 0:
                # print(f"[VisionEnv] Triggering high-res capture for {len(trigger_ids)} envs at step {self.cfg.high_res_capture_step}")
                self._trigger_high_res_capture(trigger_ids)
                
        return super().step(actions)



 