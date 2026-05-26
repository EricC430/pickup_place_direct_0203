# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections import deque
from typing import Dict, Optional
from pathlib import Path
import os

from .pickup_place_direct_0203_vision_env import PickupPlaceDirect0203VisionEnv
from .pickup_place_vision_asym_0318_env_cfg import PickupPlaceVisionAsym0318EnvCfg
from .utils.vision_encoder import PointNetEncoder
from .mdp import observations as mdp_obs
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .utils.performance_monitor import get_perf_monitor
import torch.nn as nn
# from .diagnostic_utils import DiagnosticProbe

class EmpiricalNormalizer(nn.Module):
    """
    Normalizes observations using running mean and variance.
    Matches the Behavior Cloning (BC) empirical normalizer.
    """
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1.0))
        self.epsilon = 1e-8

    def forward(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)


class PickupPlaceVisionAsym0318Env(PickupPlaceDirect0203VisionEnv):
    """
    Asymmetric Direct RL environment with dual-camera vision system and point cloud detection. (0318 Version)
    """

    cfg: PickupPlaceVisionAsym0318EnvCfg

    def __init__(self, cfg: PickupPlaceVisionAsym0318EnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize asymmetric vision environment with Point Cloud.
        
        Args:
            cfg: Configuration for the asymmetric vision environment
            render_mode: Optional render mode for visualization
            **kwargs: Additional arguments for parent class
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Tracking BC Normalization
        self._bc_normalization_loaded = False

        # Initialize Normalizers and LayerNorms with DEFAULTS so they always exist.
        # This allows "same preprocessing" logic even without BC weights (they will be identity).
        self.proprio_norm = EmpiricalNormalizer(42).to(self.device)
        self.vision_low_ln = nn.LayerNorm(512).to(self.device)
        self.pointnet_ln = nn.LayerNorm(512).to(self.device)
        self.vision_high_ln = nn.LayerNorm(64).to(self.device)
        self.proprio_norm.eval()
        self.vision_low_ln.eval()
        self.pointnet_ln.eval()
        self.vision_high_ln.eval()
        

        # ========== MULTI-FRAME CNN FEATURE BUFFER ==========
        # Low-Res Camera: Stack 4 frames for temporal dynamics
        # High-Res Camera: Single frame (fixed context for entire episode)
        # 
        # Reasoning:
        # - Low-res: 4 frames × 128 dims = 512 dims (captures motion & dynamics)
        # - High-res: 1 frame × 64 dims = 64 dims (static scene context, same throughout episode)
        # Total vision: 512 + 64 = 576 dims
        
        self.history_length = 13  # Keep 13 frames for strided history (t, t-4, t-8, t-12)
        self.cnn_feature_history_buf = torch.zeros((self.num_envs, self.history_length, 128), device=self.device, dtype=torch.float32)
        
        # Raw Data Buffers for Trainable Encoders (if enabled)
        if getattr(self.cfg, "use_raw_observations", False):
            # 4 frames of RGBD 80x128 = 4 * (128*80*4) = 163840
            self.raw_image_history_buf = torch.zeros((self.num_envs, self.history_length, 4, 80, 128), device=self.device, dtype=torch.float32)
            self.raw_pt_history_buf = torch.zeros((self.num_envs, self.history_length, 1024, 3), device=self.device, dtype=torch.float32)
            print("[INFO] Raw observations enabled. VRAM usage for history buffers will be high.")


        # High-res context cache (single frame per episode, not multi-frame)
        self.high_res_context_single = None
        self.high_res_context_valid = False
        
        # Action History Buffer 
        self.action_history_buf = torch.zeros((self.num_envs, 4, 6), device=self.device, dtype=torch.float32)
        
        # PointNet
        self.pointnet = PointNetEncoder(out_dim=128).to(self.device)
        self.pt_feature_history_buf = torch.zeros((self.num_envs, self.history_length, 128), device=self.device, dtype=torch.float32)
        
        # ========== JOINT POSITION HISTORY FOR VELOCITY ESTIMATION ==========
        # Store joint positions for numerical differentiation of velocity
        # Using deque with maxlen=2 to keep only current and previous positions
        self.joint_pos_history = deque(maxlen=2)
        
        # ========== VELOCITY ESTIMATION CACHE ==========
        # Cache for simulated joint velocity (computed via numerical differentiation)
        self.jvel_simulated = torch.zeros(
            (self.num_envs, 6), # Fixed: 5 -> 6 (5 arm + 1 gripper)
            device=self.device,
            dtype=torch.float32
        )
        self.use_simulated_jvel = False  # Toggle flag to switch between simulator and computed velocity
        
        # ========== AUTO-LOAD PRE-TRAINED WEIGHTS ==========
        # If the config specifies a path, load vision encoders and normalizers now.
        if hasattr(self.cfg, "vision_weights_path") and self.cfg.vision_weights_path is not None:
            print(f"[INFO] Automatically loading vision/norm weights from: {self.cfg.vision_weights_path}")
            # Use torch.load on the configured file
            try:
                # Need to handle potential device mismatch (config might be on CPU)
                weights = torch.load(self.cfg.vision_weights_path, map_location=self.device)
                self.load_bc_normalization_and_encoders(weights)
            except Exception as e:
                print(f"[ERROR] Failed to auto-load vision weights from {self.cfg.vision_weights_path}: {e}")
        
        # Set encoders to training mode if using raw observations (to be trained by RL)
        if getattr(self.cfg, "use_raw_observations", False):
             # These are for the environment-side pre-processing if still needed,
             # but usually for training we'll want to use the encoders in the Policy.
             # Setting to train() ensures gradients can flow if we optimize them.
             self.vision_encoder_low.train()
             self.pointnet.train()
             print("[INFO] Vision encoders set to train() mode for RL fine-tuning.")
        
        # # Initialize performance monitor for tracking bottlenecks
        # self.perf_monitor = get_perf_monitor()
        # self.perf_monitor.set_device(self.device)
        # self.perf_log_interval = 100  # Log performance every N steps

        # ========== 0318 DEBUG SNAPSHOT INITIALIZATION ==========
        if getattr(self.cfg, "debug_vision_snapshot_0318_enable", False):
            self.snapshot_episodes_done = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.debug_snapshot_0318_dir = Path(self.cfg.debug_vision_snapshot_0318_dir)
            self.debug_snapshot_0318_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VisionAsym0318] 0318 Debug snapshots enabled. Saving to {self.debug_snapshot_0318_dir}")

        # [DEBUG] Initialize Diagnostic Probe
        # self.diagnostic_probe = DiagnosticProbe(self)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.action_history_buf[env_ids] = 0.0
            if hasattr(self, 'pt_feature_history_buf'):
                self.pt_feature_history_buf[env_ids] = 0.0
            if hasattr(self, 'cnn_feature_history_buf'):
                self.cnn_feature_history_buf[env_ids] = 0.0
            if hasattr(self, 'raw_image_history_buf'):
                self.raw_image_history_buf[env_ids] = 0.0
            if hasattr(self, 'raw_pt_history_buf'):
                self.raw_pt_history_buf[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self.action_history_buf = torch.roll(self.action_history_buf, shifts=-1, dims=1)
        self.action_history_buf[:, -1, :] = actions.clone()

    def load_bc_normalization_and_encoders(self, bc_state_dict: dict):
        """
        Loads the pre-trained BC vision encoders and normalizers into the environment.
        This ensures the RL Agent (PPO) receives the exact same observation scale.
        """
        # 1. Load EmpiricalNormalizer
        self.proprio_norm.load_state_dict({
            'running_mean': bc_state_dict['proprio_norm.running_mean'].to(self.device),
            'running_var': bc_state_dict['proprio_norm.running_var'].to(self.device),
            'count': bc_state_dict['proprio_norm.count'].to(self.device)
        })
        
        # 2. Load LayerNorms
        self.vision_low_ln.load_state_dict({
            'weight': bc_state_dict['vision_low_ln.weight'].to(self.device),
            'bias': bc_state_dict['vision_low_ln.bias'].to(self.device)
        })
        self.vision_low_ln.eval()
        
        # [OPTIM] CUDA OOM 0401
        # self.pointnet_ln = nn.LayerNorm(512).to(self.device)
        # 【GPU MEMORY OPTIMIZATION】Reuse already-initialized instances instead of recreating
        # This avoids redundant model instantiation and reduces memory fragmentation
        self.pointnet_ln.load_state_dict({
            'weight': bc_state_dict['pointnet_ln.weight'].to(self.device),
            'bias': bc_state_dict['pointnet_ln.bias'].to(self.device)
        })
        self.pointnet_ln.eval()
        
        # [OPTIM] CUDA OOM 0401
        # self.vision_high_ln = nn.LayerNorm(64).to(self.device)
        self.vision_high_ln.load_state_dict({
            'weight': bc_state_dict['vision_high_ln.weight'].to(self.device),
            'bias': bc_state_dict['vision_high_ln.bias'].to(self.device)
        })
        self.vision_high_ln.eval()
        
        # 3. Load Vision Encoder Weights
        vision_low_dict = {k.replace('vision_encoder_low.', ''): v 
                           for k, v in bc_state_dict.items() if k.startswith('vision_encoder_low.')}
        if vision_low_dict:
            self.vision_encoder_low.load_state_dict(vision_low_dict)
            self.vision_encoder_low.eval()
            print("[INFO] Successfully loaded BC vision_encoder_low into environment.")
            
        pointnet_dict = {k.replace('pointnet.', ''): v 
                         for k, v in bc_state_dict.items() if k.startswith('pointnet.')}
        if pointnet_dict:
            self.pointnet.load_state_dict(pointnet_dict)
            self.pointnet.eval()
            print("[INFO] Successfully loaded BC pointnet into environment.")
            
        self._bc_normalization_loaded = True
        print("[INFO] BC Normalization and Encoders successfully loaded into environment.")


    def _get_observations(self) -> dict:
        """
        Collect asymmetric observations including multi-frame CNN features and Point Cloud.
        """
        # [0402 GPU OOM FIX] Removed per-step self.sim.render().
        # Isaac Lab's step() already calls render at the configured render_interval.
        # Calling it again here caused double-rendering every step, leading to Vulkan
        # MemoryManager chunk fragmentation and OOM after ~5 hours of training.
        # Note: _trigger_high_res_capture() in parent still calls sim.render() at
        # episode Step 0 to ensure fresh camera data for high-res snapshots.

        # # Start performance tracking
        # self.perf_monitor.start_timer("_get_observations_total")
        # self.perf_monitor.start_timer("parent_observations")
        
        # Get the full observations from the parent class (VisionEnv)
        full_obs_dict = super()._get_observations()
        full_obs = full_obs_dict["policy"]  # Shape: (B, 240)
        
        # self.perf_monitor.end_timer("parent_observations")
        
        # # ========== EXTRACT AND BUFFER MULTI-FRAME CNN FEATURES ==========
        # self.perf_monitor.start_timer("cnn_feature_extraction")
        
        # Separate low-res (128 dims) and high-res (64 dims) from parent observation
        # Parent obs structure: [State(48) | LowRes(128) | HighRes(64)]
        current_cnn_features_low = full_obs[:, 48:176]    # 128 dims (low-res continuous)
        current_cnn_features_high = full_obs[:, 176:240]  # 64 dims (high-res context)
        
        # ===== LOW-RES: Multi-frame stacking (4 strided frames for temporal dynamics) =====
        # Update 13-frame strided history buffer
        self.cnn_feature_history_buf = torch.roll(self.cnn_feature_history_buf, shifts=-1, dims=1)
        self.cnn_feature_history_buf[:, -1, :] = current_cnn_features_low.clone()
        
        # Extract strided frames: t-12 (idx 0), t-8 (idx 4), t-4 (idx 8), t (idx 12)
        strided_cnn = self.cnn_feature_history_buf[:, [0, 4, 8, 12], :] # Shape: (B, 4, 128)
        
        # Flatten multi-frame low-res features: (B, 4*128) = (B, 512)
        vision_low_multiframe = strided_cnn.reshape(
            self.num_envs, -1
        )  # Shape: (B, 512) [4 strided frames * 128 dims/frame]
        
        # ===== HIGH-RES: Single frame per episode (static context) =====
        # Store high-res context only once and reuse throughout episode
        if not self.high_res_context_valid:
            # First time: cache the high-res context
            self.high_res_context_single = current_cnn_features_high.clone()
            self.high_res_context_valid = True
        
        # Use cached high-res context (same throughout episode)
        vision_high_single = self.high_res_context_single.clone()
        
        # self.perf_monitor.end_timer("cnn_feature_extraction")
        
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
            # Physics dt = 0.01s, decimation = 2 -> Control step dt = 0.02s
            self.jvel_simulated = (jpos_curr - jpos_prev) / 0.02
        else:
            self.jvel_simulated = torch.zeros((self.num_envs, 6), device=self.device)
        
        # Get JVel from full observation (simulator's velocity)
        jvel_from_sim = full_obs[:, 6:12]  # Shape: (B, 6)
        
        # Choose which JVel to use (option: can add option to toggle in config)
        jvel_to_use = self.jvel_simulated if self.use_simulated_jvel else jvel_from_sim
        
        # ========== POINTNET FEATURE PROCESSING ==========
        # [OPTIM] 0401【GPU MEMORY OPTIMIZATION】Wrap entire point cloud computation in torch.no_grad()
        # This prevents gradients from being tracked for point cloud generation operations
        # which don't require backpropagation, reducing VRAM usage by ~50-60 MB
        # self.perf_monitor.start_timer("pointnet_extraction")
        with torch.no_grad():
            if "depth" in self.camera_low.data.output:
                depth_img = self.camera_low.data.output["depth"]
            elif "distance_to_image_plane" in self.camera_low.data.output:
                depth_img = self.camera_low.data.output["distance_to_image_plane"]
            else:
                depth_img = torch.zeros((self.num_envs, 80, 128, 1), device=self.device)
                
            B, H, W, _ = depth_img.shape
            device = depth_img.device
            
            # Unproject depth to point cloud
            intrinsics = self.camera_low.data.intrinsic_matrices
            cam_pos_w = self.camera_low.data.pos_w
            cam_quat_w = self.camera_low.data.quat_w_world
            
            from isaaclab.utils.math import matrix_from_quat
            cam_rot_w = matrix_from_quat(cam_quat_w) # (B, 3, 3)
            
            extrinsics = torch.zeros((B, 4, 4), device=device)
            extrinsics[:, :3, :3] = cam_rot_w
            extrinsics[:, :3, 3] = cam_pos_w
            extrinsics[:, 3, 3] = 1.0
            
            v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            u = u.expand(B, H, W).float()
            v = v.expand(B, H, W).float()
            
            fx = intrinsics[:, 0, 0].view(B, 1, 1)
            fy = intrinsics[:, 1, 1].view(B, 1, 1)
            cx = intrinsics[:, 0, 2].view(B, 1, 1)
            cy = intrinsics[:, 1, 2].view(B, 1, 1)
            
            depth = depth_img.squeeze(-1)
            
            z_c = depth
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy
            points_c = torch.stack([x_c, y_c, z_c], dim=-1).view(B, -1, 3)
            
            # Transform points_c to world frame to filter by height
            # points_c: (B, 10240, 3)
            # extrinsics[:, :3, :3]: (B, 3, 3) 
            # extrinsics[:, :3, 3]: (B, 3)
            cam_rot = extrinsics[:, :3, :3]
            cam_pos = extrinsics[:, :3, 3].unsqueeze(1) # (B, 1, 3)
            points_w_flat = torch.matmul(points_c, cam_rot.transpose(1, 2)) + cam_pos
            
            valid_mask = (points_c[..., 2] > 0.1) & (points_c[..., 2] < 1.2) & (points_w_flat[..., 2] > 0.005)
            
            num_samples = 1024
            ptcloud = torch.zeros((B, num_samples, 3), device=device)
            for i in range(B):
                valid_pts = points_c[i][valid_mask[i]]
                if len(valid_pts) >= num_samples:
                    idx = torch.randperm(len(valid_pts), device=device)[:num_samples]
                    ptcloud[i] = valid_pts[idx]
                elif len(valid_pts) > 0:
                    idx = torch.randint(0, len(valid_pts), (num_samples,), device=device)
                    ptcloud[i] = valid_pts[idx]
                    
            # Keep Point Cloud in Camera Frame! This is crucial for PointNet generalization.
            self.current_ptcloud = ptcloud
        
        # Optimization: Skip local PointNet inference if we are passing raw data to Trainable Encoders
        if not getattr(self.cfg, "use_raw_observations", False):
            with torch.no_grad():
                pt_feat = self.pointnet(ptcloud) # (B, 128)
            
            # Update 13-frame history
            self.pt_feature_history_buf = torch.roll(self.pt_feature_history_buf, shifts=-1, dims=1)
            self.pt_feature_history_buf[:, -1, :] = pt_feat
        
        # Extract strided frames: t-12 (idx 0), t-8 (idx 4), t-4 (idx 8), t (idx 12)
        strided_pt = self.pt_feature_history_buf[:, [0, 4, 8, 12], :] # Shape: (B, 4, 128)
        
        pointnet_history = strided_pt.view(self.num_envs, -1) # (B, 512)
        # self.perf_monitor.end_timer("pointnet_extraction")
        
        # ========== BUILD POLICY OBSERVATION ==========
        # self.perf_monitor.start_timer("construct_policy_obs")
        
        # JPos / JVel / JErr
        jpos_full = self.joint_pos[:, list(self._arm_joint_indices) + list(self._gripper_joint_idx)] - self.robot.data.default_joint_pos[:, list(self._arm_joint_indices) + list(self._gripper_joint_idx)]
        jvel_full = self.joint_vel[:, list(self._arm_joint_indices) + list(self._gripper_joint_idx)] - self.robot.data.default_joint_vel[:, list(self._arm_joint_indices) + list(self._gripper_joint_idx)]
        # [0412 ANTI-PENETRATION] Clamp all proprio signals to filter simulation explosion artifacts.
        # CCD + solve_articulation_contact_last can cause the PhysX solver to produce unrecoverable
        # articulation states where joint positions spiral to 11k-23k+ rad (1800+ rotations).
        # These are simulation "death spirals", not transient impulses — once they occur, the entire
        # environment state is corrupt until reset. Clamping prevents these from corrupting observations.
        #
        # Clamp values derived from actuator specs and joint geometry:
        #   jpos_full: ±2π rad — no joint can deviate more than 1 full rotation from default
        #              (actual joint ranges are much smaller, e.g. ±2.09 rad for arm)
        #   jvel_full: ±20 rad/s — 3x max actuator velocity (6.54 rad/s gripper)
        jpos_full = torch.clamp(jpos_full, min=-6.2832, max=6.2832)  # ±2π rad
        jvel_full = torch.clamp(jvel_full, min=-20.0, max=20.0)
        
        # [FIX] prev_actions 應該使用剛剛下達的 current action (t)，而不是上一張 (t-1)
        # 因為我們想衡量的是：機器人朝向這個目標移動後的物理殘差。
        prev_actions = self.action_history_buf[:, -1, :] if self.action_history_buf.shape[1] >= 1 else self.actions
        
        # [FIX] 將 raw actions 正確轉換為 target joint position 以計算真實物理誤差 (jerr)
        scaled_prev_actions = prev_actions * self.cfg.action_scale
        arm_offsets = torch.tensor(self.cfg.action_cfg["arm_offsets"], device=self.device)
        arm_scale = self.cfg.action_cfg["arm_scale"]
        prev_arm_targets = scaled_prev_actions[:, :5] * arm_scale + arm_offsets
        # [0402 Numerical Safety] Sync with _apply_action (arm_scale=2.09)
        prev_arm_targets = torch.clamp(prev_arm_targets, min=-2.09, max=2.09)

        gripper_scale = self.cfg.action_cfg["gripper_scale"]
        gripper_offset = self.cfg.action_cfg["gripper_offset"]
        prev_gripper_targets = scaled_prev_actions[:, 5] * gripper_scale + gripper_offset
        # [0402 Numerical Safety] Sync with _apply_action (offset=0.785, scale=0.785 -> [0, 1.57])
        prev_gripper_targets = torch.clamp(prev_gripper_targets, min=0.0, max=1.57)
        
        prev_targets_full = torch.cat([prev_arm_targets, prev_gripper_targets.unsqueeze(1)], dim=1)
        # 計算與 default_joint_pos 之間的相對目標，以對齊 jpos_full
        prev_targets_relative = prev_targets_full - self.robot.data.default_joint_pos[:, list(self._arm_joint_indices) + list(self._gripper_joint_idx)]
        jerr = prev_targets_relative - jpos_full
        
        # [0402 Numerical Stability] Mask JErr at Step 0.
        # Since Step 0 has no real "previous action", we zero it to prevent confusion from random initialization.
        jerr[self.episode_length_buf == 0] = 0.0
        
        last_4_actions = self.action_history_buf.view(self.num_envs, -1) # (B, 24)
        
        # Proprio block: JPos(6) + JVel(6) + JErr(6) + Last4Actions(24) = 42
        proprio_obs = torch.cat([jpos_full, jvel_full, jerr, last_4_actions], dim=-1)
        
        # ========== BUILD CRITIC OBSERVATION EARLY ==========
        # Must build critic_obs BEFORE returning raw observations
        if hasattr(self, 'ee_frame'):
            ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]
            ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :]
            
            obj_pos_w = self.scene["object"].data.root_com_pose_w[:, :3]
            obj_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, obj_pos_w)
            
            target_pos_w = self.target_poses + self.scene.env_origins 
            target_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, target_pos_w)
            
            world_corners_flat = mdp_obs.object_bbox_corners(self, SceneEntityCfg("object")).view(self.num_envs * 8, 3)
            ee_pos_rep = ee_pos_w.repeat_interleave(8, dim=0)
            ee_quat_rep = ee_quat_w.repeat_interleave(8, dim=0)
            bbox_ee_flat, _ = subtract_frame_transforms(ee_pos_rep, ee_quat_rep, world_corners_flat)
            bbox_ee = bbox_ee_flat.view(self.num_envs, 24)
        else:
            obj_pos_ee = torch.zeros((self.num_envs, 3), device=self.device)
            bbox_ee = torch.zeros((self.num_envs, 24), device=self.device)
            target_pos_ee = torch.zeros((self.num_envs, 3), device=self.device)
        
        if hasattr(self, "left_finger_force") and hasattr(self, "right_finger_force"):
            left_force = self.left_finger_force.data.net_forces_w.view(self.num_envs, 3)
            right_force = self.right_finger_force.data.net_forces_w.view(self.num_envs, 3)
            contact_force_6d = torch.cat([left_force, right_force], dim=-1)
        else:
            contact_force_6d = torch.zeros((self.num_envs, 6), device=self.device)
            
        friction_coeff = torch.full((self.num_envs, 1), 0.4, device=self.device)
        
        # Safety: Clamp contact forces to prevent physics explosion INF
        # contact_force_6d = torch.clamp(contact_force_6d, min=-100.0, max=100.0)
        
        critic_obs = torch.cat([
            jpos_full,              # 6
            jvel_full,              # 6
            last_4_actions,         # 24
            obj_pos_ee,             # 3 
            bbox_ee,                # 24
            target_pos_ee,          # 3 
            contact_force_6d,       # 6 
            friction_coeff          # 1 
        ], dim=-1)  # -> (B, 73)
        
        # ------------------- [DEBUG 爆炸偵測器 - 0318 觀測值] -------------------
        components = {
            "jpos_full": jpos_full,
            "jvel_full": jvel_full,
            "jerr": jerr,
            "last_4_actions": last_4_actions,
            "obj_pos_ee": obj_pos_ee,
            "bbox_ee": bbox_ee,
            "target_pos_ee": target_pos_ee,
            "contact_force_6d": contact_force_6d,
            "proprio_obs": proprio_obs,
            "critic_obs": critic_obs
        }
        for name, comp in components.items():
            if torch.isnan(comp).any() or torch.isinf(comp).any() or comp.abs().max() > 10000.0:
                print(f"🚨 [DEBUG DETECTOR] 0318觀測板塊 '{name}' 發生異常數值！ (Step {self.common_step_counter})")
                print(f"   --> Max: {comp.abs().max().item():.2f}, NaN: {comp.isnan().any().item()}, Inf: {comp.isinf().any().item()}")
        # ------------------------------------------------------------------------

        # Final Safety: Protect against any transient NaNs in critic_obs
        # if torch.isnan(critic_obs).any() or torch.isinf(critic_obs).any():
        #     critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if getattr(self.cfg, "use_raw_observations", False):
            # For trainable encoders, we need to provide the raw inputs in the observation.
            # We'll return a flattened vector of [Proprio(42) | RawImages(163840) | RawPt(12288) | VisionHigh(64)]
            # Note: Normalization of images (0-1) is done in _process methods, we use that.
            
            # RGBD stack from history (80x128x4) x 4 frames
            # 1. Update Raw Image History
            # Get current RGB only (80, 128, 3) since VisionEncoder expects 3 channels
            rgb_low = self.camera_low.data.output["rgb"][..., :3].float() / 255.0
            current_rgb = rgb_low.permute(0, 3, 1, 2) # (B, 3, 80, 128)
            
            self.raw_image_history_buf = torch.roll(self.raw_image_history_buf, shifts=-1, dims=1)
            self.raw_image_history_buf[:, -1, 0:3, ...] = current_rgb
            
            # Extract strided frames: t-12, t-8, t-4, t (RGB only)
            strided_raw_images = self.raw_image_history_buf[:, [0, 4, 8, 12], 0:3, ...] # (B, 4, 3, 80, 128)
            
            # 2. Update Raw PointNet History
            # ptcloud is (B, 1024, 3)
            self.raw_pt_history_buf = torch.roll(self.raw_pt_history_buf, shifts=-1, dims=1)
            self.raw_pt_history_buf[:, -1, ...] = ptcloud
            
            # Extract strided frames
            strided_raw_pts = self.raw_pt_history_buf[:, [0, 4, 8, 12], ...] # (B, 4, 1024, 3)
            
            # Modular observation (FLATTENED at top level for rsl_rl RolloutStorage compatibility)
            # We must NOT use a nested dict because rsl_rl internal RolloutStorage 
            # might not handle it correctly when allocating mini-batch buffers.
            obs = {
                "policy_proprio": proprio_obs,                     # 42
                "policy_images": strided_raw_images,               # (B, 4, 3, 80, 128)
                "policy_points": strided_raw_pts,                  # (B, 4, 1024, 3)
                "policy_high_res": vision_high_single,             # 64
                "critic": critic_obs                               # 73
            }
            
            # Final Safety: nan_to_num for all policy inputs
            # for k in ["policy_proprio", "policy_high_res", "critic"]:
            #     if torch.isnan(obs[k]).any() or torch.isinf(obs[k]).any():
            #         obs[k] = torch.nan_to_num(obs[k], nan=0.0, posinf=1.0, neginf=-1.0)
            
            # ------------------- [DEBUG 爆炸偵測器 - Raw Obs] -------------------
            for k, v in obs.items():
                if hasattr(v, 'isnan') and (v.isnan().any() or v.isinf().any() or v.abs().max() > 10000.0):
                    print(f"🚨 [CRITICAL DETECTOR] Raw Observation '{k}' 異常在 step {self.common_step_counter}!")
                    print(f"   --> Max: {v.abs().max().item():.2f}, NaN: {v.isnan().any().item()}, Inf: {v.isinf().any().item()}")
            
            # [DEBUG] One-time structure verification
            if not hasattr(self, "_obs_debug_done"):
                print(f"🚀 [DEBUG] Env _get_observations - Flat obs keys: {obs.keys()}")
                for k, v in obs.items():
                    print(f"  - {k} shape: {v.shape if hasattr(v, 'shape') else 'no shape'}")
                self._obs_debug_done = True
            # ------------------------------------------------------------------
            # Trigger Diagnostic Probe before raw obs return
            # [DEBUG] diagnostic output
            # if hasattr(self, "diagnostic_probe"):
            #    self.diagnostic_probe.update(obs, self.common_step_counter)

            return obs

        # Normal execution path (if not use_raw_observations)
        with torch.no_grad():
            proprio_obs = self.proprio_norm(proprio_obs).detach()
            vision_low_norm = self.vision_low_ln(vision_low_multiframe).detach()
            pointnet_norm = self.pointnet_ln(pointnet_history).detach()
            vision_high_norm = self.vision_high_ln(vision_high_single).detach()
            
        policy_obs = torch.cat([proprio_obs, vision_low_norm, pointnet_norm, vision_high_norm], dim=-1)
        
        # ------------------- [DEBUG 爆炸偵測器 - Policy Obs (Normalized)] -------------------
        if torch.isnan(policy_obs).any() or torch.isinf(policy_obs).any() or policy_obs.abs().max() > 10000.0:
            print(f"🚨 [DEBUG DETECTOR] 0318 policy_obs 在 Normalization 後出現異常！ (Step {self.common_step_counter})")
            print(f"   --> Max: {policy_obs.abs().max().item():.2f}, NaN: {policy_obs.isnan().any().item()}, Inf: {policy_obs.isinf().any().item()}")
            
            norm_components = {
                "proprio_obs_norm": proprio_obs,
                "vision_low_norm": vision_low_norm,
                "pointnet_norm": pointnet_norm,
                "vision_high_norm": vision_high_norm
            }
            for name, comp in norm_components.items():
                if torch.isnan(comp).any() or torch.isinf(comp).any() or comp.abs().max() > 10000.0:
                    print(f"       💥 兇手 '{name}': Max={comp.abs().max().item():.2f}, NaN={comp.isnan().any().item()}, Inf={comp.isinf().any().item()}")
        # ------------------------------------------------------------------------------------
        
        # End total timing and log if needed
        # self.perf_monitor.end_timer("_get_observations_total")
        
        # # Log performance summary periodically
        # if hasattr(self, 'common_step_counter') and self.common_step_counter % self.perf_log_interval == 0:
        #     self.perf_monitor.log_summary(
        #         step=self.common_step_counter,
        #         num_envs=self.num_envs,
        #         prefix="[AsymEnv] "
        #     )
        
        # Assemble final observation structure
        final_obs = {
            "policy": policy_obs,
            "critic": critic_obs
        }

        # Trigger Diagnostic Probe before normal return
        # [DEBUG] diagnostic output
        # if hasattr(self, "diagnostic_probe"):
        #    self.diagnostic_probe.update(final_obs, self.common_step_counter)

        return final_obs

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with numerical safety."""
        # Use base reward calculation from grandparents
        total_reward = super()._get_rewards()
        
        # [STABILITY] Final reward safety: prevent inf/nan from corrupting returns
        # if torch.isnan(total_reward).any() or torch.isinf(total_reward).any():
        #     if self.common_step_counter % 100 == 0:
        #         print(f"⚠️ [STABILITY] Env _get_rewards - Invalid total_reward detected at step {self.common_step_counter}! Applying nan_to_num.")
        #     total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # ------------------- [DEBUG 爆炸偵測器 - 總獎勵] -------------------
        if torch.isnan(total_reward).any() or torch.isinf(total_reward).any() or total_reward.abs().max() > 100000.0:
            print(f"🚨 [DEBUG DETECTOR] 0318 Env total_reward 異常！ Max: {total_reward.abs().max().item():.2f}, NaN: {total_reward.isnan().any().item()}, Inf: {total_reward.isinf().any().item()}")
        # ---------------------------------------------------------------------------
            
        return total_reward

    def _trigger_high_res_capture(self, env_ids: Optional[torch.Tensor] = None):
        """
        Trigger high-res capture and check for 0318-specific debug snapshots.
        """
        # 2. Call parent to update context features (mandatory for RL)
        super()._trigger_high_res_capture(env_ids)
        
        # 3. Add 0318-specific snapshots logic
        if getattr(self.cfg, "debug_vision_snapshot_0318_enable", False):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            
            # Find envs that haven't reached the 3-episode limit
            mask = self.snapshot_episodes_done[env_ids] < self.cfg.debug_vision_snapshot_0318_max_episodes
            eligible_envs = env_ids[mask]
            
            if len(eligible_envs) > 0:
                self._save_debug_snapshots_0318(eligible_envs)

    def _save_debug_snapshots_0318(self, env_ids: torch.Tensor):
        """
        Save high-res camera snapshots for 0318 debug version. (RGB ONLY)
        
        Args:
            env_ids: Environment indices to save snapshots for.
        """
        try:
            # Get current data from high-res camera (now synchronized via render() above)
            rgb_data = self.camera_high.data.output["rgb"] # (N, 400, 640, 4)
            
            from PIL import Image
            import numpy as np
            
            step = self.common_step_counter
            
            for env_id in env_ids:
                env_id_int = int(env_id.item())
                ep_idx = int(self.snapshot_episodes_done[env_id_int].item())
                curr_ep_len = int(self.episode_length_buf[env_id_int].item())
                
                # Create directory for this episode
                ep_dir = self.debug_snapshot_0318_dir / f"ep_{ep_idx}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                
                # Naming conversion: env_{id}_START if at reset point, else env_{id}_step_{step}
                suffix = "START" if curr_ep_len == 0 else f"step_{step:08d}"
                base_name = f"env_{env_id_int:04d}_{suffix}"
                
                # 1. Save RGB
                env_rgb = rgb_data[env_id_int].clone()
                if env_rgb.shape[-1] == 4:
                    env_rgb = env_rgb[..., :3]
                
                # Ensure values are in [0, 255]
                rgb_np = env_rgb.cpu().numpy().astype(np.uint8)
                Image.fromarray(rgb_np).save(str(ep_dir / f"{base_name}_rgb.png"))
                
                # Increment episodes done for this environment
                self.snapshot_episodes_done[env_id_int] += 1
                
            print(f"[VisionAsym0318] Saved 0318 debug snapshots for {len(env_ids)} envs (RGB only, Step 0 sync applied)")
            
        except Exception as e:
            print(f"[VisionAsym0318] Warning: Failed to save 0318 debug snapshots: {e}")
