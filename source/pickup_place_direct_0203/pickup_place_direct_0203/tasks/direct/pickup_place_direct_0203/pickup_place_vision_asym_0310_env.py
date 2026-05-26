# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections import deque
from typing import Dict

from .pickup_place_direct_0203_vision_env import PickupPlaceDirect0203VisionEnv
from .pickup_place_vision_asym_0310_env_cfg import PickupPlaceVisionAsym0310EnvCfg
from .utils.vision_encoder import PointNetEncoder
from .mdp import observations as mdp_obs
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .utils.performance_monitor import get_perf_monitor


class PickupPlaceVisionAsym0310Env(PickupPlaceDirect0203VisionEnv):
    """
    Asymmetric Direct RL environment with dual-camera vision system and point cloud detection.
    """

    cfg: PickupPlaceVisionAsym0310EnvCfg

    def __init__(self, cfg: PickupPlaceVisionAsym0310EnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize asymmetric vision environment with Point Cloud.
        
        Args:
            cfg: Configuration for the asymmetric vision environment
            render_mode: Optional render mode for visualization
            **kwargs: Additional arguments for parent class
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # ========== MULTI-FRAME STRIDED FEATURE BUFFER ==========
        # Low-Res Camera & PointCloud: Store 13 frames for strided temporal dynamics
        # High-Res Camera: Single frame context
        
        self.history_length = 13  # t to t-12
        # Use simple tensors instead of deque for efficient batched env resetting and extraction
        self.cnn_feature_history_buf = torch.zeros((self.num_envs, self.history_length, 128), device=self.device, dtype=torch.float32)
        
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
            (self.num_envs, 5),
            device=self.device,
            dtype=torch.float32
        )
        self.use_simulated_jvel = False  # Toggle flag to switch between simulator and computed velocity
        
        # # Initialize performance monitor for tracking bottlenecks
        # self.perf_monitor = get_perf_monitor()
        # self.perf_monitor.set_device(self.device)
        # self.perf_log_interval = 100  # Log performance every N steps

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.action_history_buf[env_ids] = 0.0
            if hasattr(self, 'pt_feature_history_buf'):
                self.pt_feature_history_buf[env_ids] = 0.0
            if hasattr(self, 'cnn_feature_history_buf'):
                self.cnn_feature_history_buf[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self.action_history_buf = torch.roll(self.action_history_buf, shifts=-1, dims=1)
        self.action_history_buf[:, -1, :] = actions.clone()

    def _get_observations(self) -> dict:
        """
        Collect asymmetric observations including multi-frame CNN features and Point Cloud.
        """
        # # Start performance tracking
        # self.perf_monitor.start_timer("_get_observations_total")
        # self.perf_monitor.start_timer("parent_observations")
        
        # Get the full observations from the parent class (VisionEnv)
        # Parent returns: {"policy": combined_obs} where combined_obs is (B, 238)
        # Structure: [State(48) | Vision_Low(128) | Vision_High(64)]
        # State(48): JPos(6), JVel(6), ObjPos(3), ObjBBox(24), Target(3), Action(6)
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
        # Note: self.history_length is 13
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
            # dt = 0.01s (100Hz simulator, decimation=2 -> 50Hz control)
            self.jvel_simulated = (jpos_curr - jpos_prev) / 0.01
        
        # Get JVel from full observation (simulator's velocity)
        jvel_from_sim = full_obs[:, 6:12]  # Shape: (B, 6)
        
        # Choose which JVel to use (option: can add option to toggle in config)
        jvel_to_use = self.jvel_simulated if self.use_simulated_jvel else jvel_from_sim
        
        # ========== POINTNET FEATURE PROCESSING ==========
        # self.perf_monitor.start_timer("pointnet_extraction")
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
        valid_mask = (points_c[..., 2] > 0.2) & (points_c[..., 2] < 2.5)
        
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
        
        # User requested: jerr is previous action minus current joint position
        prev_actions = self.action_history_buf[:, -2, :] if self.action_history_buf.shape[1] >= 2 else self.actions
        jerr = prev_actions - jpos_full
        
        last_4_actions = self.action_history_buf.view(self.num_envs, -1) # (B, 24)
        
        # Concatenate all components
        policy_obs = torch.cat([
            jpos_full,              # 6
            jvel_full,              # 6
            jerr,                   # 6
            last_4_actions,         # 24
            vision_low_multiframe,  # 512
            pointnet_history,       # 512
            vision_high_single      # 64
        ], dim=-1)  # → (B, 1130)
        # self.perf_monitor.end_timer("construct_policy_obs")
        
        # ========== BUILD CRITIC OBSERVATION ==========
        # Compute EE-relative coordinates for generalized critic representation
        if hasattr(self, 'ee_frame'):
            ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]
            ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :]
            
            # Object pos world -> EE frame
            obj_pos_w = self.scene["object"].data.root_com_pose_w[:, :3]
            obj_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, obj_pos_w)
            
            # Target pos world -> EE frame
            target_pos_w = self.target_poses + self.scene.env_origins 
            target_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, target_pos_w)
            
            # BBox corners world -> EE frame
            world_corners_flat = mdp_obs.object_bbox_corners(self, SceneEntityCfg("object")).view(self.num_envs * 8, 3)
            ee_pos_rep = ee_pos_w.repeat_interleave(8, dim=0)
            ee_quat_rep = ee_quat_w.repeat_interleave(8, dim=0)
            bbox_ee_flat, _ = subtract_frame_transforms(ee_pos_rep, ee_quat_rep, world_corners_flat)
            bbox_ee = bbox_ee_flat.view(self.num_envs, 24)
        else:
            # Fallback if EE frame is not initialized yet
            obj_pos_ee = torch.zeros((self.num_envs, 3), device=self.device)
            bbox_ee = torch.zeros((self.num_envs, 24), device=self.device)
            target_pos_ee = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Requested Critic Space: [JPos | JVel | Last_4_Actions | obj_ground_truth_pos | obj_ground_truth_bbox | BasketPos | Contact_Forces | Friction_Coeff]
        dummy_contact_friction = torch.zeros((self.num_envs, 7), device=self.device)  # Reserve 7 dims for contact force and friction
        
        critic_obs = torch.cat([
            jpos_full,              # 6
            jvel_full,              # 6
            last_4_actions,         # 24
            obj_pos_ee,             # 3 (object pos relative to EE)
            bbox_ee,                # 24 (object bbox relative to EE)
            target_pos_ee,          # 3 (basket/target pos relative to EE)
            dummy_contact_friction  # 7 (Contact Forces + Friction Coeff)
        ], dim=-1)  # -> (B, 73)
        
        # End total timing and log if needed
        # self.perf_monitor.end_timer("_get_observations_total")
        
        # # Log performance summary periodically
        # if hasattr(self, 'common_step_counter') and self.common_step_counter % self.perf_log_interval == 0:
        #     self.perf_monitor.log_summary(
        #         step=self.common_step_counter,
        #         num_envs=self.num_envs,
        #         prefix="[AsymEnv] "
        #     )
        
        return {
            "policy": policy_obs,
            "critic": critic_obs
        }