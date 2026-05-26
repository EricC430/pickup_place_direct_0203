# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections import deque
from typing import Dict

from .pickup_place_direct_0203_vision_env import PickupPlaceDirect0203VisionEnv
from .pickup_place_vision_asym_0313_env_cfg import PickupPlaceVisionAsym0313EnvCfg
from .utils.vision_encoder import PointNetEncoder
from .mdp import observations as mdp_obs
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .utils.performance_monitor import get_perf_monitor


class PickupPlaceVisionAsym0313Env(PickupPlaceDirect0203VisionEnv):
    """
    Asymmetric Direct RL environment with dual-camera vision system and point cloud detection. (0313 Version)
    """

    cfg: PickupPlaceVisionAsym0313EnvCfg

    def __init__(self, cfg: PickupPlaceVisionAsym0313EnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize asymmetric vision environment with Point Cloud.
        
        Args:
            cfg: Configuration for the asymmetric vision environment
            render_mode: Optional render mode for visualization
            **kwargs: Additional arguments for parent class
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        print("\n" + "!"*50)
        print(f"🚀 [DEBUG] PickupPlaceVisionAsym0313Env __init__")
        print(f"🚀 [DEBUG] Config Randomize: {self.cfg.randomize_arm_init}")
        print(f"🚀 [DEBUG] Joint3 Range: {self.cfg.arm_init_offset_range['joint3']}")
        print("!"*50 + "\n")
        
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
        
        # High-res context cache (single frame per episode, not multi-frame)
        self.high_res_context_single = None
        self.high_res_context_valid = False
        
        # Pre-allocate buffer: (num_envs, num_frames, feature_dim)
        # Will be filled on first call to _get_observations()
        self.cnn_features_stacked = None
        
        # Action History Buffer 
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
        
        # # Initialize performance monitor for tracking bottlenecks
        # self.perf_monitor = get_perf_monitor()
        # self.perf_monitor.set_device(self.device)
        # self.perf_log_interval = 100  # Log performance every N steps

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.action_history_buf[env_ids] = 0.0
            self.pt_feature_history_buf[env_ids] = 0.0
            # [Fix] Reset high-res cache flag so context is re-captured for new episode
            self.high_res_context_valid = False

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
        
        # [STABILITY] 防止關節速度與位置的物理爆炸毒化神經網路
        jvel_full = torch.clamp(jvel_full, min=-50.0, max=50.0)
        jpos_full = torch.clamp(jpos_full, min=-50.0, max=50.0)
        
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

        # =====================================================================
        # [TEST EXPLOSION 實驗 1] 模擬 0318 版本 ActorCritic 中的 LayerNorm
        # 未限制的 jvel 物理爆炸在計算變異數 (平方) 時會超出 float32 上限 (1e38) -> Inf
        # import torch.nn.functional as F
        # policy_obs = F.layer_norm(policy_obs, [1130])
        # =====================================================================
        
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
        # Extract object contact force (Force exerted on object by robot)
        # if hasattr(self, "left_finger_force") and hasattr(self, "right_finger_force"):
        #     # 分別抓取左邊和右邊的 3 維力量 (X, Y, Z)
        #     left_force = self.left_finger_force.data.net_forces_w.view(self.num_envs, 3)
        #     right_force = self.right_finger_force.data.net_forces_w.view(self.num_envs, 3)
            
        #     # 把它們在最後一個維度拼起來，變成 6 維 (X_l, Y_l, Z_l, X_r, Y_r, Z_r)
        #     contact_force_6d = torch.cat([left_force, right_force], dim=-1)
        # else:
        contact_force_6d = torch.zeros((self.num_envs, 6), device=self.device)
            
        # 將預設摩擦係數設定為合理的物理數值 0.4
        friction_coeff = torch.full((self.num_envs, 1), 0.4, device=self.device) # config
        
        critic_obs = torch.cat([
            jpos_full,              # 6
            jvel_full,              # 6
            last_4_actions,         # 24
            obj_pos_ee,             # 3 (object pos relative to EE)
            bbox_ee,                # 24 (object bbox relative to EE)
            target_pos_ee,          # 3 (basket/target pos relative to EE)
            contact_force_6d,       # 6 (3D Left + 3D Right)
            friction_coeff          # 1 (Friction Coefficient)
        ], dim=-1)  # -> (B, 73)
        
        # # ------------------- [DEBUG 爆炸偵測器 - 觀測值] -------------------
        # if torch.isnan(policy_obs).any() or torch.isinf(policy_obs).any() or policy_obs.abs().max() > 1000.0:
        #     print(f"🚨 [DEBUG DETECTOR] policy_obs 在進入防護前就爆炸了！ (Step {self.common_step_counter})")
        #     print(f"   --> Max: {policy_obs.abs().max().item():.2f}, NaN: {policy_obs.isnan().any().item()}, Inf: {policy_obs.isinf().any().item()}")
            
        #     # 逐一盤查是哪一個特徵板塊爆炸了
        #     components = {
        #         "jpos_full": jpos_full, "jvel_full": jvel_full, "jerr": jerr, 
        #         "last_4_actions": last_4_actions, "vision_low_multiframe": vision_low_multiframe, 
        #         "pointnet_history": pointnet_history, "vision_high_single": vision_high_single
        #     }
        #     for name, comp in components.items():
        #         if torch.isnan(comp).any() or torch.isinf(comp).any() or comp.abs().max() > 1000.0:
        #             print(f"       💥 抓到兇手 '{name}': Max={comp.abs().max().item():.2f}, NaN={comp.isnan().any().item()}, Inf={comp.isinf().any().item()}")

        # if torch.isnan(critic_obs).any() or torch.isinf(critic_obs).any() or critic_obs.abs().max() > 1000.0:
        #     print(f"🚨 [DEBUG DETECTOR] critic_obs 發生爆炸！ Max: {critic_obs.abs().max().item():.2f}")
        # # ------------------------------------------------------------------------

        # [STABILITY] 進入 Critic 之前的最後一道防線
        #critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        critic_obs = torch.clamp(critic_obs, min=-100.0, max=100.0)
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # [STABILITY] 進入 Policy 之前的最後一道防線 (徹底阻絕 NaN 毒化網路)
        #policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        policy_obs = torch.clamp(policy_obs, min=-100.0, max=100.0)
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
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

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with numerical safety."""
        # Use base reward calculation from grandparents
        total_reward = super()._get_rewards()
        
        # [STABILITY] 防止 JVel^2 懲罰變成 1e40 -> Inf -> Value Loss Inf
        # # ------------------- [DEBUG 爆炸偵測器 - 總獎勵 (0313)] -------------------
        # if torch.isnan(total_reward).any() or torch.isinf(total_reward).any() or total_reward.abs().max() > 1e4:
        #     print(f"🚨 [DEBUG DETECTOR] 0313 Env total_reward 爆炸！ Max: {total_reward.abs().max().item():.2f}, NaN: {total_reward.isnan().any().item()}")
        # # ---------------------------------------------------------------------------

        # # [STABILITY] 數值防護 (先 nan_to_num 把 NaN 洗掉，再用 clamp 限制)
        # total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=100.0, neginf=-100.0)
        total_reward = torch.clamp(total_reward, min=-100.0, max=100.0)
        if torch.isnan(total_reward).any() or torch.isinf(total_reward).any():
            total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=100.0, neginf=-100.0)
            
        return total_reward