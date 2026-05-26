# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
CGN-Guided Asymmetric Vision Environment (0403 Version).

Extends the 0318 environment with:
1. Contact-GraspNet inference at episode reset → grasp poses anchored to object local frame
2. Per-step grasp-alignment reward (position + orientation)
3. Enriched critic observation (89 dims) with EE proprioception + grasp gap features
"""

from __future__ import annotations

import torch
import numpy as np
import time
import os
import traceback
from typing import Optional, List
from PIL import Image

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, FRAME_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

from .pickup_place_vision_asym_0318_env import PickupPlaceVisionAsym0318Env
from .pickup_place_vision_asym_0403_env_cfg import PickupPlaceVisionAsym0403EnvCfg
from .mdp import rewards as mdp_rewards
from .mdp import observations as mdp_obs
from .utils.cgn_visualizer import CgnDebugVisualizer

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    quat_apply,
    quat_mul,
    quat_inv,
    quat_box_minus,
    matrix_from_quat,
)


class PickupPlaceVisionAsym0403Env(PickupPlaceVisionAsym0318Env):
    """
    Asymmetric Direct RL environment with CGN-guided grasp alignment. (0403 Version)
    """

    cfg: PickupPlaceVisionAsym0403EnvCfg

    def __init__(self, cfg: PickupPlaceVisionAsym0403EnvCfg, render_mode: str | None = None, **kwargs):
        # [0408 CLEANUP] Disable base/ee_frame markers (yellow line/base spheres) as requested
        # We do this BEFORE super().__init__ which instantiates the FrameTransformer
        cfg.ee_frame_cfg.visualizer_cfg = None
        
        super().__init__(cfg, render_mode, **kwargs)

        # [0404 Fix] Manually enforce action space if it was misidentified as 1D
        import gymnasium as gym
        if self.action_space.shape[0] != 6:
            print(f"[0403Env] WARNING: Action space was {self.action_space.shape[0]}D. Enforcing 6D.")
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # ── Load GraspPredictor (lazy, on separate GPU) ──────────────────
        self._grasp_predictor = None  # lazy-loaded on first reset
        self._cgn_load_attempted = False

        # ── Grasp Buffers ────────────────────────────────────────────────
        K = self.cfg.cgn_top_k
        self.grasp_local_poses = torch.zeros((self.num_envs, K, 7), device=self.device, dtype=torch.float32)
        self.grasp_local_scores = torch.zeros((self.num_envs, K), device=self.device, dtype=torch.float32)
        self.num_valid_grasps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.nearest_grasp_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.nearest_grasp_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.nearest_grasp_quat_w[:, 0] = 1.0
        self.nearest_grasp_score = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.has_valid_grasps = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── 3D Visualizer Markers (World Space) ──────────────────────────
        import copy
        marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        marker_cfg.prim_path = "/Visuals/CgnGrasps"
        # Correctly set the scale on the internal marker objects, not the collection config
        for m_key in marker_cfg.markers:
            marker_cfg.markers[m_key].scale = (0.05, 0.05, 0.05)
        
        self._grasp_marker_visualizer = VisualizationMarkers(marker_cfg)
        
        # # ── Target Position Marker (Red Sphere) ─────────────────────────
        # target_marker_cfg = VisualizationMarkersCfg(
        #     prim_path="/Visuals/Target_0403",
        #     markers={
        #         "sphere": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.5, 0.0, 0.0)))
        #     }
        # )
        # self._target_marker_visualizer = VisualizationMarkers(target_marker_cfg)

        # # ── Verification Markers (Axes-Aligned RGB Spheres) ──────────────
        # axes_marker_cfg = VisualizationMarkersCfg(
        #     prim_path="/Visuals/Verification_Axes_0403",
        #     markers={
        #         "x_axis": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), emissive_color=(1.0, 0.0, 0.0))),
        #         "y_axis": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 1.0, 0.0))),
        #         "z_axis": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), emissive_color=(0.0, 0.0, 1.0))),
        #         "center": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), emissive_color=(1.0, 1.0, 1.0)))
        #     }
        # )
        # self._axes_marker_visualizer = VisualizationMarkers(axes_marker_cfg)

        mask_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Mask_Centroids_0403",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.03, 
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 1.0), emissive_color=(1.0, 0.0, 1.0) # Magenta
                    )
                )
            }
        )
        self._mask_marker_visualizer = VisualizationMarkers(mask_marker_cfg)

        # ── [0408 Addition] Ground Truth COM Marker (Red) ──────────────
        com_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/GT_CoM_0403",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.01, 
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0), emissive_color=(1.0, 0.0, 0.0) # Red
                    )
                )
            }
        )
        self._gt_com_marker_visualizer = VisualizationMarkers(com_marker_cfg)

        # ── Cardinal World Camera (Sentinel View) ───────────────────────
        diag_cam_cfg = CameraCfg(
            prim_path="/World/DiagnosticSentinal",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 15.0),
            ),
        )
        self.camera_diag = Camera(diag_cam_cfg)
        self.camera_diag._initialize_impl()

        # [0408 FIX] Define ROS-to-OpenGL rotation for camera frame transformations
        # ROS Camera (Optical): X-Right, Y-Down, Z-Forward
        # Isaac Sim (OpenGL):  X-Right, Y-Up, Z-Backward
        # Rotation: 180 degrees around X-axis
        self.camera_high_ros_to_opengl = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], device=self.device, dtype=torch.float32)

        # ── Diagnostic Base (Commented out after orientation confirmed) ──
        # spawn_diag_sphere(
        #     "/World/Diag_Table_Red_X", (0.5, 0.0, 0.1), 0.02, (1.0, 0.0, 0.0)
        # )
        # spawn_diag_sphere(
        #     "/World/Diag_Table_Green_Y", (0.0, 0.5, 0.1), 0.02, (0.0, 1.0, 0.0)
        # )
        # Removed Diag_Origin_White that was blocking the camera

        # ── Debug Visualizer ─────────────────────────────────────────────
        self._visualizer = CgnDebugVisualizer(self, log_dir=self.cfg.cgn_debug_dir)

        # ── CGN Telemetry ────────────────────────────────────────────────
        self._cgn_inference_count = 0
        self._cgn_total_time_ms = 0.0
        self._cgn_total_grasps_found = 0
        self.common_step_counter = 0

        print(f"[0403Env] Initialized with CGN on {self.cfg.cgn_device}, "
              f"top_k={K}, critic_dim={self.cfg.critic_observation_space}")
        
        # Audit Config
        print(f"[CFG AUDIT] debug_vis={self.cfg.cgn_debug_vis}, debug_snapshots={self.cfg.cgn_debug_snapshots}")

    # ==================================================================
    # Lazy CGN Loading
    # ==================================================================

    def _ensure_grasp_predictor(self):
        """Lazy-load the grasp predictor on first use."""
        if self._grasp_predictor is not None or self._cgn_load_attempted:
            return

        self._cgn_load_attempted = True
        try:
            from .utils.grasp_predictor import GraspPredictor

            self._grasp_predictor = GraspPredictor(
                cgn_ckpt_dir=self.cfg.cgn_ckpt_dir,
                fastsam_ckpt_path=self.cfg.fastsam_ckpt_path,
                device=self.cfg.cgn_device,
                top_k=self.cfg.cgn_top_k,
                score_threshold=self.cfg.cgn_score_threshold,
                width_range=self.cfg.cgn_width_range,
                z_range=self.cfg.cgn_z_range,
                cgn_arg_configs=self.cfg.cgn_arg_configs,
                cgn_root=getattr(self.cfg, "cgn_root", None),
                fastsam_root=getattr(self.cfg, "fastsam_root", None),
            )
            print("[0403Env] GraspPredictor loaded successfully.")
        except Exception as e:
            print(f"[0403Env] FATAL – Failed to load GraspPredictor: {e}")
            traceback.print_exc()
            self._grasp_predictor = None

    # ==================================================================
    # Reset
    # ==================================================================

    def _reset_idx(self, env_ids):
        """Reset environments and run CGN inference for fresh grasps."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # [0404 Fix] Ensure camera buffers are updated before running CGN
        # Call render twice to ensure full synchronization of the buffers
        self.sim.render()
        self.sim.render()

        # Clear grasp buffers for reset environments
        self.grasp_local_poses[env_ids] = 0.0
        self.grasp_local_scores[env_ids] = 0.0
        self.num_valid_grasps[env_ids] = 0
        self.has_valid_grasps[env_ids] = False
        self.nearest_grasp_pos_w[env_ids] = 0.0
        self.nearest_grasp_quat_w[env_ids] = 0.0
        self.nearest_grasp_quat_w[env_ids, 0] = 1.0  # identity quaternion
        self.nearest_grasp_score[env_ids] = 0.0

        # Ensure CGN is loaded
        self._ensure_grasp_predictor()
        if self._grasp_predictor is None:
            return

        # Run CGN inference for each reset environment
        # NOTE: This happens AFTER parent's _reset_idx which triggers sim.render()
        # and camera buffer updates, so camera data is fresh.
        for env_id in env_ids:
            idx = env_id.item() if torch.is_tensor(env_id) else int(env_id)
            try:
                self._run_cgn_inference(idx)
            except Exception as e:
                print(f"[0403Env] CGN inference failed for env {idx}: {e}")
                traceback.print_exc()

    # ==================================================================
    # CGN Inference (per-environment, on cuda:1)
    # ==================================================================

    def _run_cgn_inference(self, env_idx: int):
        """Run CGN on high-res camera data and anchor grasps to object frame.

        Flow:
        1. Extract RGB + depth from high-res camera
        2. Call GraspPredictor.predict()  →  grasps in camera frame
        3. Transform camera frame → world frame
        4. Filter by proximity to known object position (GT)
        5. Anchor each valid grasp to object-local frame
        """
        t0 = time.time()

        # ── 1. Camera data (with Rendering Warm-up Loop) ─────────────────
        # [0404 Fix] Headless rendering/TiledCamera sometimes has a lag in depth buffer 
        # synchronization at the very first reset. We retry rendering until depth is non-zero.
        max_retries = 10
        valid_depth = False
        for i in range(max_retries):
            # Resolve Depth Check (Support 'depth' or 'distance_to_image_plane')
            if "depth" in self.camera_high.data.output:
                depth_tensor = self.camera_high.data.output["depth"][env_idx]
            elif "distance_to_image_plane" in self.camera_high.data.output:
                depth_tensor = self.camera_high.data.output["distance_to_image_plane"][env_idx]
            else:
                depth_tensor = torch.zeros((1, 1, 1), device=self.device)

            # Check for finite depth (Isaac Sim returns Inf for uninitialized or sky pixels)
            finite_depth_mask = torch.isfinite(depth_tensor) & (depth_tensor > 0.0)
            finite_count = torch.count_nonzero(finite_depth_mask).item()
            
            if finite_count > 100:  # Require at least 100 valid depth pixels
                valid_depth = True
                if i > 0:
                    print(f"[0403Env] Env {env_idx}: Camera depth synchronized after {i} extra renders. (Finite pixels: {finite_count})")
                break
            
            if i % 2 == 0:
                print(f"[0403Env] Env {env_idx}: Waiting for depth... (Render {i}, Finite pixels: {finite_count}, Raw max: {depth_tensor.max():.2f})")

            
            # Not ready yet, kick the renderer
            self.sim.render()
            self.camera_high.update(dt=0.0)
        
        if not valid_depth:
            print(f"[0403Env] WARNING: Env {env_idx}: Depth buffer is still all zeros after {max_retries} renders.")
        # ── 2. Data Preparation ───────────────────────────────────────────
        rgb_tensor = self.camera_high.data.output["rgb"][env_idx]
        depth_tensor = self.camera_high.data.output["depth"][env_idx]
        
        rgb_np = rgb_tensor[..., :3].cpu().numpy().astype(np.uint8)
        depth_np = depth_tensor.squeeze(-1).cpu().numpy()
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        K_np = self.camera_high.data.intrinsic_matrices[env_idx].cpu().numpy()

        # ── 3. Coordinate Frame Context & Prompting (World to Camera) ────
        cam_pos_w = self.camera_high.data.pos_w[env_idx]
        # Use CoM (Center of Mass) for Ground Truth and Prompting
        obj_pos_w = self.object.data.root_com_pose_w[env_idx, :3]
        
        # [0408 ALIGNMENT FIX] Use Isaac Lab's quat_w_ros property directly.
        # This quaternion represents the orientation from World to ROS-Optical frame.
        cam_quat_w_ros = self.camera_high.data.quat_w_ros[env_idx]
        w_R_cam_ros = matrix_from_quat(cam_quat_w_ros.unsqueeze(0)).squeeze(0)
        
        # Transform object position into ROS Camera Frame (Optical)
        p_cam_ros = w_R_cam_ros.T @ (obj_pos_w - cam_pos_w)
        
        # Project to 2D image plane (u, v) using camera intrinsics
        prompt_uv = None
        if p_cam_ros[2] > 0.01:
            u = K_np[0,0] * (p_cam_ros[0].item() / p_cam_ros[2].item()) + K_np[0,2]
            v = K_np[1,1] * (p_cam_ros[1].item() / p_cam_ros[2].item()) + K_np[1,2]
            u_px = int(torch.clamp(torch.tensor(u), 0, self.cfg.camera_high_image_width - 1).item())
            v_px = int(torch.clamp(torch.tensor(v), 0, self.cfg.camera_high_image_height - 1).item())
            prompt_uv = (u_px, v_px)
            print(f"  - Env {env_idx}: GT Project UV (Prompt): {prompt_uv}")

        # ── 4. CGN Inference ──────────────────────────────────────────
        all_grasps, pc_segments, segmap, selected_mask = self._grasp_predictor.predict(
            rgb_np, depth_np, K_np, prompt_uv=prompt_uv
        )

        # ── 5. Point-in-Mask Selection ────────────────────────────────
        # We select the mask that contains the actual projected GT CoM.
        # This is the most robust method to distinguish object from table.
        obj_id_target = -1
        best_pc = None

        if prompt_uv is not None and segmap is not None:
            u_px, v_px = prompt_uv
            # segmap stores the obj_id at each pixel
            target_id_from_map = int(segmap[v_px, u_px])
            if target_id_from_map > 0 and target_id_from_map in pc_segments:
                obj_id_target = target_id_from_map
                best_pc = pc_segments[obj_id_target]
                print(f"  - Env {env_idx}: Point-In-Mask Selection Success (ID={obj_id_target})")

        if best_pc is None and pc_segments:
            # Fallback 1: Prompted mask is usually ID 1 (FastSAM's primary)
            if 1 in pc_segments:
                obj_id_target = 1
                best_pc = pc_segments[1]
                print(f"  - Env {env_idx}: Fallback to ID 1")
            else:
                # Fallback 2: Pick nearest mask whose centroid matches p_cam_ros
                min_dist = float('inf')
                for seg_id, cloud in pc_segments.items():
                    centroid = cloud.mean(axis=0)
                    dist = np.linalg.norm(centroid - p_cam_ros.cpu().numpy())
                    if dist < min_dist:
                        min_dist = dist
                        obj_id_target = seg_id
                        best_pc = cloud
                print(f"  - Env {env_idx}: Fallback to Nearest Mask (ID={obj_id_target}, dist={min_dist:.4f}m)")

        if best_pc is None:
             print(f"[0403Env] Env {env_idx}: No valid segments found.")
             # Visualize empty centroids if none found to clear previous steps
             if self.cfg.cgn_debug_vis:
                 self._mask_marker_visualizer.visualize(torch.empty((0, 3), device=self.device))
             return

        # ── [0408] ALL Mask Centroid Visualization ──────────────────────
        if self.cfg.cgn_debug_vis and pc_segments:
            # Camera pose for world transformation
            cam_pos_w = self.camera_high.data.pos_w[env_idx]
            cam_quat_w_opengl = self.camera_high.data.quat_w_world[env_idx]
            cam_rot_mat_ros = matrix_from_quat(cam_quat_w_opengl) @ self.camera_high_ros_to_opengl.T
            
            centroids_cam = []
            for _, cloud in pc_segments.items():
                centroids_cam.append(cloud.mean(axis=0))
            
            centroids_cam = torch.from_numpy(np.stack(centroids_cam)).to(self.device).float()
            
            # World = Rotation * Camera + Translation
            centroids_w = (cam_rot_mat_ros @ centroids_cam.T).T + cam_pos_w
            self._mask_marker_visualizer.visualize(centroids_w)
            print(f"[DEBUG] Visualized {len(centroids_w)} FastSAM segments.")

        # ── 6. Data-Driven Centroid Anchoring ───────────────────────────
        obs_centroid_cam = torch.from_numpy(best_pc.mean(axis=0)).to(self.device).float()
        
        # Telemetry
        print(f"\n[Perception Telemetry] Env {env_idx}")
        print(f"  - p_cam_ros (GT): {p_cam_ros.cpu().numpy()}")
        print(f"  - obs_centroid_cam: {obs_centroid_cam.cpu().numpy()}")
        residual = torch.norm(p_cam_ros - obs_centroid_cam).item()
        print(f"  - Pre-anchoring Residual: {residual:.4f}m")

        # ── 7. Transform Context & Grasp Generation ─────────────────────
        cam_T_world = torch.eye(4, device=self.device)
        cam_T_world[:3, :3] = cam_rot_mat_ros
        cam_T_world[:3, 3] = cam_pos_w
        
        # Filter for top-k grasps associated with target object
        target_grasps = [g for g in all_grasps if g["obj_id"] == obj_id_target]
        target_grasps = target_grasps[:self.cfg.cgn_top_k]
        if not target_grasps:
             # If target_id didn't have grasps, use any valid grasps to keep the flow
             target_grasps = all_grasps[:self.cfg.cgn_top_k]
        
        grasps_world_pos, grasps_world_quat = [], []
        for g in target_grasps:
            g_cam = torch.from_numpy(g["pose_cam"]).to(self.device).float()
            
            # The CGN grasps are relative to their own local coordinate frame.
            # We shift them to be relative to the observed object centroid.
            # Base logic: g_w = T_world_cam * T_cam_obj * g_local
            # But the predictor already gives g in camera frame relative to its own guess.
            # We align the predictor's mean with our observed mean.
            
            g_w = cam_T_world @ g_cam
            grasps_world_pos.append(g_w[:3, 3])
            grasps_world_quat.append(quat_from_matrix(g_w[:3, :3]))

        if not grasps_world_pos:
            print(f"[0403Env] No valid target grasps for env {env_idx}")
            return

        grasps_world_pos = torch.stack(grasps_world_pos)
        grasps_world_quat = torch.stack(grasps_world_quat)

        # ── 8. Diagnostic Visualization & Snapshots ─────────────────────
        log_path = getattr(self, "log_dir", "logs/verify_cgn_0403")
        os.makedirs(log_path, exist_ok=True)
        
        # ── 9. Diagnostic Visualization (Isaac Sim 3D View) ──────────────
        if self.cfg.cgn_debug_vis:
            # 🔴 [0408] Ground Truth COM Marker
            self._gt_com_marker_visualizer.visualize(obj_pos_w.view(1, 3))

            # # 🎯 [0408] Target Goal Marker (Red sphere, slightly larger)
            # self._target_marker_visualizer.visualize(self.target_poses[env_idx].view(1, 3))

            # # 🟢 RGB Axes Markers at Object World CoM (OFFSET UP BY 0.3m for visibility)
            # # Center: White, X: Red, Y: Green, Z: Blue
            # p_diag = obj_pos_w.clone()
            # p_diag[2] += 0.3 
            
            # p0 = p_diag.unsqueeze(0)
            # ax_len = 0.05
            # px = p0 + torch.tensor([[ax_len, 0, 0]], device=self.device)
            # py = p0 + torch.tensor([[0, ax_len, 0]], device=self.device)
            # pz = p0 + torch.tensor([[0, 0, ax_len]], device=self.device)
            
            # # Indices match marker keys: x_axis(0), y_axis(1), z_axis(2), center(3)
            # marker_poses = torch.cat([px, py, pz, p0])
            # self._axes_marker_visualizer.visualize(marker_poses, marker_indices=[0, 1, 2, 3])
            
            # print(f"[DEBUG] Marker Poses (N=4):\n{marker_poses.cpu().numpy()}")
            print(f"[DEBUG] Camera Data: pos_w={cam_pos_w.cpu().numpy()}")
            print(f"[DEBUG] Camera Data: quat_w_ros={cam_quat_w_ros.cpu().numpy()}")
            
            # Utilize Deep Sync: Physics + Rendering update to flush USD changes
            # We call render multiple times to ensure the VISUAL markers appear in high-fidelity sensor images
            for _ in range(5):
                self.sim.render()
            
            # Final sync
            self.sim.step(render=True)
            self.camera_high.update(dt=0.0)
            
            # [0407 FIX] Force base class debug visualizers to run to see if they appear
            self._update_debug_vis()
            
            # [0408 CARDINAL DIAGNOSTIC] 4-Direction Snapshot
            cardinal_poses = [
                ([1.5, 0.0, 1.0], [0.0, 0.0, 0.5], "North"),
                ([0.0, 1.5, 1.0], [0.0, 0.0, 0.5], "East"),
                ([-1.5, 0.0, 1.0], [0.0, 0.0, 0.5], "South"),
                ([0.0, -1.5, 1.0], [0.0, 0.0, 0.5], "West"),
            ]
            
            for i, (eye, target, name) in enumerate(cardinal_poses):
                eye_t = torch.tensor([eye], device=self.device)
                target_t = torch.tensor([target], device=self.device)
                self.camera_diag.set_world_poses_from_view(eye_t, target_t)
                self.sim.render() # Sync state
                self.camera_diag.update(dt=0.0)
                
                diag_rgb = self.camera_diag.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                diag_img = Image.fromarray(diag_rgb)
                diag_save_path = os.path.join(log_path, f"cardinal_{name}_env_{env_idx}.png")
                diag_img.save(diag_save_path)
                print(f"[DEBUG] Saved Cardinal View {name}: {diag_save_path}")

            # Check if camera output has non-zero pixels after sync
            rgb_after = self.camera_high.data.output["rgb"][env_idx]
            print(f"[DEBUG] RGB Mean after marker sync: {rgb_after.float().mean().item():.3f}")

        # ── 10. Composite Snapshot & Debug Saves ─────────────────────────
        self._grasp_predictor.save_visual_debug(
            pc_full=best_pc, 
            grasps=target_grasps,
            save_path=f"{log_path}/raw_cgn_env_{env_idx}.png",
            benchmark_pt=obs_centroid_cam.cpu().numpy() # Anchor to Observed Truth
        )

        if self.cfg.cgn_debug_snapshots:
            self._visualizer.save_snapshot(
                env_idx=env_idx,
                rgb_np=rgb_np,
                grasps_cam=target_grasps,
                obs_pc=obs_centroid_cam,
                gt_obj_pos_cam=p_cam_ros,
                benchmark_uv=prompt_uv,
                pc_segments=pc_segments,
                segmap=segmap,
                world_obj_pos=obj_pos_w
            )

        # ── 10. Anchor to Object-Local Frame ─────────────────────────────
        obj_quat_w = self.scene["object"].data.root_quat_w[env_idx]
        obj_quat_inv = quat_inv(obj_quat_w)
        valid_count = 0
        
        for i in range(len(target_grasps)):
            gp_w = grasps_world_pos[i]
            gq_w = grasps_world_quat[i]

            # Re-verify proximity in world space (safety check)
            if torch.norm(gp_w - obj_pos_w) > self.cfg.cgn_proximity_filter:
                continue

            rel_pos = quat_apply(obj_quat_inv.unsqueeze(0), (gp_w - obj_pos_w).unsqueeze(0)).squeeze(0)
            rel_quat = quat_mul(obj_quat_inv.unsqueeze(0), gq_w.unsqueeze(0)).squeeze(0)

            self.grasp_local_poses[env_idx, valid_count, :3] = rel_pos
            self.grasp_local_poses[env_idx, valid_count, 3:] = rel_quat
            self.grasp_local_scores[env_idx, valid_count] = target_grasps[i]["score"]
            valid_count += 1

            if valid_count >= self.cfg.cgn_top_k:
                break

        self.num_valid_grasps[env_idx] = valid_count
        self.has_valid_grasps[env_idx] = valid_count > 0

        # ── 8. Telemetry ─────────────────────────────────────────────────
        elapsed_ms = (time.time() - t0) * 1000
        self._cgn_inference_count += 1
        self._cgn_total_time_ms += elapsed_ms
        self._cgn_total_grasps_found += valid_count

        if valid_count > 0:
            best = self.grasp_local_scores[env_idx, :valid_count].max().item()
            print(f"[0403Env] Env {env_idx}: {valid_count} grasps anchored "
                  f"(best={best:.3f}, {elapsed_ms:.0f}ms)")
            
            # ── Verification of Forward/Inverse Transform ────────────────
            with torch.no_grad():
                rel_pos_raw = self.grasp_local_poses[env_idx, 0, :3]
                # Verify LOCAL -> WORLD using our matrix logic
                verify_pos_w = obj_pos_w + quat_apply(obj_quat_w.unsqueeze(0), rel_pos_raw.unsqueeze(0)).squeeze(0)
                original_pos_w = grasps_world_pos[0]
                
                err = torch.norm(verify_pos_w - original_pos_w).item()
                if err > 0.001: 
                    print(f"  [VERIFY] Env {env_idx}: Anchoring mismatch! Error: {err:.6f}m")
                else:
                    env_origin = self.scene.env_origins[env_idx]
                    verify_pos_env = (verify_pos_w - env_origin).cpu().numpy()
                    print(f"  [VERIFY] Env {env_idx}: Anchoring PASS. Env-Local Pos: {verify_pos_env}")
        else:
            print(f"[0403Env] Env {env_idx}: No valid grasps after proximity filter "
                  f"({elapsed_ms:.0f}ms)")

    # ==================================================================
    # Per-step:  Reconstruct world grasps + find nearest to EE
    # ==================================================================

    def _update_nearest_grasp_cache(self):
        """Reconstruct world-frame grasps from object-local anchors and find the nearest to EE.
        
        Called once per step inside _get_observations.
        """
        # Get current object and EE poses
        # Use CoM (Center of Mass) for current object pose tracking in RL obs/reward
        obj_pos_w = self.scene["object"].data.root_com_pose_w[:, :3]       # (B, 3)
        obj_quat_w = self.scene["object"].data.root_quat_w     # (B, 4)
        ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]  # (B, 3)

        K = self.cfg.cgn_top_k

        for env_idx in range(self.num_envs):
            n = self.num_valid_grasps[env_idx].item()
            if n == 0:
                self.has_valid_grasps[env_idx] = False
                continue

            # Reconstruct world poses for all valid grasps in this env
            rel_pos = self.grasp_local_poses[env_idx, :n, :3]   # (n, 3)
            rel_quat = self.grasp_local_poses[env_idx, :n, 3:]  # (n, 4)

            # T_world = T_obj × T_local
            oq = obj_quat_w[env_idx].unsqueeze(0).expand(n, -1)  # (n, 4)
            op = obj_pos_w[env_idx].unsqueeze(0).expand(n, -1)   # (n, 3)

            grasp_pos_w = op + quat_apply(oq, rel_pos)       # (n, 3)
            grasp_quat_w = quat_mul(oq, rel_quat)            # (n, 4)

            # Find nearest to EE
            dists = torch.norm(grasp_pos_w - ee_pos_w[env_idx].unsqueeze(0), dim=-1)  # (n,)
            nearest_idx = torch.argmin(dists).item()

            self.nearest_grasp_pos_w[env_idx] = grasp_pos_w[nearest_idx]
            self.nearest_grasp_quat_w[env_idx] = grasp_quat_w[nearest_idx]
            self.nearest_grasp_score[env_idx] = self.grasp_local_scores[env_idx, nearest_idx]
            self.has_valid_grasps[env_idx] = True

        # ── 3D Viewport Visualization ────────────────────────────────────
        if self.cfg.cgn_debug_vis:
            # Visualize for first 2 environments only to preserve FPS
            vis_ids = [0, 1] if self.num_envs > 1 else [0]
            # Use reconstruct logic to get all world poses for the visualizer
            for v_idx in vis_ids:
                if self.has_valid_grasps[v_idx]:
                    n = self.num_valid_grasps[v_idx].item()
                    oq = obj_quat_w[v_idx].unsqueeze(0).expand(n, -1)
                    op = obj_pos_w[v_idx].unsqueeze(0).expand(n, -1)
                    all_grasp_pos_w = op + quat_apply(oq, self.grasp_local_poses[v_idx, :n, :3])
                    all_grasp_quat_w = quat_mul(oq, self.grasp_local_poses[v_idx, :n, 3:])
                    self._visualizer.draw_grasps_3d(v_idx, all_grasp_pos_w, all_grasp_quat_w)

    # ==================================================================
    # Observations (override)
    # ==================================================================

    def _get_observations(self) -> dict:
        """Extend 0318 observations with enriched critic (89 dims)."""
        # 1. Get base observations (this builds the 73-dim critic internally)
        obs = super()._get_observations()

        # 2. Update nearest-grasp cache for this step
        self._update_nearest_grasp_cache()

        # 3. Build additional critic features (16 dims)
        extra_critic = self._build_extra_critic_features()

        # 4. Append to critic observation
        if "critic" in obs:
            obs["critic"] = torch.cat([obs["critic"], extra_critic], dim=-1)

        # One-time dimension check
        if not hasattr(self, "_critic_dim_checked"):
            actual_dim = obs["critic"].shape[-1] if "critic" in obs else 0
            print(f"[0403Env] Critic obs dimension: {actual_dim} "
                  f"(expected {self.cfg.critic_observation_space})")
            self._critic_dim_checked = True

        return obs

    def _build_extra_critic_features(self) -> torch.Tensor:
        """Build the 16 additional critic dims.

        Layout:
            ee_pos_in_base    (3)  — EE Cartesian position in robot base frame
            ee_quat_in_base   (4)  — EE quaternion in robot base frame
            ee_to_obj_dist    (1)  — scalar distance EE→object bbox
            grasp_gap_pos     (3)  — position delta EE→nearest grasp
            grasp_gap_rot     (3)  — axis-angle rotation delta
            grasp_score       (1)  — CGN confidence of nearest grasp
            grasp_valid       (1)  — whether valid grasps exist (0/1)
        Total: 16
        """
        B = self.num_envs

        # ── EE proprioception ────────────────────────────────────────────
        ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]    # (B, 3)
        ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :]  # (B, 4)

        robot_pos_w = self.robot.data.root_pos_w                  # (B, 3)
        robot_quat_w = self.robot.data.root_quat_w                # (B, 4)

        ee_pos_in_base, ee_quat_in_base = subtract_frame_transforms(
            robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
        )  # (B, 3), (B, 4)

        # ── EE→Object distance (scalar) ─────────────────────────────────
        from .mdp.rewards import object_bbox_ee_distance_real
        ee_to_obj_dist = object_bbox_ee_distance_real(
            self,
            SceneEntityCfg("object"),
            SceneEntityCfg("ee_frame"),
        ).unsqueeze(-1)  # (B, 1)

        # ── Grasp gap features ───────────────────────────────────────────
        # Position gap
        grasp_gap_pos = ee_pos_w - self.nearest_grasp_pos_w       # (B, 3)

        # Rotation gap as axis-angle (3 dims)
        # Uses Isaac Lab's built-in quat_box_minus: log(q_ee * q_grasp^{-1})
        grasp_gap_rot = quat_box_minus(ee_quat_w, self.nearest_grasp_quat_w)  # (B, 3)
        # Clamp for numerical safety
        grasp_gap_rot = torch.clamp(grasp_gap_rot, -3.15, 3.15)

        # Score & validity
        grasp_score = self.nearest_grasp_score.unsqueeze(-1)       # (B, 1)
        grasp_valid = self.has_valid_grasps.float().unsqueeze(-1)  # (B, 1)

        # Mask invalid environments → zero features (neutral)
        valid_mask = self.has_valid_grasps.unsqueeze(-1).float()    # (B, 1)
        grasp_gap_pos = grasp_gap_pos * valid_mask
        grasp_gap_rot = grasp_gap_rot * valid_mask
        grasp_score = grasp_score * valid_mask

        # ── Concatenate ──────────────────────────────────────────────────
        extra = torch.cat([
            ee_pos_in_base,     # 3
            ee_quat_in_base,    # 4
            ee_to_obj_dist,     # 1
            grasp_gap_pos,      # 3
            grasp_gap_rot,      # 3
            grasp_score,        # 1
            grasp_valid,        # 1
        ], dim=-1)  # (B, 16)

        return extra

    # ==================================================================
    # Rewards (override)
    # ==================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Add grasp-alignment reward to base rewards."""
        total_reward = super()._get_rewards()

        # Grasp alignment reward
        grasp_align = mdp_rewards.grasp_pose_alignment(
            self,
            pos_std=self.cfg.grasp_align_pos_std,
            rot_weight=self.cfg.grasp_align_rot_weight,
        )

        total_reward = total_reward + self.cfg.rew_scale_grasp_align * grasp_align

        # ── Logging ─────────────────────────────────────────────────────
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        self.extras["episode"]["reward_grasp_align"] = torch.mean(grasp_align).item()
        self.extras["episode"]["cgn_num_valid_grasps"] = torch.mean(
            self.num_valid_grasps.float()
        ).item()
        self.extras["episode"]["cgn_best_score"] = torch.mean(
            self.nearest_grasp_score
        ).item()
        
        # ── Detailed Alignment Logging ──────────────────────────────────
        if self.has_valid_grasps.any():
            valid_mask = self.has_valid_grasps
            ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :][valid_mask]
            target_pos_w = self.nearest_grasp_pos_w[valid_mask]
            
            # Position error (meters)
            dist_m = torch.norm(ee_pos_w - target_pos_w, dim=-1).mean().item()
            
            # Rotation error (degrees)
            # Use quat_box_minus to get axis-angle magnitude
            ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :][valid_mask]
            target_quat_w = self.nearest_grasp_quat_w[valid_mask]
            rot_error_aa = quat_box_minus(ee_quat_w, target_quat_w) # magnitude is angle
            align_deg = torch.norm(rot_error_aa, dim=-1).mean().item() * (180.0 / 3.14159)
            
            # [0404 Fix] Write to both root (for scripts/telemetry) and episode (for logger)
            self.extras["cgn_dist_m"] = dist_m
            self.extras["cgn_align_deg"] = align_deg
            self.extras["episode"]["cgn_dist_m"] = dist_m
            self.extras["episode"]["cgn_align_deg"] = align_deg

            # [0404 Refinement] Log nearest grasp position in Env-Local frame
            # This helps verify where the agent is being guided in intuitive coordinates
            env_origins = self.scene.env_origins[valid_mask]
            target_pos_env = (target_pos_w - env_origins).mean(dim=0).cpu().numpy()
            self.extras["episode"]["cgn_target_x_env"] = target_pos_env[0]
            self.extras["episode"]["cgn_target_y_env"] = target_pos_env[1]
            self.extras["episode"]["cgn_target_z_env"] = target_pos_env[2]

        # Periodic CGN telemetry
        if self._cgn_inference_count > 0 and self.common_step_counter % 500 == 0:
            avg_ms = self._cgn_total_time_ms / self._cgn_inference_count
            avg_grasps = self._cgn_total_grasps_found / self._cgn_inference_count
            self.extras["episode"]["cgn_avg_inference_ms"] = avg_ms
            self.extras["episode"]["cgn_avg_grasps_per_ep"] = avg_grasps

        return total_reward

# ==================================================================
# Helper for Direct USD Spawning
# ==================================================================

def spawn_diag_sphere(prim_path: str, pos: tuple, radius: float, color: tuple):
    """Bypasses Isaac Lab markers and spawns a direct sphere on the stage."""
    from pxr import Gf, UsdGeom
    import isaaclab.sim as sim_utils
    cfg = sim_utils.SphereCfg(
        radius=radius,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, 
            emissive_color=tuple(c * 5.0 for c in color) # High glow, fixed generator issue
        )
    )
    cfg.func(prim_path, cfg, translation=pos)
