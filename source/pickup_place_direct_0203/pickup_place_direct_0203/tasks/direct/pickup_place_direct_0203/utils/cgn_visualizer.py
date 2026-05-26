# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw
try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    _debug_draw = None

class CgnDebugVisualizer:
    """Helper for visualizing Contact-GraspNet predictions in 3D and 2D."""
    
    def __init__(self, env, log_dir="logs/cgn_debug"):
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._draw = None
        if _debug_draw is not None:
            self._draw = _debug_draw.acquire_debug_draw_interface()
            
        self.snapshot_counts = {} # env_idx -> count
        
    def draw_grasps_3d(self, env_ids, grasp_pos_w, grasp_quat_w, axis_length=0.05):
        """Draw RGB axes for grasps in the world frame.
        
        Args:
            env_ids: Sequence of environment indices to visualize.
            grasp_pos_w: World positions (num_envs, 3) or (num_envs, K, 3).
            grasp_quat_w: World quaternions (num_envs, 4) or (num_envs, K, 4).
        """
        if self._draw is None:
            return

        # Prepare line lists
        p1_list = []
        p2_list = []
        colors = []
        
        # Identity axes in local frame
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.env.device) * axis_length
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.env.device) * axis_length
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.env.device) * axis_length
        
        from isaaclab.utils.math import quat_apply
        
        # Flatten grasp_pos_w and grasp_quat_w if they have K dimension
        # grasp_pos_w: (num_envs, 3) or (num_envs, K, 3)
        if grasp_pos_w.ndim == 3:
            B, K, _ = grasp_pos_w.shape
            pos = grasp_pos_w.view(-1, 3)
            quat = grasp_quat_w.view(-1, 4)
        else:
            pos = grasp_pos_w
            quat = grasp_quat_w

        # Rotate axes to world frame
        # quat_apply expects (N, 4) and (N, 3) or (4,) and (3,)
        # We need to broadcast axis tensors
        x_w = quat_apply(quat, x_axis.expand(pos.shape[0], 3))
        y_w = quat_apply(quat, y_axis.expand(pos.shape[0], 3))
        z_w = quat_apply(quat, z_axis.expand(pos.shape[0], 3))
        
        # Build line segments for current step
        # X-axis (Red)
        p1_list.extend(pos.tolist())
        p2_list.extend((pos + x_w).tolist())
        colors.extend([(1, 0, 0, 1)] * pos.shape[0])
        
        # Y-axis (Green)
        p1_list.extend(pos.tolist())
        p2_list.extend((pos + y_w).tolist())
        colors.extend([(0, 1, 0, 1)] * pos.shape[0])
        
        # Z-axis (Blue)
        p1_list.extend(pos.tolist())
        p2_list.extend((pos + z_w).tolist())
        colors.extend([(0, 0, 1, 1)] * pos.shape[0])
        
        # Clear previous markers (optional, but debug_draw usually accumulates unless cleared)
        # However, for per-step draw, we often just draw and let it expire or clear manully
        # Here we use draw_lines which is persistent until frame end usually? No, it's persistent.
        # So we usually use clear_lines()
        self._draw.clear_lines()
        self._draw.draw_lines(p1_list, p2_list, colors, [2.0] * len(p1_list))

    def save_snapshot(
        self, env_idx, rgb_np, grasps_cam, 
        obs_pc=None, gt_obj_pos_cam=None, benchmark_uv=None, 
        mask_np=None, world_obj_pos=None, pc_segments=None, segmap=None
    ):
        """Save RGB image with 2D projections of grasps, centroids, and masks.
        
        Args:
            env_idx: environment index.
            rgb_np: (H, W, 3) uint8 RGB image.
            grasps_cam: List of grasp dicts with 'pose_cam'.
            obs_pc: (N, 3) tensor of observed segmented points in camera frame.
            gt_obj_pos_cam: (3,) tensor of ground truth object center in camera frame.
            benchmark_uv: (2,) optional pixel coordinate [u, v] for truth check.
            mask_np: (H, W) optional binary mask to overlay.
            world_obj_pos: (3,) optional world position for axis verification.
            pc_segments: dict[int, np.ndarray] map of obj_id to point clouds.
            segmap: (H, W) full segmentation map from FastSAM.
        """
        img = Image.fromarray(rgb_np).convert("RGBA")
        overlay_img = Image.new("RGBA", img.size, (0, 0, 0, 0)) # Alpha plane
        
        # Color Palette for Masks/Centroids
        MASK_COLORS = [
            (0, 255, 255, 80),   # Cyan
            (255, 0, 255, 80),   # Magenta
            (255, 255, 0, 80),   # Yellow
            (0, 255, 0, 80),     # Green
            (255, 100, 0, 80),   # Orange
            (100, 0, 255, 80),   # Purple
        ]
        
        # ── 1. Draw Multi-Segment Mask Overlay ──────────────────────────
        if segmap is not None:
             # Iterate through all unique IDs in segmap (excluding background 0)
             unique_ids = np.unique(segmap)
             for i, uid in enumerate(unique_ids):
                 if uid == 0: continue
                 color = MASK_COLORS[i % len(MASK_COLORS)]
                 rows, cols = np.where(segmap == uid)
                 for r, c in zip(rows, cols):
                     overlay_img.putpixel((c, r), color)
        elif mask_np is not None:
             # Fallback to single binary mask
             rows, cols = np.where(mask_np > 0)
             for r, c in zip(rows, cols):
                 overlay_img.putpixel((c, r), MASK_COLORS[0])
        
        img = Image.alpha_composite(img, overlay_img)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Projection parameters
        K = self.env.camera_high.data.intrinsic_matrices[env_idx].cpu().numpy()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # ── 2. Draw Ground Truth Object Center (Yellow Circle) ──────────
        if gt_obj_pos_cam is not None:
            pos_gt = gt_obj_pos_cam.cpu().numpy() if hasattr(gt_obj_pos_cam, "cpu") else np.array(gt_obj_pos_cam)
            if pos_gt[2] > 0.01:
                u_gt = (fx * pos_gt[0] / pos_gt[2]) + cx
                v_gt = (fy * pos_gt[1] / pos_gt[2]) + cy
                draw.ellipse((u_gt-8, v_gt-8, u_gt+8, v_gt+8), outline="yellow", width=4)
                draw.text((u_gt+15, v_gt+15), "GT_COM", fill="yellow")

        # ── 3. Draw Prompt Marker (Red Cross) ───────────────────────────
        if benchmark_uv is not None:
            u, v = benchmark_uv
            sz = 15
            draw.line([u-sz, v, u+sz, v], fill="red", width=3)
            draw.line([u, v-sz, u, v+sz], fill="red", width=3)
            draw.text((u+18, v-18), "PROMPT", fill="red")

        # ── 4. Draw Per-Mask Centroids (Crosses) ───────────────────────
        if pc_segments is not None:
            for i, (uid, pc) in enumerate(pc_segments.items()):
                color_hex = "#%02x%02x%02x" % MASK_COLORS[i % len(MASK_COLORS)][:3]
                # Use median for robustness
                centroid = np.median(pc, axis=0)
                if centroid[2] > 0.01:
                    u = (fx * centroid[0] / centroid[2]) + cx
                    v = (fy * centroid[1] / centroid[2]) + cy
                    sz = 8
                    draw.line([u-sz, v, u+sz, v], fill=color_hex, width=2)
                    draw.line([u, v-sz, u, v+sz], fill=color_hex, width=2)
                    draw.text((u+10, v+10), f"OBS_{uid}", fill=color_hex)
        elif obs_pc is not None and len(obs_pc) > 0:
             # Fallback to single obs_pc
             pc_np = obs_pc.cpu().numpy() if hasattr(obs_pc, "cpu") else obs_pc
             centroid = np.median(pc_np, axis=0)
             if centroid[2] > 0.01:
                 u = (fx * centroid[0] / centroid[2]) + cx
                 v = (fy * centroid[1] / centroid[2]) + cy
                 draw.line([u-5, v, u+5, v], fill="cyan", width=2)
                 draw.line([u, v-5, u, v+5], fill="cyan", width=2)
                 draw.text((u+8, v+8), "OBS_PC", fill="cyan")

        # ── 6. World Axis Projection Overlay (Diagnostic Backup) ─────────
        if world_obj_pos is not None:
            from isaaclab.utils.math import matrix_from_quat
            
            # Fetch Camera Pose
            cam_pos_w = self.env.camera_high.data.pos_w[env_idx]
            quat_w_gl = self.env.camera_high.data.quat_w_opengl[env_idx]
            R_wc = matrix_from_quat(quat_w_gl.unsqueeze(0)).squeeze(0)
            
            # Points to project: [Origin, +X, +Y, +Z]
            p0_w = world_obj_pos
            px_w = p0_w + torch.tensor([0.1, 0, 0], device=self.env.device)
            py_w = p0_w + torch.tensor([0, 0.1, 0], device=self.env.device)
            pz_w = p0_w + torch.tensor([0, 0, 0.1], device=self.env.device)
            
            pts_w = torch.stack([p0_w, px_w, py_w, pz_w])
            uvs_axes = []
            
            for pt_w in pts_w:
                # Transform to OpenGL Camera Frame: p_cam = R^T (p_w - c_w)
                p_gl = R_wc.T @ (pt_w - cam_pos_w)
                # Map to ROS: X_ros=X_gl, Y_ros=-Y_gl, Z_ros=Z_gl
                p_ros = torch.stack([p_gl[0], -p_gl[1], p_gl[2]])
                
                # Project
                if p_ros[2] > 0.01:
                    u = (fx * p_ros[0].item() / p_ros[2].item()) + cx
                    v = (fy * p_ros[1].item() / p_ros[2].item()) + cy
                    uvs_axes.append((u, v))
                else:
                    uvs_axes.append(None)
            
            # Draw Lines: Origin(0) -> X(1) Red, Y(2) Green, Z(3) Blue
            if uvs_axes[0]:
                u0, v0 = uvs_axes[0]
                colors = ["red", "lime", "blue"] # Use lime for visibility
                labels = ["+X_w", "+Y_w", "+Z_w"]
                for i in range(1, 4):
                    if uvs_axes[i]:
                        u_idx, v_idx = uvs_axes[i]
                        draw.line([u0, v0, u_idx, v_idx], fill=colors[i-1], width=4)
                        draw.text((u_idx, v_idx), labels[i-1], fill=colors[i-1])

        # ── 7. Save to Disk ──────────────────────────────────────────────
        ep_count = self.snapshot_counts.get(env_idx, 0)
        save_path = self.log_dir / f"reset_env_{env_idx}_ep_{ep_count}.png"
        img.save(save_path)
        print(f"[CgnDebugVisualizer] Saved snapshot: {save_path}")
        self.snapshot_counts[env_idx] = ep_count + 1
