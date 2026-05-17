# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Frozen Contact-GraspNet + FastSAM Oracle for RL Environments (0403 Version).

Runs CGN inference on a separate GPU (cuda:1) or CPU to avoid VRAM contention
with the training pipeline on cuda:0.  Called once per episode reset.

NOTE on paths:
  - Development (host):    /home/eric/isaaclab_volume/...
  - Container runtime:     /workspace/test_isaaclab/...
  The config should supply the correct path for the execution environment.
"""

import os
import sys
import time
import numpy as np
import torch
import traceback

# ---------------------------------------------------------------------------
# Lazy imports – resolve on first use so the module can be imported even
# when the CGN / FastSAM packages are not on the current PYTHONPATH yet.
# ---------------------------------------------------------------------------
_CGN_IMPORTED = False
_FASTSAM_IMPORTED = False


def _ensure_cgn_imports(cgn_root: str | None = None):
    """Add CGN to sys.path and import GraspEstimator + config_utils."""
    global _CGN_IMPORTED
    if _CGN_IMPORTED:
        return

    # Try several candidate roots
    candidates = [
        cgn_root,
        "/workspace/test_isaaclab/contact_graspnet_pytorch",
        "/home/eric/isaaclab_volume/contact_graspnet_pytorch",
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator  # noqa: F401
        from contact_graspnet_pytorch import config_utils  # noqa: F401
        _CGN_IMPORTED = True
    except ImportError as e:
        print(f"[GraspPredictor] FATAL – Cannot import CGN modules: {e}")
        raise


def _ensure_fastsam_imports(fastsam_root: str | None = None):
    """Add FastSAM to sys.path and import FastSAM."""
    global _FASTSAM_IMPORTED
    if _FASTSAM_IMPORTED:
        return

    candidates = [
        fastsam_root,
        "/workspace/test_isaaclab/FastSAM",
        "/home/eric/isaaclab_volume/FastSAM",
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from fastsam import FastSAM  # noqa: F401
        _FASTSAM_IMPORTED = True
    except ImportError as e:
        print(f"[GraspPredictor] FATAL – Cannot import FastSAM: {e}")
        raise


class GraspPredictor:
    """Frozen Contact-GraspNet + FastSAM oracle for RL environments.

    Lifecycle
    ---------
    1.  Called **once** at environment ``__init__`` – loads models onto the
        chosen device (default ``cuda:1``).
    2.  Called **once per episode reset** via ``predict()`` – runs the full
        FastSAM → CGN pipeline and returns grasps in **camera frame**.
    3.  The caller (environment) is responsible for transforming grasps from
        camera frame → world frame → object-local frame.

    Thread Safety
    -------------
    Not thread-safe.  Each environment process should own its own instance.
    """

    def __init__(
        self,
        cgn_ckpt_dir: str,
        fastsam_ckpt_path: str,
        device: str = "cuda:1",
        top_k: int = 5,
        score_threshold: float = 0.1,
        width_range: tuple = (0.005, 0.055),
        z_range: tuple = (0.1, 1.2),
        cgn_arg_configs: list | None = None,
        cgn_root: str | None = None,
        fastsam_root: str | None = None,
    ):
        """
        Args:
            cgn_ckpt_dir:      Directory containing CGN checkpoint + config.yaml.
            fastsam_ckpt_path: Path to FastSAM-x.pt weights.
            device:            Torch device string for inference (e.g. "cuda:1", "cpu").
            top_k:             Keep top-K grasps after filtering.
            score_threshold:   Minimum CGN confidence to keep a grasp.
            width_range:       (min, max) gripper width filter in metres.
            z_range:           (min, max) depth clip for point-cloud extraction.
            cgn_arg_configs:   Extra config overrides for CGN (list of "KEY:VALUE").
            cgn_root:          Optional path to the CGN package root for sys.path.
            fastsam_root:      Optional path to the FastSAM package root for sys.path.
        """
        self.device_str = device
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.width_min, self.width_max = width_range
        self.z_range = list(z_range)

        # ── Import packages ──────────────────────────────────────────────
        _ensure_cgn_imports(cgn_root)
        _ensure_fastsam_imports(fastsam_root)

        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
        from contact_graspnet_pytorch import config_utils
        from fastsam import FastSAM

        # ── Load FastSAM ─────────────────────────────────────────────────
        print(f"[GraspPredictor] Loading FastSAM from {fastsam_ckpt_path} …")
        self.fastsam = FastSAM(fastsam_ckpt_path)
        print("[GraspPredictor] FastSAM loaded.")

        # ── Load Contact-GraspNet ────────────────────────────────────────
        print(f"[GraspPredictor] Loading CGN from {cgn_ckpt_dir} on {device} …")
        if cgn_arg_configs is None:
            cgn_arg_configs = [
                "TEST.first_thres:0.05",
                "TEST.second_thres:0.05",
                "TEST.filter_thres:0.005",
            ]
        # Initialize Contact-GraspNet on the target device
        cg_cfg = config_utils.load_config(cgn_ckpt_dir, 
                                          batch_size=1, 
                                          arg_configs=cgn_arg_configs)
        self.estimator = GraspEstimator(cg_cfg, device=device)
        
        # Freezing the model
        for p in self.estimator.model.parameters():
            p.requires_grad = False
            
        print(f"[GraspPredictor] CGN loaded ({sum(p.numel() for p in self.estimator.model.parameters()) / 1e6:.1f}M params) on {device}.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        rgb_np: np.ndarray,
        depth_np: np.ndarray,
        K_np: np.ndarray,
        prompt_uv: tuple | None = None,
    ) -> tuple[list[dict], dict[int, np.ndarray], np.ndarray, np.ndarray | None]:
        """Run full FastSAM → CGN pipeline.

        Args:
            rgb_np:    (H, W, 3) uint8 RGB image.
            depth_np:  (H, W) float32 depth in metres.
            K_np:      (3, 3) float64/float32 camera intrinsic matrix.
            prompt_uv: (u, v) pixel coordinates to guide segmentation (optional).

        Returns:
            Tuple of (list_of_grasps, pc_segments, segmap, selected_mask).
        """
        t0 = time.time()

        # 1. ── Masking (FastSAM) ─────────────────────────────────────────
        segmap, masks = self._run_fastsam(rgb_np, prompt_uv=prompt_uv)
        
        # 2. ── Extract Point Clouds ──────────────────────────────────────
        pc_full, pc_segments, _ = self.estimator.extract_point_clouds(
            depth_np, K_np, segmap=segmap, rgb=rgb_np, z_range=self.z_range
        )
        t_seg = time.time()

        if pc_full is None or len(pc_full) == 0:
            return [], {}, segmap, None

        # Determine which mask was actually used (if prompt_uv was provided)
        # For simplicity, if pc_segments exists, we take the mask corresponding to obj_id 1
        selected_mask = None
        if masks is not None and len(masks) > 0:
            selected_mask = masks[0] # The prompted mask is usually the first one returned

        # 3. ── CGN inference ─────────────────────────────────────────────
        use_segments = bool(pc_segments)
        pred_grasps_cam, scores, contact_pts, gripper_openings = \
            self.estimator.predict_scene_grasps(
                pc_full,
                pc_segments=pc_segments,
                local_regions=use_segments,
                filter_grasps=use_segments,
            )
        t_cgn = time.time()

        # 4. ── Flatten per-segment results and filter ────────────────────
        all_grasps = []
        for obj_id in pred_grasps_cam:
            grasps = pred_grasps_cam[obj_id]
            obj_scores = scores[obj_id]
            obj_widths = gripper_openings[obj_id]
            obj_contacts = contact_pts[obj_id]

            if grasps is None or len(grasps) == 0:
                continue

            for i in range(len(grasps)):
                s = float(obj_scores[i]) if np.ndim(obj_scores) > 0 else float(obj_scores)
                w = float(obj_widths[i]) if np.ndim(obj_widths) > 0 else float(obj_widths)
                if s < self.score_threshold or not (self.width_min <= w <= self.width_max):
                    continue

                all_grasps.append({
                    "pose_cam": grasps[i].copy(),
                    "score": s,
                    "width": w,
                    "contact_pt": obj_contacts[i][:3].copy(),
                    "obj_id": obj_id,
                })

        # 5. ── Sort by score ─────────────────────────────────────────────
        all_grasps.sort(key=lambda g: g["score"], reverse=True)

        self._last_all_grasps = all_grasps
        self._last_pc_full = pc_full

        t_end = time.time()
        print(f"[GraspPredictor] Generated {len(all_grasps)} grasps (total={1000*(t_end-t0):.1f}ms)")
        
        return all_grasps, pc_segments, segmap, selected_mask

    def get_last_pc_full(self):
        """Internal helper for debug visualizers."""
        return getattr(self, "_last_pc_full", None), getattr(self, "_last_all_grasps", [])

    def save_visual_debug(self, pc_full, grasps, save_path="logs/raw_cgn_perception.png", benchmark_pt=None):
        """Save a professional 3D diagnostic plot of the CGN perception in Camera Frame.
        Matches the 'standard' style of the original repository (dark background, 
        colored points, multi-colored oriented grippers).
        """
        self._draw_3d_diagnostic(pc_full, grasps, save_path, benchmark_pt)

    def _draw_3d_diagnostic(self, pc_full, grasps, save_path, benchmark_pt=None):
        """
        Custom 3D Plotter with equal axis scaling and black backgrounds.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Plot Point Cloud
        if pc_full is not None and len(pc_full) > 0:
            step = max(1, len(pc_full) // 3000)
            pc_draw = pc_full[::step]
            # Use Z for color to see depth variation
            colors = pc_draw[:, 2]
            ax.scatter(pc_draw[:, 0], pc_draw[:, 2], -pc_draw[:, 1], s=2, c=colors, cmap='viridis', alpha=0.5)

        if benchmark_pt is not None:
            ax.scatter([benchmark_pt[0]], [benchmark_pt[2]], [-benchmark_pt[1]], s=250, c='red', marker='X', label='CoM Benchmark Truth')

        # 3. Plot Predicted Grasps (Professional U-shape)
        GRASP_COLORS = [
            '#FF3B30', '#4CD964', '#007AFF', '#FFCC00', '#5856D6', '#5AC8FA'
        ]
        
        num_grasps = min(len(grasps), 6)
        for i in range(num_grasps):
            g = grasps[i]
            pose = g["pose_cam"]
            color = GRASP_COLORS[i % len(GRASP_COLORS)]
            
            # Gripper geometry (Same as Visualizer)
            w_h, d_b, l_f = 0.04, 0.02, 0.05
            u_pts_local = np.array([
                [-w_h, 0, l_f], [-w_h, 0, 0], [0, 0, 0], [w_h, 0, 0], [w_h, 0, l_f]
            ])
            
            # Transform U-shape points: p_cam = R @ p_local + t
            pts_cam = (pose[:3, :3] @ u_pts_local.T).T + pose[:3, 3]
            
            # Draw U lines: 0-1, 1-2, 2-3, 3-4
            x, y, z = pts_cam[:, 0], pts_cam[:, 2], -pts_cam[:, 1]
            ax.plot(x, y, z, color=color, linewidth=3, label=f'Grasp {i}')
            
            # Mark the point of contact
            ax.scatter(pose[0,3], pose[2,3], -pose[1,3], color=color, s=20)

        # 4. Scale Fixing (Prevention of "Flat Plane" visual bug)
        if pc_full is not None and len(pc_full) > 0:
            all_pts = pc_full
            x_min, y_min, z_min = all_pts[:,0].min(), all_pts[:,2].min(), -all_pts[:,1].max()
            x_max, y_max, z_max = all_pts[:,0].max(), all_pts[:,2].max(), -all_pts[:,1].min()
            
            # Incorporate benchmark point if exists
            if benchmark_pt is not None:
                x_min = min(x_min, benchmark_pt[0])
                x_max = max(x_max, benchmark_pt[0])
                y_min = min(y_min, benchmark_pt[2])
                y_max = max(y_max, benchmark_pt[2])
                z_min = min(z_min, -benchmark_pt[1])
                z_max = max(z_max, -benchmark_pt[1])

            # Remove cubic "max_range" scaling to allow true aspect ratio (rectangular)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

        ax.set_title("Raw CGN Perception (Camera Local Frame)", color='white')
        ax.set_xlabel("X (Right)")
        ax.set_ylabel("Z (Forward)")
        ax.set_zlabel("-Y (Up)")
        plt.legend(loc='upper right')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[GraspPredictor] Saved raw perception debug to {save_path}")

    def _run_fastsam(self, rgb_np: np.ndarray, prompt_uv: tuple | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run FastSAM and return (segmap, masks)."""
        try:
            results = self.fastsam(
                rgb_np,
                device=self.device_str,
                retina_masks=True,
                imgsz=640,
                conf=0.4,
                iou=0.9,
                verbose=False,
            )
            
            from fastsam import FastSAMPrompt
            prompt = FastSAMPrompt(rgb_np, results, device=self.device_str)
            
            if prompt_uv is not None:
                # Prompt with point (u, v)
                ann = prompt.point_prompt(points=[[int(prompt_uv[0]), int(prompt_uv[1])]], pointlabel=[1])
            else:
                ann = prompt.everything_prompt()
                
            if ann is None or len(ann) == 0:
                print("[GraspPredictor] FastSAM: No masks found.")
                return None, None
            
            # Results from point_prompt may be numpy or torch depending on version
            if torch.is_tensor(ann):
                masks = ann.cpu().numpy()
            else:
                masks = np.array(ann)
            
            if masks.ndim == 2: masks = masks[None] # Ensure (N, H, W)
            print(f"[GraspPredictor] FastSAM: Found {masks.shape[0]} masks.")
            
            h, w = masks.shape[1], masks.shape[2]
            segmap = np.zeros((h, w), dtype=np.int32)
            for i, mask in enumerate(masks):
                segmap[mask > 0] = i + 1
            return segmap, masks
        except Exception as e:
            print(f"[GraspPredictor] FastSAM failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
